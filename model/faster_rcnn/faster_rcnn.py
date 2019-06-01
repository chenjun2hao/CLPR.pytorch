import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.ocr_roi_pooling.ocr_roi_pooing import roi_pooling
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from warpctc_pytorch import CTCLoss
from ocr_dataloader.ocrdataloader import ImgDataset
import math


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_ocr_roi_pooling = roi_pooling(2)                                      # ocr roi_pooling

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()
        # rnn初始化，隐藏节点256
        nh = 256
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh))

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois//RPN网络做目标区域的建议
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)          # RPN网络

        # if it is training phrase, then use ground trubut bboxes for refining//池化的时候，应该将IOU最大的设为正样本
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)             # 输入的rois为2000×4, 经过筛选之后，输出为256×4
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data                       # rcnn部分就是一个分类器和一个回归器

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        cfg.POOLING_MODE = 'ocr'
        # cfg.POOLING_MODE = 'pool'
        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
        elif cfg.POOLING_MODE == 'ocr':
            pooled_feat = self.RCNN_ocr_roi_pooling(base_feat, rois.view(-1,5))

        b,c,h,w = pooled_feat.size()
        rnn_input = pooled_feat.view(b, -1, w)
        rnn_input = rnn_input.permute(2, 0, 1)
        encoder_output = self.rnn(rnn_input)
        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()


class _fasterRCNN_OCR(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN_OCR, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_ocr_roi_pooling = roi_pooling(2)  # ocr roi_pooling

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        # rnn初始化，隐藏节点256
        nh = 256
        nclass = len('0123456789.') + 1
        self.rnn = nn.Sequential(
            BidirectionalLSTM(1024, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        self.ctc_critition = CTCLoss().cuda()

    def forward(self, im_data, im_info, gt_boxes, text, text_length):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data


        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois//RPN网络做目标区域的建议
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes)  # RPN网络

        # if it is training phrase, then use ground trubut bboxes for refining//池化的时候，应该将IOU最大的设为正样本

        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes)  # 输入的rois为2000×4, 经过筛选之后，输出为256×4
            # rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data  # rcnn部分就是一个分类器和一个回归器
            rois = roi_data

            # rois_label = Variable(rois_label.view(-1).long())
            # rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            # rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            # rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            # rois_label = None
            # rois_target = None
            # rois_inside_ws = None
            # rois_outside_ws = None
            # rpn_loss_cls = 0
            # rpn_loss_bbox = 0

            roi_data = self.RCNN_proposal_target(rois, gt_boxes)  # 输入的rois为2000×4, 经过筛选之后，输出为256×4
            # rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data  # rcnn部分就是一个分类器和一个回归器
            rois = roi_data



        rois = Variable(rois)
        # do roi pooling based on predicted rois

        cfg.POOLING_MODE = 'ocr'
        # cfg.POOLING_MODE = 'pool'
        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'ocr':

            pooled_height = 2
            maxratio = ((rois[:, :, 3] - rois[:, :, 1]) / 16) / ((rois[:, :, 4] - rois[:, :, 2]) / 16)
            maxratio = maxratio.max().item()
            pooled_width = math.ceil(pooled_height * maxratio)
            pooling = _RoIPooling(pooled_height, pooled_width, 1.0/16, 1.0/16)             # height_spatial, width_spatial

            pooled_feat = pooling(base_feat, rois.view(-1, 5))
            # pooled_feat = self.RCNN_ocr_roi_pooling(base_feat, rois.view(-1, 5))                # 用numpy写的ocr pooling

        # 利用rnn进行序列学习
        b, c, h, w = pooled_feat.size()
        rnn_input = pooled_feat.view(b, -1, w)
        rnn_input = rnn_input.permute(2, 0, 1)
        preds = self.rnn(rnn_input)

        if self.training:
            # 计算ctc_loss
            ctcloss = 0
            batch_size = preds.size(1)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            text = text.squeeze(0); text_length = text_length.squeeze(0)
            text = text.cpu(); text_length = text_length.cpu()
            # preds_size = preds_size.cuda()
            ctcloss = self.ctc_critition(preds, text, preds_size, text_length) / batch_size             # 还需完善

            return rois, rpn_loss_cls, rpn_loss_bbox, ctcloss
        else:
            return preds


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        # normal_init(self.rnn, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
