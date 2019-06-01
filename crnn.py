import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from lib.model.roi_pooling.modules.roi_pool import _RoIPooling
from warpctc_pytorch import CTCLoss


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


class crnn(nn.Module):
    def __init__(self):
        super(crnn, self).__init__()

        # 测试不同的主干网络
        nc = 3
        leakyRelu = False
        ks = [3, 3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 1, 1]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                            nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                            nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x50
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x25
        convRelu(2, True)
        convRelu(3)
        # cnn.add_module('pooling{0}'.format(2),
        #                nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x26
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))  # 128x8x25

        convRelu(4, True)
        convRelu(5)
        # cnn.add_module('pooling{0}'.format(3),
        #                nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x27
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d(2, 2))  # 128x8x25

        # convRelu(6, True)  # 512x1x26
        # cnn.add_module('pooling{0}'.format(4),
        #                nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x27
        self.cnn = cnn

        # rnn初始化，隐藏节点256
        nh = 256
        nclass = len('0123456789.') + 1
        self.rnn = nn.Sequential(
            BidirectionalLSTM(1024, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        self.ctc_critition = CTCLoss().cuda()
    
    def forward(self, x, text, text_length, rois):
        base_feat = self.cnn(x)
        rois = Variable(rois)

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

            return ctcloss
        else:
            return preds