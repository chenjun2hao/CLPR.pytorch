# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN, _fasterRCNN_OCR
import pdb

class vgg16(_fasterRCNN_OCR):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN_OCR.__init__(self, classes, class_agnostic)

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

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    # self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])      # 测试不同的cnn主干网络
    self.RCNN_base = self.cnn

    # Fix the layers before conv3:
    # for layer in range(10):
    #   for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

