import torch.nn as nn
from warpctc_pytorch import CTCLoss         # ctcloss
from model.roi_pooling.modules.roi_pool import _RoIPooling          # ocr roipooling
import math
from torch.autograd import Variable
import collections
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


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


class OcrLoss(nn.Module):
    def __init__(self):
        super(OcrLoss, self).__init__()

        # rnn初始化，隐藏节点256
        nh = 256
        nclass = len('0123456789.') + 1
        self.rnn = nn.Sequential(
            BidirectionalLSTM(1024, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        self.ctc_critition = CTCLoss().cuda()

    def forward(self, *input):
        if self.training:
            base_feat, text, text_length, rois = input
            rois = Variable(rois)

            pooled_height = 2
            maxratio = (rois[:, 3] - rois[:, 1]) / (rois[:, 4] - rois[:, 2])
            maxratio = maxratio.max().item()
            pooled_width = math.ceil(pooled_height * maxratio)
            pooling = _RoIPooling(pooled_height, pooled_width, 1.0, 1.0)  # height_spatial, width_spatial

            b, c, h, w = base_feat.size()
            rois[:,[1,3]] *= w
            rois[:,[2,4]] *= h
            pooled_feat = pooling(base_feat, rois.view(-1, 5))
            # pooled_feat = self.RCNN_ocr_roi_pooling(base_feat, rois.view(-1, 5))                # 用numpy写的ocr pooling

            # 利用rnn进行序列学习
            b, c, h, w = pooled_feat.size()
            rnn_input = pooled_feat.view(b, -1, w)
            rnn_input = rnn_input.permute(2, 0, 1)
            preds = self.rnn(rnn_input)

            # 计算ctc_loss
            ctcloss = 0
            batch_size = preds.size(1)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            text = text.squeeze(0);
            text_length = text_length.squeeze(0)
            text = text.cpu();
            text_length = text_length.cpu()
            # preds_size = preds_size.cuda()
            ctcloss = self.ctc_critition(preds, text, preds_size, text_length) / batch_size  # 还需完善

            return ctcloss
        else:
            base_feat, rois = input
            rois = Variable(rois)

            pooled_height = 2
            maxratio = (rois[:, 3] - rois[:, 1]) / (rois[:, 4] - rois[:, 2])
            maxratio = maxratio.max().item()
            pooled_width = math.ceil(pooled_height * maxratio)
            pooling = _RoIPooling(pooled_height, pooled_width, 1.0, 1.0)  # height_spatial, width_spatial

            b, c, h, w = base_feat.size()
            rois[:, [1, 3]] *= w
            rois[:, [2, 4]] *= h
            temp = rois.numpy().tolist()
            # rois = torch.FloatTensor([0] + [temp[0][1], temp[0][2], temp[0][3], temp[0][4]] )
            rois = torch.tensor([0] + [temp[0][1], temp[0][2], temp[0][3], temp[0][4]])
            pooled_feat = pooling(base_feat, rois.view(-1, 5))
            # pooled_feat = self.RCNN_ocr_roi_pooling(base_feat, rois.view(-1, 5))                # 用numpy写的ocr pooling

            # 利用rnn进行序列学习
            b, c, h, w = pooled_feat.size()
            rnn_input = pooled_feat.view(b, -1, w)
            rnn_input = rnn_input.permute(2, 0, 1)
            preds = self.rnn(rnn_input)

            return preds

