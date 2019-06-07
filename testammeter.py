from __future__ import print_function
import sys
import os, cv2
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from ssd import build_ssd
import matplotlib.pyplot as plt
from utils.ocrloss import OcrLoss
from lib.ocr_dataloader.ocrdataloader import strLabelConverter
from data import *


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_OCR_5000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, ocrdataloader, thresh, converter, lossnet):
    # dump predictions and assoc. ground truth to text file for now
    numcorrect = 0

    for index, data in enumerate(ocrdataloader):

        rois = data[-1]
        data = map(lambda x: x.cuda(), data[:-1])
        images, targets, text, text_length = data

        features = net(images)
        preds = lossnet(features[-1], rois)

        # crnn进行解码操作
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        target_size = torch.IntTensor([text.size(0)])
        target_str = converter.decode(text.squeeze().data, target_size.data, raw=True)
        if target_str == sim_pred:
            numcorrect += 1
        print(' %-20s => %-20s' % (target_str, sim_pred))

    print(numcorrect)



def test_voc():
    # load net
    num_classes = 1 + 1 # +1 background
    net = build_ssd('train', 300, num_classes) # initialize SSD
    ocrnet = OcrLoss()
    load_name = './weights/ssd300_OCR_12000.pth'
    checkpoint = torch.load(load_name)
    net.load_state_dict(checkpoint['model'])
    ocrnet.load_state_dict(checkpoint['ocr'])
    net.eval()
    ocrnet.eval()

    print('Finished loading model! {}'.format(load_name))

    converter = strLabelConverter('0123456789.')
    dataset = ImgDataset(
        # root='/media/chenjun/data/4_OCR/mech_demo2/dataset/ENDimgs/image',
        # csv_root='/media/chenjun/data/4_OCR/mech_demo2/dataset/ENDimgs/train_list.txt',
        root='/home/chenjun/1_chenjun/mech_demo2/dataset/imgs/image',
        csv_root='/home/chenjun/1_chenjun/mech_demo2/dataset/imgs/train_list.txt',
        transform=BaseTransform(net.size, (104, 117, 123)),
        target_transform=converter.encode
    )
    ocrdataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True,
        collate_fn=detection_collate
    )

    if args.cuda:
        net = net.cuda()
        ocrnet = ocrnet.cuda()
        cudnn.benchmark = True
        
    # evaluation
    test_net(args.save_folder, net, args.cuda, ocrdataloader,
             args.visual_threshold, converter, ocrnet)

if __name__ == '__main__':
    test_voc()
