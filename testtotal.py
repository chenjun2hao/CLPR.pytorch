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
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES, ImgDataset, LPDataset
import torch.utils.data as data
from ssd import build_ssd
import matplotlib.pyplot as plt
import numpy as np
from utils.ocrloss import strLabelConverter
from utils.ocrloss import OcrLoss
from PIL import Image, ImageDraw, ImageFont



parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_OCR_3000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
parser.add_argument('--alphabets', default='0123456789ABCDEFGHJKLMNOPQRSTUVWXYZ云京冀吉学宁川新晋桂沪津浙渝湘琼甘皖粤苏蒙藏警豫贵赣辽鄂闽陕青鲁黑')
parser.add_argument('--output', default='./output')

args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, ocrnet, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test1.txt'
    num_images = len(testset)
    for k in range(num_images):
        print('Testing image {:d}/{:d}....'.format(k+1, num_images))
        img = testset.pull_image(k)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        # with open(filename, mode='a') as f:
        #     f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
        #     for box in annotation:
        #         f.write('label: '+' || '.join(str(b) for b in box)+'\n')
        if cuda:
            x = x.cuda()

        y, ocrfeature = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])

        converter = strLabelConverter(args.alphabets)

        pred_num = 0
        for i in range(1, detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                coords = [int(x) for x in coords]

                # ocr识别ctc
                # crnn进行解码操作
                rois = torch.tensor([[0] + detections[0, i, j, 1:].cpu().numpy().tolist()])
                rois = rois.cpu()
                preds = ocrnet(ocrfeature, rois)

                _, preds = preds.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)

                preds_size = Variable(torch.IntTensor([preds.size(0)]))
                raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
                sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
                print('predict string is:{}'.format(sim_pred))
                # target_size = torch.IntTensor([text.size(1)])
                # target_str = converter.decode(text.squeeze().data, target_size.data, raw=True)

                # 画图显示
                cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pilimg = Image.fromarray(cv2img)
                draw = ImageDraw.Draw(pilimg)  # 图片上打印
                font = ImageFont.truetype("NotoSansCJK-Black.ttc", 40, encoding="utf-8")
                draw.text((coords[0], coords[1] - 40), sim_pred, (255, 0, 0), font=font)
                draw.rectangle((coords[0], coords[1], coords[2], coords[3]), outline=(255,0,0))
                img = cv2.cvtColor(np.asarray(pilimg), cv2.COLOR_RGB2BGR)
                # cv2.imshow("print chinese to image", img)
                # cv2.waitKey(1000)
                cv2.imwrite(('%s/%d.jpg'%(args.output, k)), img)
                # 不能显示中文
                # cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (0,0,255), 5)
                # cv2.putText(img, sim_pred, (coords[0], coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
                # plt.imshow(img); plt.show()
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1


def test_voc():
    # load net
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    num_classes = 1 + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    ocrnet = OcrLoss(args.alphabets)
    load_name = './weights/ssd300_OCR_24000.pth'
    checkpoint = torch.load(load_name)
    net.load_state_dict(checkpoint['model'])
    ocrnet.load_state_dict(checkpoint['ocr'])
    net.eval()
    ocrnet.eval()
    print('Finished loading model! {}'.format(load_name))

    testset = LPDataset(
            root=r"/media/chenjun/profile/2_learning/ssd.pytorch/data/test_data/*.jpg",
            csv_root=None,
        transform=BaseTransform(net.size, (104, 117, 123)),
        target_transform=None
    )

    if args.cuda:
        net = net.cuda()
        ocrnet = ocrnet.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, ocrnet, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
