from __future__ import print_function
import os, cv2
import argparse
import torch
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from data import BaseTransform, LPDataset
from ssd import build_ssd
import numpy as np
from utils.ocrloss import strLabelConverter
from utils.ocrloss import OcrLoss
from PIL import Image, ImageDraw, ImageFont



parser = argparse.ArgumentParser(description='END TO END chinese license plate recognition')
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
parser.add_argument('--model', type=str, default='./weights/ssd300_OCR_24000.pth')
parser.add_argument('--test_folder', default=r"./data/test_data/*.jpg")
parser.add_argument('--threshed', type=float, default=0.5)

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
            while detections[0, i, j, 0] >= args.threshed:
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

                # show and save the image
                cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pilimg = Image.fromarray(cv2img)
                draw = ImageDraw.Draw(pilimg)  # 图片上打印
                font = ImageFont.truetype("NotoSansCJK-Black.ttc", 40, encoding="utf-8")
                draw.text((coords[0], coords[1] - 40), sim_pred, (255, 0, 0), font=font)
                draw.rectangle((coords[0], coords[1], coords[2], coords[3]), outline=(255,0,0))
                img = cv2.cvtColor(np.asarray(pilimg), cv2.COLOR_RGB2BGR)
                cv2.imwrite(('%s/%d.jpg'%(args.output, k)), img)

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
    load_name = args.model
    checkpoint = torch.load(load_name)
    net.load_state_dict(checkpoint['model'])
    ocrnet.load_state_dict(checkpoint['ocr'])
    net.eval()
    ocrnet.eval()
    print('Finished loading model! {}'.format(load_name))

    testset = LPDataset(
            root=args.test_folder,
            csv_root=None,
        transform=BaseTransform(net.size, (104, 117, 123)),
        target_transform=None
    )

    if args.cuda:
        net = net.cuda()
        ocrnet = ocrnet.cuda()

    # evaluation
    test_net(args.save_folder, net, ocrnet, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
