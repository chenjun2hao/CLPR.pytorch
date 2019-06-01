from crnn import crnn
from lib.ocr_dataloader.ocrdataloader import ImgDataset             # ocr dataset
from lib.ocr_dataloader.ocrdataloader import strLabelConverter
import torch
from torch.autograd import Variable


## 数据集
converter = strLabelConverter('0123456789.')
dataset = ImgDataset(
    root='/media/chenjun/data/4_OCR/mech_demo2/dataset/ENDimgs/image',
    csv_root='/media/chenjun/data/4_OCR/mech_demo2/dataset/ENDimgs/train_list.txt',
    transform=None,
    target_transform=converter.encode
)
ocrdataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True
)
model_path = './weights/ocr.pth'
model = crnn()
model.load_state_dict(torch.load(model_path))
device = torch.device('cuda')
model.cuda()


model.eval()
numcorrect = 0

for index, data in enumerate(ocrdataloader):

    data = map(lambda x: x.cuda(), data[:-1])
    im_data, im_info, gt_boxes, text, text_length = data

    preds = model(im_data, text, text_length, gt_boxes)
    # crnn进行解码操作
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    target_size = torch.IntTensor([text.size(1)])
    target_str = converter.decode(text.squeeze().data, target_size.data, raw=True)
    if target_str == sim_pred:
        numcorrect += 1
    print(' %-20s => %-20s' % (target_str, sim_pred))

print(numcorrect)