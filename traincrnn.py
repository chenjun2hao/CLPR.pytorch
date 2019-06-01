from crnn import crnn
from lib.ocr_dataloader.ocrdataloader import ImgDataset             # ocr dataset
from lib.ocr_dataloader.ocrdataloader import strLabelConverter
import torch

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

model = crnn()
device = torch.device('cuda')
model.cuda()
lr=0.0001

optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)

data_iter = iter(ocrdataloader)             # ocr dataloader
for epoch in range(11000):
    model.train()
    try:
        data = next(data_iter)
    except:
        data_iter = iter(ocrdataloader)
        data = next(data_iter)
    data = map(lambda x:x.cuda(), data[:-1])
    im_data, im_info, gt_boxes, text, text_length = data

    model.zero_grad()

    ctcloss = model(im_data, text, text_length, gt_boxes)

    optimizer.zero_grad()
    ctcloss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print("[epoch %2d][iter %4d] loss: %.4f, lr: %.2e" \
                                    % ( epoch, 18000, ctcloss.item(), lr))


torch.save(model.state_dict(), './weights/ocr.pth')