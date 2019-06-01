# coding:utf-8
# 端到端ocr
# 对不同rois区域做max pooling，同时以最大长度进行填充

import torch
from torch.autograd import Function, Variable
import numpy as np 
from torch.autograd.gradcheck import gradcheck
from torch.nn.modules.module import Module

class roi_pooling(Function):
    '''
        staticmethod中ctx只能保存torch tensor
    '''
    def __init__(self, pooled_height):
        super(roi_pooling, self).__init__()
        self.pooled_height = pooled_height

    # @staticmethod
    def forward(self, input, rois_input):                  # pooled height:设置的需要池化的高度
        # cast to numpy
        rois_input[:, [1,3]] /= 4                                    # cnn卷集的比例
        rois_input[:, [2,4]] /= 16
        input_data = input.cpu().detach().numpy()
        rois = rois_input.cpu().detach().numpy()
        batch, num_channels, height, weight = input_data.shape
        pooled_height = self.pooled_height

        num_rois = rois.shape[0]

        max_ratio = max((rois[:,3] - rois[:,1]) / (rois[:,4] - rois[:,2]))      # 求出最大的宽度/高度的比值
        max_pooled_weight = int(np.ceil(max_ratio) * pooled_height)
        output = np.zeros((num_rois, num_channels, pooled_height, max_pooled_weight))       # 300×64×2×10;10是序列的长度
        argmax = np.ones((num_rois, num_channels, pooled_height, max_pooled_weight)) * -1       # 最大值的索引

        for n in range(num_rois):
            roi_start_w = np.round(rois[n, 1])                  # 每个区域w方向开始的坐标
            roi_start_h = np.round(rois[n, 2])                  # h方向开始的坐标

            rois_weight = np.max([rois[n,3] - rois[n,1], 1])                     # 每个区域的宽度
            rois_height = np.max([rois[n,4] - rois[n,2], 1])                     # 每个区域的高度

            pooled_weight = np.ceil(np.float(rois_weight) / rois_height * pooled_height)        # 和rois等比例的池化
            pooled_weight = int(pooled_weight)

            bin_size_w = np.float(rois_weight) / pooled_weight                  # 每个bin块的宽度
            bin_size_h = np.float(rois_height) / pooled_height                  # 每个bin块的高度

            ## 如何目标区域的rois太小则跳过
            th = np.floor(1 * bin_size_h)
            tw = np.floor(1 * bin_size_w)
            eh = np.ceil((1 + 1) * bin_size_h)
            ew = np.ceil((1 + 1) * bin_size_w)
            if eh - th == 0 or ew - tw == 0:
                continue
            try:
                for c in range(num_channels):
                    for ph in range(pooled_height):                                 # numpy的矩阵展开的时候是行优先的
                        for pw in range(pooled_weight):

                            hstart = np.floor(ph * bin_size_h)
                            wstart = np.floor(pw * bin_size_w)
                            hend = np.ceil((ph + 1) * bin_size_h)
                            wend = np.ceil((pw + 1) * bin_size_w)

                            hstart = min(max(hstart + roi_start_h, 0),height)           # 将每个bin限制在图片的尺寸范围内
                            hend = min(max(hend + roi_start_h, 0), height)
                            wstart = min(max(wstart + roi_start_w, 0), weight)
                            wend = min(max(wend + roi_start_w, 0), weight)

                            hstart, hend, wstart, wend = map(lambda x: int(x), [hstart, hend, wstart, wend])

                            output[n, c, ph, pw] = np.max(input_data[:, c, hstart:hend, wstart:wend])    # 最大值
                            argmax[n, c, ph, pw] = np.argmax(input_data[:,c, hstart:hend, wstart:wend])  # 最大值的索引//
            except:
                print('hstart:{},hend:{},wstart:{},wend:{}'.format(hstart,hend,wstart,wend))
        # ctx.save_for_backward()
        self.argmax = argmax
        self.height = height
        self.weight = weight
        self.rois = rois_input
        result = Variable(torch.tensor(output, dtype = input.dtype))
        # return torch.tensor(output, dtype = input.dtype).cuda()
        return result.cuda()
        # return torch.as_tensor(output, dtype=input.dtype)               # 300*512*2*20
        # return torch.from_numpy(output)

    # @staticmethod
    def backward(self, grad_input_t):
        # backward反向传播
        # grad_input = np.random.randn(num_rois, num_channels, pooled_height, max_pooled_weight)       # 梯度
        grad_input = grad_input_t.cpu().detach().numpy(); rois = self.rois.cpu().detach().numpy()
        argmax = self.argmax
        height = self.height
        weight = self.weight
        pooled_height = self.pooled_height
        num_rois, num_channels = grad_input.shape[0:2]


        mask = argmax.copy()                
        mask[0 <= mask] = 1
        mask[mask < 0] = 0
        grad_input = np.multiply(grad_input, mask)      # 填充区域的梯度不计入计算
        grad_output = np.zeros((num_rois, num_channels, height, weight))        # 反向传播的输出
        for n in range(num_rois):
            # 求bin的weight和height
            roi_start_w = np.round(rois[n, 1])
            roi_start_h = np.round(rois[n, 2])

            rois_weight = np.max([rois[n,3] - rois[n,1], 1])                     # 每个区域的宽度
            rois_height = np.max([rois[n,4] - rois[n,2], 1])                     # 每个区域的高度

            pooled_weight = np.ceil(np.float(rois_weight) / rois_height * pooled_height)        # 和rois等比例的池化
            pooled_weight = int(pooled_weight)

            bin_size_w = np.float(rois_weight) / pooled_weight                  # 每个bin块的宽度
            bin_size_h = np.float(rois_height) / pooled_height                  # 每个bin块的高度
            for c in range(num_channels):
                for ph in range(pooled_height):
                    for pw in range(pooled_weight):

                        hstart = np.floor(ph * bin_size_h)
                        wstart = np.floor(pw * bin_size_w)
                        hend = np.ceil((ph + 1) * bin_size_h)
                        wend = np.ceil((pw + 1) * bin_size_w)

                        hstart = min(max(hstart + roi_start_h, 0),height)           # 将每个bin限制在图片的尺寸范围内
                        hend = min(max(hend + roi_start_h, 0), height)
                        wstart = min(max(wstart + roi_start_w, 0), weight)
                        wend = min(max(wend + roi_start_w, 0), weight)
                        
                        hstart, hend, wstart, wend = map(lambda x: int(x), [hstart, hend, wstart, wend])

                        temp = np.zeros((hend-hstart, wend-wstart))             # 每个bin区域的临时值
                        temp = temp.flatten()
                        temp[int(argmax[n,c,ph,pw])] = grad_input[n,c,ph,pw]
                        temp = np.reshape(temp, (hend-hstart, -1))
                        grad_output[n,c, hstart:hend, wstart:wend] = temp           # 对一块bin进行赋值

        grad_output = np.sum(grad_output, axis=0)
        grad_output = np.expand_dims(grad_output, 0)
        return torch.tensor(grad_output, dtype=grad_input_t.dtype).cuda(), torch.tensor(rois, dtype=self.rois.dtype).cuda()
        # return torch.as_tensor(grad_output, dtype=grad_input_t.dtype), torch.as_tensor(rois, dtype=self.rois.dtype)


class ScipyConv2d(Module):
    def __init__(self):
        super(ScipyConv2d, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, 1)
        self.roi_pooling = roi_pooling(2)

    def forward(self, img, rois):
        conv = self.conv1(img)
        pooled = self.roi_pooling(conv, rois)
        return pooled


if __name__ == "__main__":
    ## 第一个版本的测试

    # dim = 10
    # input_data = [x for x in range(dim ** 2)]
    # input_data = np.array(input_data)
    # input_data = input_data.reshape((dim,dim))
    # input_data = torch.from_numpy(input_data).to(torch.float)
    # input_data = Variable(input_data, requires_grad=True)
    # input_data = input_data.view(1,1,dim,-1)
    base_feat = torch.randn(1,1,10,10)
    base_feat = Variable(base_feat, requires_grad=True)
    rois = np.asarray([[0,2,2,8,8], [0,2,2,7,7]])
    rois = torch.from_numpy(rois).to(torch.float)
    pooled_height = 2
    my_roi = roi_pooling(pooled_height)

    output = my_roi(input_data, rois)

    # output.backward(torch.ones(output.size()))
    # output.backward(torch.ones(output.size()))
    output.backward(output)

    print(input_data.grad)
    #
    temp = 1

    # pooled_height = 2                                       # 每个区域池化之后的高度
    # input_data = np.random.randint(1, 255,(1,64,50,50))       # image
    # rois = np.asarray([[0,10,10,20,20], [1,20,20,40,40],[2,1,2,30,7]])
    #
    # # input_data = torch.from_numpy(input_data).to(torch.float)
    # input_data = torch.randn(1,64,50,50)
    # input_data = Variable(input_data, requires_grad=True)
    #
    # rois = torch.from_numpy(rois). to(torch.float)
    #
    # my_roi = roi_pooling(pooled_height)
    #
    # print("反向传播之前，input的梯度为：{}".format(input_data.grad))
    #
    # output = my_roi(input_data, rois)
    #
    # output.backward(torch.randn(output.size()))
    #
    # print("反向传播之后，input的梯度为：{}".format(input_data.grad))
    # a = 1

    #///// 用module进行测试
    # input_data = np.random.randint(1, 255,(1,64,50,50))       # image 
    # rois = np.asarray([[0,10,10,20,20], [1,20,20,40,40],[2,1,2,30,7]])
    #
    # # input_data = torch.from_numpy(input_data).to(torch.float)
    # input_data = torch.randn(1,1,50,50)
    # input_data = Variable(input_data, requires_grad=True)
    #
    # rois = torch.from_numpy(rois). to(torch.float)
    # model = ScipyConv2d()
    # output = model(input_data, rois)
    # print("反向传播之前参数的梯度为：{}".format(list(model.parameters())[0].grad))
    # output.backward(torch.randn(output.size()))
    #
    # print(list(model.parameters()))
    # print("反向传播之后参数的梯度为：{}".format(list(model.parameters())[0].grad))