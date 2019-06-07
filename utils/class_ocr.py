import cupy as cp
import numpy as np
import math
import torch.nn as nn
import torch


roi_pooling_2d_fwd = cp.ElementwiseKernel(
            '''
            raw T bottom_data, T height_spatial_scale, T width_spatial_scale, int32 channels,
            int32 height, int32 width, int32 pooled_height, int32 pooled_width,
            raw T bottom_rois
            ''',
            'T top_data',
            '''
            // pos in output filter
            // 获得每个rois的左上和右下坐标
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int num = i / pooled_width / pooled_height / channels;
            int roi_batch_ind = bottom_rois[num * 5 + 0];
            int roi_start_w = round(bottom_rois[num * 5 + 1] * width_spatial_scale);
            int roi_start_h = round(bottom_rois[num * 5 + 2] * height_spatial_scale);
            int roi_end_w = round(bottom_rois[num * 5 + 3] * width_spatial_scale);
            int roi_end_h = round(bottom_rois[num * 5 + 4] * height_spatial_scale);

            // Force malformed ROIs to be 1x1
            // 计算rois_bin的宽度和高度
            int roi_width = max(roi_end_w - roi_start_w + 1, 1);
            int roi_height = max(roi_end_h - roi_start_h + 1, 1);

            // 进行等比例池化
            int rois_pooled_width = (int)(ceil((float)(pooled_height * roi_width) / (float)(roi_height) ));          // 等比例池化，减小
            float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
            float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(rois_pooled_width);
            int hstart = static_cast<int>(floor(static_cast<float>(ph) * bin_size_h));
            int wstart = static_cast<int>(floor(static_cast<float>(pw) * bin_size_w));
            int hend = static_cast<int>(ceil(static_cast<float>(ph + 1) * bin_size_h));
            int wend = static_cast<int>(ceil(static_cast<float>(pw + 1) * bin_size_w));

            // Add roi offsets and clip to input boundaries
            // 将每个roi_bin限制在图片的范围内
            hstart = min(max(hstart + roi_start_h, 0), height);
            hend = min(max(hend + roi_start_h, 0), height);
            wstart = min(max(wstart + roi_start_w, 0), width);
            wend = min(max(wend + roi_start_w, 0), width);
            bool is_empty = (hend <= hstart) || (wend <= wstart);

            // Define an empty pooling region to be zero
            float maxval = is_empty ? 0 : -1E+37;
            int maxidx = -1;
            // If nothing is pooled, argmax=-1 causes nothing to be backprop'd
            // 求roi_bin内的最大值

            // Skip if ROI doesn't include (h, w)
            const bool in_roi = (wstart >= roi_end_w );           // 判断当前区域是否在roi中
            if (in_roi) {                                 // 超出的部分填充为0
                top_data = 0;
            }
            else{
                int data_offset = (roi_batch_ind * channels + c) * height * width;
                for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                        int bottom_index = h * width + w;
                        if (bottom_data[data_offset + bottom_index] > maxval) {
                            maxval = bottom_data[data_offset + bottom_index];
                            maxidx = bottom_index;
                        }
                    }
                }
                top_data = maxval;              // 这里没有对top_data加索引i，这应该是自动加的。
            }
            ''', 'roi_pooling_2d_fwd'
        )


class ocr_roi_pooling(nn.Module):
    def __init__(self, pooled_height, pooled_width, height_spatial_scale, width_spatial_scale):
        super(ocr_roi_pooling, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.height_spatial_scale = float(height_spatial_scale)
        self.width_spatial_scale = float(width_spatial_scale)

    def forward(self, np_features, np_rois):
        batch, channels, height, width = np_features.shape
        bottom_data = cp.array(np_features, np.float32)
        rois = cp.array(np_rois, np.float32)
        nrois = rois.shape[0]

        top_data = cp.zeros((nrois, channels, self.pooled_height, self.pooled_width), dtype=np.float32)		# 输出的feature map
        roi_pooling_2d_fwd(bottom_data, self.height_spatial_scale, self.width_spatial_scale, 
                channels, height, width, self.pooled_height, self.pooled_width, rois, top_data)

        top_data = torch.Tensor(top_data.get())                 # 转换成torch tensor
        return top_data

if __name__=='__main__':
    
    batch, channels, height, width = 1,1,40,40
    bottom_data = np.random.randn(batch, channels, height, width)			# 特征feature 

    height_spatial_scale = 1.0													# 原始特征和feature的比例
    width_spatial_scale = 1.0
    rois = np.array([[0, 2, 2, 10, 10],
                    [1, 3, 15, 20, 20],
                    [2, 3, 15, 20, 20],
                    [3, 3, 15, 20, 20],
                    [4, 3, 15, 20, 20],
                    [5, 3, 15, 20, 20],
                    [6, 3, 15, 20, 20],
                    [7, 3, 15, 20, 20]], dtype=np.float32)				# rois
    rois = np.array([[0, 2, 2, 10, 10],
                    [0, 3, 15, 20, 20] ], dtype=np.float32)				# rois
    pooled_height = 2													# 池化之后的高度
    maxratio = (rois[:, 3] - rois[:, 1]) / (rois[:, 4] - rois[:, 2])
    maxratio = maxratio.max()
    pooled_width = math.ceil(pooled_height * maxratio)

    roi_pooling = ocr_roi_pooling(pooled_height, pooled_width, height_spatial_scale, width_spatial_scale)

    output = roi_pooling(bottom_data, rois)

    print(output.shape)