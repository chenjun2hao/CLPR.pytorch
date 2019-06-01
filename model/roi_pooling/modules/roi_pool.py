from torch.nn.modules.module import Module
from ..functions.roi_pool import RoIPoolFunction


class _RoIPooling(Module):
    def __init__(self, pooled_height, pooled_width, height_spatial_scale, width_spatial_scale):
        super(_RoIPooling, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.height_spatial_scale = float(height_spatial_scale)
        self.width_spatial_scale = width_spatial_scale

    def forward(self, features, rois):
        return RoIPoolFunction(self.pooled_height, self.pooled_width, self.height_spatial_scale, self.width_spatial_scale)(features, rois)
