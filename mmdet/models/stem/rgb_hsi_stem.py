import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class RGBHSIStem(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
        ) -> None:
        super().__init__()
        self.rgb_channels = 3
        self.hsi_channels = in_channels - 3
        self.out_channels = out_channels
        
        self.rgb_branch = nn.Sequential(
            nn.Conv2d(self.rgb_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU()
        )
        
        self.hsi_branch = nn.Sequential(
            nn.Conv2d(self.hsi_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU()
        )
    
    def forward(self, x):
        
        rgb = x[:, :3, :, :]
        hsi = x[:, 3:, :, :]
        
        rgb = self.rgb_branch(rgb)
        hsi = self.hsi_branch(hsi)
        
        return torch.cat((rgb, hsi), dim=1)