import torch
import torch.nn as nn
from pytorch_msssim import ssim, SSIM


class SSIMLoss(nn.Module):
    
    def __init__(self, window_size=11, data_range=1.0):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        
    def forward(self, pred, target):
        ssim_val = ssim(pred, target, 
                        data_range=self.data_range,
                        win_size=self.window_size,
                        size_average=True)
        return 1.0 - ssim_val