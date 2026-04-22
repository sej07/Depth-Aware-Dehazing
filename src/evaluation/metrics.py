import torch
import numpy as np
from pytorch_msssim import ssim


def calculate_psnr(pred, target, max_val=1.0):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    mse = np.mean((pred - target) ** 2)
    
    if mse == 0:
        return float('inf')
        
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr


def calculate_ssim(pred, target, data_range=1.0):
    if not isinstance(pred, torch.Tensor):
        pred = torch.from_numpy(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target)

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
        
    ssim_val = ssim(pred, target, data_range=data_range, size_average=True)
    return ssim_val.item()


def calculate_metrics(pred, target):
    return {
        'psnr': calculate_psnr(pred, target),
        'ssim': calculate_ssim(pred, target)
    }