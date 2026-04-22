import torch
import torch.nn as nn
from .pixel_losses import L1Loss
from .perceptual_loss import PerceptualLoss
from .ssim_loss import SSIMLoss


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, use_perceptual=True, use_ssim=True):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.use_perceptual = use_perceptual
        self.use_ssim = use_ssim
        
        # Initialize losses
        self.l1_loss = L1Loss()
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
            
        if use_ssim:
            self.ssim_loss = SSIMLoss()
            
    def forward(self, pred, target):
        losses = {}
        
        # L1 loss
        losses['l1'] = self.l1_loss(pred, target)
        total = losses['l1']
        
        # Perceptual loss
        if self.use_perceptual:
            losses['perceptual'] = self.perceptual_loss(pred, target)
            total = total + self.alpha * losses['perceptual']
            
        # SSIM loss
        if self.use_ssim:
            losses['ssim'] = self.ssim_loss(pred, target)
            total = total + self.beta * losses['ssim']
            
        losses['total'] = total
        
        return losses
