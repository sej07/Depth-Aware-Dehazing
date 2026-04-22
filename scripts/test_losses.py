import sys
sys.path.append('/home/barshikar.s/depth-aware-dehazing')

import torch
from src.losses import L1Loss, MSELoss, PerceptualLoss, SSIMLoss, CombinedLoss

# Create dummy images
pred = torch.rand(2, 3, 256, 256)
target = torch.rand(2, 3, 256, 256)

print("Testing loss functions")

# Test L1
l1 = L1Loss()
print(f"  L1 Loss: {l1(pred, target).item():.4f}")

# Test MSE
mse = MSELoss()
print(f"  MSE Loss: {mse(pred, target).item():.4f}")

# Test SSIM
ssim = SSIMLoss()
print(f"  SSIM Loss: {ssim(pred, target).item():.4f}")

# Test Perceptual
print("  Loading VGG for perceptual loss")
perceptual = PerceptualLoss()
print(f"  Perceptual Loss: {perceptual(pred, target).item():.4f}")

# Test Combined
print("  Testing combined loss")
combined = CombinedLoss(alpha=0.1, beta=0.1)
losses = combined(pred, target)
print(f"  Combined Loss:")
print(f"    L1: {losses['l1'].item():.4f}")
print(f"    Perceptual: {losses['perceptual'].item():.4f}")
print(f"    SSIM: {losses['ssim'].item():.4f}")
print(f"    Total: {losses['total'].item():.4f}")

print("\nAll losses working")