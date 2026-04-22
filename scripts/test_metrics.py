import sys
sys.path.append('/home/barshikar.s/depth-aware-dehazing')

import torch
from src.evaluation import calculate_psnr, calculate_ssim, calculate_metrics

print("Testing evaluation metrics")

pred = torch.rand(2, 3, 256, 256)
target = torch.rand(2, 3, 256, 256)

print("\nRandom prediction vs random target:")
psnr = calculate_psnr(pred, target)
ssim_val = calculate_ssim(pred, target)
print(f"  PSNR: {psnr:.2f} dB")
print(f"  SSIM: {ssim_val:.4f}")

print("\nIdentical images:")
psnr = calculate_psnr(target, target)
ssim_val = calculate_ssim(target, target)
print(f"  PSNR: {psnr:.2f} dB (should be inf or very high)")
print(f"  SSIM: {ssim_val:.4f} (should be 1.0)")

print("\nTarget + small noise:")
noisy = target + 0.1 * torch.rand_like(target)
noisy = torch.clamp(noisy, 0, 1)
metrics = calculate_metrics(noisy, target)
print(f"  PSNR: {metrics['psnr']:.2f} dB")
print(f"  SSIM: {metrics['ssim']:.4f}")

print("\nMetrics working")