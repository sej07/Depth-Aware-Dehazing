import sys
sys.path.append('/home/barshikar.s/depth-aware-dehazing')

import torch
from src.models import AODNet

print("Testing AOD-Net")

model = AODNet(in_channels=3)
print(f"  Parameters: {model.get_num_params():,}")

x = torch.rand(2, 3, 256, 256) 
print(f"  Input shape: {x.shape}")


output = model(x)
print(f"  Output shape: {output.shape}")

print(f"  Output min: {output.min().item():.4f}")
print(f"  Output max: {output.max().item():.4f}")

print("\nTesting with depth input (4 channels)")
model_depth = AODNet(in_channels=4)
x_depth = torch.rand(2, 4, 256, 256)  # RGB + Depth
output_depth = model_depth(x_depth)
print(f"  Input shape: {x_depth.shape}")
print(f"  Output shape: {output_depth.shape}")

print("\nAOD-Net working")