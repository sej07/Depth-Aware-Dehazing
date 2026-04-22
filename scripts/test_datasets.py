import sys
sys.path.append('/home/barshikar.s/depth-aware-dehazing')

from src.datasets import SOTSDataset, OHazeDataset, IHazeDataset
from torchvision import transforms

# Basic transform: convert to tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Test SOTS
print("Testing SOTS")
sots = SOTSDataset(
    root_dir='/home/barshikar.s/depth-aware-dehazing/data/reside/SOTS',
    split='outdoor',
    transform=transform
)
print(f"  Total pairs: {len(sots)}")
sample = sots[0]
print(f"  Sample hazy shape: {sample['hazy'].shape}")
print(f"  Sample clean shape: {sample['clean'].shape}")
print(f"  Sample filename: {sample['filename']}")

# Test O-HAZE
print("\nTesting O-HAZE")
ohaze = OHazeDataset(
    root_dir='/home/barshikar.s/depth-aware-dehazing/data/ohaze',
    transform=transform
)
print(f"  Total pairs: {len(ohaze)}")
sample = ohaze[0]
print(f"  Sample hazy shape: {sample['hazy'].shape}")
print(f"  Sample clean shape: {sample['clean'].shape}")
print(f"  Sample filename: {sample['filename']}")

# Test I-HAZE
print("\nTesting I-HAZE")
ihaze = IHazeDataset(
    root_dir='/home/barshikar.s/depth-aware-dehazing/data/ihaze',
    transform=transform
)
print(f"  Total pairs: {len(ihaze)}")
sample = ihaze[0]
print(f"  Sample hazy shape: {sample['hazy'].shape}")
print(f"  Sample clean shape: {sample['clean'].shape}")
print(f"  Sample filename: {sample['filename']}")

print("\n All datasets loaded successfully!")