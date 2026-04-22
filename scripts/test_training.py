import sys
sys.path.append('/home/barshikar.s/depth-aware-dehazing')

import torch
from torch.optim import Adam
from src.models import AODNet
from src.datasets import SOTSDataset
from src.datasets.transforms import get_train_transforms, get_test_transforms
from src.losses import CombinedLoss
from src.trainers import DehazeTrainer

print("Testing training pipeline")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device}")

transform = get_train_transforms(image_size=128)

train_dataset = SOTSDataset(
    root_dir='/home/barshikar.s/depth-aware-dehazing/data/reside/SOTS',
    split='outdoor',
    transform=transform
)

val_dataset = SOTSDataset(
    root_dir='/home/barshikar.s/depth-aware-dehazing/data/reside/SOTS',
    split='outdoor',
    transform=transform
)

print(f"  Train samples: {len(train_dataset)}")
print(f"  Val samples: {len(val_dataset)}")

model = AODNet(in_channels=3)
print(f"  Model parameters: {model.get_num_params():,}")

criterion = CombinedLoss(alpha=0.1, beta=0.1)
optimizer = Adam(model.parameters(), lr=0.001)

trainer = DehazeTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    experiment_dir='/home/barshikar.s/depth-aware-dehazing/experiments/test_run'
)

print("\nRunning 1 test epoch")

from torch.utils.data import DataLoader, Subset
small_train = Subset(train_dataset, range(10))
small_val = Subset(val_dataset, range(10))

train_loader = DataLoader(small_train, batch_size=2, shuffle=True)
val_loader = DataLoader(small_val, batch_size=2, shuffle=False)

model = model.to(device)
model.train()

for batch in train_loader:
    hazy = batch['hazy'].to(device)
    clean = batch['clean'].to(device)
    
    output = model(hazy)
    losses = criterion(output, clean)
    
    print(f"\n  Batch test:")
    print(f"    Input shape: {hazy.shape}")
    print(f"    Output shape: {output.shape}")
    print(f"    L1 loss: {losses['l1'].item():.4f}")
    print(f"    Total loss: {losses['total'].item():.4f}")
    break

print("\nTraining pipeline ready")