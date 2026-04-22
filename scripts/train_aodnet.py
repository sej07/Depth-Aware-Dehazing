import sys
sys.path.append('/home/barshikar.s/depth-aware-dehazing')

import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models import AODNet
from src.datasets import OTSDataset, SOTSDataset
from src.datasets.transforms import get_train_transforms, get_test_transforms
from src.losses import CombinedLoss
from src.trainers import DehazeTrainer


def main(args):
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Transforms
    train_transform = get_train_transforms(image_size=args.image_size)
    val_transform = get_test_transforms(image_size=args.image_size)
    
    # Datasets
    print("Loading datasets")
    
    # Train on OTS
    train_dataset = OTSDataset(
        root_dir=args.train_dir,
        transform=train_transform
    )
    
    # Validate on SOTS
    val_dataset = SOTSDataset(
        root_dir=args.val_dir,
        split='outdoor',
        transform=val_transform
    )
    
    # Model
    print("Creating model")
    model = AODNet(in_channels=3)
    print(f"  Parameters: {model.get_num_params():,}")
    
    # Loss
    criterion = CombinedLoss(
        alpha=args.perceptual_weight,
        beta=args.ssim_weight
    )
    
    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Trainer
    trainer = DehazeTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        experiment_dir=args.experiment_dir,
        scheduler=scheduler
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AOD-Net')
    
    # Data
    parser.add_argument('--train_dir', type=str, 
                        default='/home/barshikar.s/depth-aware-dehazing/data/reside/OTS',
                        help='Path to OTS dataset (training)')
    parser.add_argument('--val_dir', type=str,
                        default='/home/barshikar.s/depth-aware-dehazing/data/reside/SOTS',
                        help='Path to SOTS dataset (validation)')
    
    # Model
    parser.add_argument('--image_size', type=int, default=256,
                        help='Training image size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Loss weights
    parser.add_argument('--perceptual_weight', type=float, default=0.1,
                        help='Weight for perceptual loss')
    parser.add_argument('--ssim_weight', type=float, default=0.1,
                        help='Weight for SSIM loss')
    
    # Experiment
    parser.add_argument('--experiment_dir', type=str,
                        default='/home/barshikar.s/depth-aware-dehazing/experiments/aodnet_baseline',
                        help='Directory to save experiment')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)