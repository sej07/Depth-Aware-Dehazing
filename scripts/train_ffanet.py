import sys
sys.path.append('/home/barshikar.s/depth-aware-dehazing')

import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models import FFANet
from src.datasets import OTSDataset, SOTSDataset
from src.datasets.transforms import get_train_transforms, get_test_transforms
from src.losses import CombinedLoss
from src.trainers import DehazeTrainer


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_transform = get_train_transforms(image_size=args.image_size)
    val_transform = get_test_transforms(image_size=args.image_size)
    
    print("Loading datasets")
    
    train_dataset = OTSDataset(
        root_dir=args.train_dir,
        transform=train_transform
    )
    
    val_dataset = SOTSDataset(
        root_dir=args.val_dir,
        split='outdoor',
        transform=val_transform
    )

    print("Creating model")
    model = FFANet(
        in_channels=3,
        channels=args.channels,
        num_groups=args.num_groups,
        num_blocks=args.num_blocks
    )
    print(f"  Parameters: {model.get_num_params():,}")
    
    criterion = CombinedLoss(
        alpha=args.perceptual_weight,
        beta=args.ssim_weight
    )
    
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

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
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FFA-Net')
    
    parser.add_argument('--train_dir', type=str, 
                        default='/home/barshikar.s/depth-aware-dehazing/data/reside/OTS',
                        help='Path to OTS dataset (training)')
    parser.add_argument('--val_dir', type=str,
                        default='/home/barshikar.s/depth-aware-dehazing/data/reside/SOTS',
                        help='Path to SOTS dataset (validation)')
    
    parser.add_argument('--image_size', type=int, default=256,
                        help='Training image size')
    parser.add_argument('--channels', type=int, default=64,
                        help='Number of feature channels')
    parser.add_argument('--num_groups', type=int, default=3,
                        help='Number of FA groups')
    parser.add_argument('--num_blocks', type=int, default=6,
                        help='Number of FA blocks per group')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (smaller for FFA-Net)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    parser.add_argument('--perceptual_weight', type=float, default=0.1,
                        help='Weight for perceptual loss')
    parser.add_argument('--ssim_weight', type=float, default=0.1,
                        help='Weight for SSIM loss')
    
    parser.add_argument('--experiment_dir', type=str,
                        default='/home/barshikar.s/depth-aware-dehazing/experiments/ffanet_baseline',
                        help='Directory to save experiment')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)