import sys
sys.path.append('/home/barshikar.s/depth-aware-dehazing')

import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import JointDehazeDepthNet
from src.datasets import OTSDepthDataset, SOTSDepthDataset
from src.datasets.transforms import get_train_transforms, get_test_transforms


class JointLoss(nn.Module):

    def __init__(self, lambda_depth=0.5):
        super().__init__()
        self.lambda_depth = lambda_depth
        self.l1 = nn.L1Loss()
    
    def forward(self, dehazed, pred_depth, clean, gt_depth):
        loss_dehaze = self.l1(dehazed, clean)
        loss_depth = self.l1(pred_depth, gt_depth)
        
        total = loss_dehaze + self.lambda_depth * loss_depth
        
        return {
            'dehaze': loss_dehaze,
            'depth': loss_depth,
            'total': total
        }


class JointTrainer:
    
    def __init__(self, model, train_dataset, val_dataset, criterion, optimizer,
                 device='cuda', experiment_dir='experiments/default', scheduler=None):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.experiment_dir = experiment_dir
        
        import os
        self.checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
        for batch in pbar:
            hazy = batch['hazy'].to(self.device)
            clean = batch['clean'].to(self.device)
            gt_depth = batch['depth'].to(self.device)
            
            self.optimizer.zero_grad()
            
            dehazed, pred_depth = self.model(hazy)
            
            losses = self.criterion(dehazed, pred_depth, clean, gt_depth)
            loss = losses['total']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': loss.item(),
                'dh': losses['dehaze'].item(),
                'dp': losses['depth'].item()
            })
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
            for batch in pbar:
                hazy = batch['hazy'].to(self.device)
                clean = batch['clean'].to(self.device)
                gt_depth = batch['depth'].to(self.device)
                
                dehazed, pred_depth = self.model(hazy)
                
                losses = self.criterion(dehazed, pred_depth, clean, gt_depth)
                total_loss += losses['total'].item()
                pbar.set_postfix({'loss': losses['total'].item()})
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, epoch, is_best=False):
        import os
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best.pth'))
            print(f"  Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def train(self, num_epochs, batch_size=8, num_workers=4):
        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        print(f"Training for {num_epochs} epochs")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples: {len(self.val_dataset)}")
        print(f"  Batch size: {batch_size}")
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)
            
            if self.scheduler:
                self.scheduler.step()
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, is_best)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        print("Training complete")


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Transforms
    train_transform = get_train_transforms(image_size=args.image_size)
    val_transform = get_test_transforms(image_size=args.image_size)
    
    # Datasets with depth
    print("Loading datasets")
    train_dataset = OTSDepthDataset(
        root_dir='/home/barshikar.s/depth-aware-dehazing/data/reside/OTS',
        depth_dir='/home/barshikar.s/depth-aware-dehazing/data/depth_cache/OTS/hazy',
        transform=train_transform
    )
    
    val_dataset = SOTSDepthDataset(
        root_dir='/home/barshikar.s/depth-aware-dehazing/data/reside/SOTS',
        depth_dir='/home/barshikar.s/depth-aware-dehazing/data/depth_cache/SOTS/outdoor/hazy',
        split='outdoor',
        transform=val_transform
    )
    
    # Model
    print("Creating JointDehazeDepthNet")
    model = JointDehazeDepthNet(in_channels=3, base_channels=args.base_channels)
    print(f"  Parameters: {model.get_num_params():,}")
    
    # Loss
    criterion = JointLoss(lambda_depth=args.lambda_depth)
    print(f"  Lambda depth: {args.lambda_depth}")
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # Trainer
    trainer = JointTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        experiment_dir=args.experiment_dir,
        scheduler=scheduler
    )
    
    # Train
    trainer.train(num_epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Joint Dehaze-Depth Model')
    
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base channels (32 for smaller model, 64 for larger)')
    parser.add_argument('--lambda_depth', type=float, default=0.5,
                        help='Weight for depth loss')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (smaller due to large model)')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--experiment_dir', type=str,
                        default='/home/barshikar.s/depth-aware-dehazing/experiments/depth_joint')
    
    args = parser.parse_args()
    main(args)