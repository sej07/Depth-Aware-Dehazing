import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class DehazeTrainer:
    def __init__(self, model, train_dataset, val_dataset, criterion, optimizer, device='cuda', 
    experiment_dir='experiments/default', scheduler=None):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        
        self.experiment_dir = experiment_dir
        self.checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
        self.log_dir = os.path.join(experiment_dir, 'logs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        
        self.best_val_loss = float('inf')
        
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, batch in enumerate(pbar):
            hazy = batch['hazy'].to(self.device)
            clean = batch['clean'].to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(hazy)
            
            losses = self.criterion(output, clean)
            loss = losses['total']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            global_step = epoch * len(dataloader) + batch_idx
            if batch_idx % 100 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def validate(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
            for batch in pbar:
                hazy = batch['hazy'].to(self.device)
                clean = batch['clean'].to(self.device)
                output = self.model(hazy)
                
                losses = self.criterion(output, clean)
                loss = losses['total']
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
        avg_loss = total_loss / len(dataloader)
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, path)
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, path)
            print(f"  Saved best model (val_loss: {self.best_val_loss:.4f})")
            
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint['epoch']
    
    def train(self, num_epochs, batch_size=16, num_workers=4):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"Training for {num_epochs} epochs...")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples: {len(self.val_dataset)}")
        print(f"  Batch size: {batch_size}")
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            
            val_loss = self.validate(val_loader, epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                
            self.save_checkpoint(epoch, is_best)
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
        self.writer.close()
        print("Training complete!")