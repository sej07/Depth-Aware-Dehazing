import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OTSDepthDataset(Dataset):
    
    def __init__(self, root_dir, depth_dir, transform=None):
        self.root_dir = root_dir
        self.depth_dir = depth_dir
        self.transform = transform
        
        # Set paths
        hazy_dir = os.path.join(root_dir, 'hazy')
        clear_dir = os.path.join(root_dir, 'clear')
        
        # Get all hazy images
        hazy_files = sorted(os.listdir(hazy_dir))
        
        self.triplets = []
        for hazy_name in hazy_files:
            img_id = hazy_name.split('_')[0]
            
            clear_name = f"{img_id}.jpg"
            clear_path = os.path.join(clear_dir, clear_name)
            
            # Depth map
            depth_name = os.path.splitext(hazy_name)[0] + '.npy'
            depth_path = os.path.join(depth_dir, depth_name)
            
            if os.path.exists(clear_path) and os.path.exists(depth_path):
                hazy_path = os.path.join(hazy_dir, hazy_name)
                self.triplets.append((hazy_path, clear_path, depth_path))
        
        print(f"OTS+Depth: Found {len(self.triplets)} triplets")
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        hazy_path, clear_path, depth_path = self.triplets[idx]
        
        # Load images
        hazy_img = Image.open(hazy_path).convert('RGB')
        clean_img = Image.open(clear_path).convert('RGB')
        
        # Load depth
        depth = np.load(depth_path)
        depth = torch.from_numpy(depth).unsqueeze(0).float() 
        
        if self.transform:
            hazy_img = self.transform(hazy_img)
            clean_img = self.transform(clean_img)
            
            _, H, W = hazy_img.shape
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0)
        
        return {
            'hazy': hazy_img,
            'clean': clean_img,
            'depth': depth,
            'filename': os.path.basename(hazy_path)
        }


class SOTSDepthDataset(Dataset):
    def __init__(self, root_dir, depth_dir, split='outdoor', transform=None):
        self.root_dir = root_dir
        self.depth_dir = depth_dir
        self.split = split
        self.transform = transform
        
        # Set paths
        hazy_dir = os.path.join(root_dir, split, 'hazy')
        gt_dir = os.path.join(root_dir, split, 'gt')
        
        # Get all hazy images
        hazy_files = sorted(os.listdir(hazy_dir))
        
        # Build triplets
        self.triplets = []
        for hazy_name in hazy_files:
            img_id = hazy_name.split('_')[0]
            
            gt_path = None
            for ext in ['.png', '.jpg']:
                path = os.path.join(gt_dir, f"{img_id}{ext}")
                if os.path.exists(path):
                    gt_path = path
                    break
            
            # Depth map
            depth_name = os.path.splitext(hazy_name)[0] + '.npy'
            depth_path = os.path.join(depth_dir, depth_name)
            
            if gt_path and os.path.exists(depth_path):
                hazy_path = os.path.join(hazy_dir, hazy_name)
                self.triplets.append((hazy_path, gt_path, depth_path))
        
        print(f"SOTS+Depth ({split}): Found {len(self.triplets)} triplets")
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        hazy_path, gt_path, depth_path = self.triplets[idx]
        
        hazy_img = Image.open(hazy_path).convert('RGB')
        clean_img = Image.open(gt_path).convert('RGB')
        
        # Load depth
        depth = np.load(depth_path)
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        
        if self.transform:
            hazy_img = self.transform(hazy_img)
            clean_img = self.transform(clean_img)
            
            _, H, W = hazy_img.shape
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0)
        
        return {
            'hazy': hazy_img,
            'clean': clean_img,
            'depth': depth,
            'filename': os.path.basename(hazy_path)
        }