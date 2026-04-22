import sys
sys.path.append('/home/barshikar.s/depth-aware-dehazing')

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

from src.models import AODNet
from src.datasets.transforms import get_test_transforms
from src.evaluation import calculate_psnr, calculate_ssim


class SOTSCleanDepthDataset(Dataset):
    
    def __init__(self, root_dir, hazy_depth_dir, clean_depth_dir, split='outdoor', transform=None):
        self.root_dir = root_dir
        self.hazy_depth_dir = hazy_depth_dir
        self.clean_depth_dir = clean_depth_dir
        self.transform = transform
        
        hazy_dir = os.path.join(root_dir, split, 'hazy')
        gt_dir = os.path.join(root_dir, split, 'gt')
        
        hazy_files = sorted(os.listdir(hazy_dir))
        
        self.samples = []
        for hazy_name in hazy_files:
            img_id = hazy_name.split('_')[0]
            
            # Find GT
            gt_path = None
            for ext in ['.png', '.jpg']:
                path = os.path.join(gt_dir, f"{img_id}{ext}")
                if os.path.exists(path):
                    gt_path = path
                    break
            
            # Hazy depth
            hazy_depth_name = os.path.splitext(hazy_name)[0] + '.npy'
            hazy_depth_path = os.path.join(hazy_depth_dir, hazy_depth_name)
            
            clean_depth_path = os.path.join(clean_depth_dir, f"{img_id}.npy")
            
            if gt_path and os.path.exists(hazy_depth_path) and os.path.exists(clean_depth_path):
                self.samples.append({
                    'hazy_path': os.path.join(hazy_dir, hazy_name),
                    'gt_path': gt_path,
                    'hazy_depth_path': hazy_depth_path,
                    'clean_depth_path': clean_depth_path,
                    'filename': hazy_name
                })
        
        print(f"Found {len(self.samples)} samples with both depth types")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        hazy = Image.open(sample['hazy_path']).convert('RGB')
        clean = Image.open(sample['gt_path']).convert('RGB')
        
        hazy_depth = np.load(sample['hazy_depth_path'])
        hazy_depth = torch.from_numpy(hazy_depth).unsqueeze(0).float()
        
        clean_depth = np.load(sample['clean_depth_path'])
        clean_depth = torch.from_numpy(clean_depth).unsqueeze(0).float()
        
        if self.transform:
            hazy = self.transform(hazy)
            clean = self.transform(clean)
            _, H, W = hazy.shape
            hazy_depth = torch.nn.functional.interpolate(
                hazy_depth.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0)
            clean_depth = torch.nn.functional.interpolate(
                clean_depth.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0)
        
        return {
            'hazy': hazy,
            'clean': clean,
            'hazy_depth': hazy_depth,
            'clean_depth': clean_depth,
            'filename': sample['filename']
        }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    transform = get_test_transforms(image_size=256)
    
    print("\nLoading AOD-Net + Depth Concat")
    model = AODNet(in_channels=4)
    ckpt = torch.load('experiments/aodnet_depth_concat/checkpoints/best.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    dataset = SOTSCleanDepthDataset(
        root_dir='/home/barshikar.s/depth-aware-dehazing/data/reside/SOTS',
        hazy_depth_dir='/home/barshikar.s/depth-aware-dehazing/data/depth_cache/SOTS/outdoor/hazy',
        clean_depth_dir='/home/barshikar.s/depth-aware-dehazing/data/depth_cache/SOTS/outdoor/clean',
        split='outdoor',
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Evaluate with both depth types
    hazy_depth_psnr = []
    clean_depth_psnr = []
    
    print("\nComparing hazy depth vs clean depth")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            hazy = batch['hazy'].to(device)
            clean = batch['clean'].to(device)
            hazy_depth = batch['hazy_depth'].to(device)
            clean_depth = batch['clean_depth'].to(device)
            
            # With hazy depth
            inp_hazy = torch.cat([hazy, hazy_depth], dim=1)
            out_hazy = model(inp_hazy)
            psnr_hazy = calculate_psnr(out_hazy, clean)
            hazy_depth_psnr.append(psnr_hazy)
            
            # With clean depth (oracle)
            inp_clean = torch.cat([hazy, clean_depth], dim=1)
            out_clean = model(inp_clean)
            psnr_clean = calculate_psnr(out_clean, clean)
            clean_depth_psnr.append(psnr_clean)
    
    avg_hazy = sum(hazy_depth_psnr) / len(hazy_depth_psnr)
    avg_clean = sum(clean_depth_psnr) / len(clean_depth_psnr)
    
    print("Depth Quality Comparison")
    print(f"{'Depth Source':<25} {'Avg PSNR (dB)':<15}")
    print(f"{'Depth from HAZY image':<25} {avg_hazy:.2f}")
    print(f"{'Depth from CLEAN image':<25} {avg_clean:.2f}")
    print(f"{'Difference':<25} {avg_clean - avg_hazy:+.2f}")
    
    if avg_clean > avg_hazy:
        print("\nConclusion: Clean depth improves results")
        print("This confirms that noisy depth from hazy images hurts performance.")
    else:
        print("\nConclusion: Depth quality doesn't matter much.")


if __name__ == '__main__':
    main()