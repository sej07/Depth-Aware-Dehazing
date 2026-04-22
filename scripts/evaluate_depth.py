import sys
sys.path.append('/home/barshikar.s/depth-aware-dehazing')

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

from src.models import AODNet, FFANet
from src.datasets.transforms import get_test_transforms
from src.evaluation import calculate_psnr, calculate_ssim


class EvalDepthDataset(Dataset):
    
    def __init__(self, hazy_dir, gt_dir, depth_dir, transform=None):
        self.hazy_dir = hazy_dir
        self.gt_dir = gt_dir
        self.depth_dir = depth_dir
        self.transform = transform
        
        self.hazy_files = sorted([f for f in os.listdir(hazy_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Found {len(self.hazy_files)} images")
    
    def __len__(self):
        return len(self.hazy_files)
    
    def __getitem__(self, idx):
        hazy_name = self.hazy_files[idx]
        
        hazy_path = os.path.join(self.hazy_dir, hazy_name)
        hazy_img = Image.open(hazy_path).convert('RGB')
        
        img_id = hazy_name.split('_')[0]
        gt_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.JPG']:
            path = os.path.join(self.gt_dir, f"{img_id}{ext}")
            if os.path.exists(path):
                gt_path = path
                break
            gt_name = hazy_name.replace('_hazy', '_GT')
            path = os.path.join(self.gt_dir, gt_name)
            if os.path.exists(path):
                gt_path = path
                break
        
        if gt_path is None:
            raise FileNotFoundError(f"GT not found for {hazy_name}")
        
        clean_img = Image.open(gt_path).convert('RGB')
        
        # Load depth
        depth_name = os.path.splitext(hazy_name)[0] + '.npy'
        depth_path = os.path.join(self.depth_dir, depth_name)
        depth = np.load(depth_path)
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        
        # Transform
        if self.transform:
            hazy_img = self.transform(hazy_img)
            clean_img = self.transform(clean_img)
            _, H, W = hazy_img.shape
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0)
        
        return {'hazy': hazy_img, 'clean': clean_img, 'depth': depth, 'filename': hazy_name}


def evaluate(model, dataloader, device):
    model.eval()
    psnr_list, ssim_list = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            hazy = batch['hazy'].to(device)
            clean = batch['clean'].to(device)
            depth = batch['depth'].to(device)
            
            # Concat and forward
            input_concat = torch.cat([hazy, depth], dim=1)
            output = model(input_concat)
            
            for i in range(output.shape[0]):
                psnr = calculate_psnr(output[i:i+1], clean[i:i+1])
                ssim = calculate_ssim(output[i:i+1], clean[i:i+1])
                psnr_list.append(psnr)
                ssim_list.append(ssim)
    
    return sum(psnr_list)/len(psnr_list), sum(ssim_list)/len(ssim_list)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    if args.model == 'aodnet':
        model = AODNet(in_channels=4)
    elif args.model == 'ffanet':
        model = FFANet(in_channels=4)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Parameters: {model.get_num_params():,}")
    
    transform = get_test_transforms(image_size=256)
    results = {}
    
    # SOTS Outdoor
    print("\n SOTS Outdoor")
    dataset = EvalDepthDataset(
        hazy_dir='/home/barshikar.s/depth-aware-dehazing/data/reside/SOTS/outdoor/hazy',
        gt_dir='/home/barshikar.s/depth-aware-dehazing/data/reside/SOTS/outdoor/gt',
        depth_dir='/home/barshikar.s/depth-aware-dehazing/data/depth_cache/SOTS/outdoor/hazy',
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    psnr, ssim = evaluate(model, dataloader, device)
    results['SOTS-Outdoor'] = {'PSNR': psnr, 'SSIM': ssim}
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")
    
    # O-HAZE
    print("\n O-HAZE")
    dataset = EvalDepthDataset(
        hazy_dir='/home/barshikar.s/depth-aware-dehazing/data/ohaze/hazy',
        gt_dir='/home/barshikar.s/depth-aware-dehazing/data/ohaze/GT',
        depth_dir='/home/barshikar.s/depth-aware-dehazing/data/depth_cache/ohaze/hazy',
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    psnr, ssim = evaluate(model, dataloader, device)
    results['O-HAZE'] = {'PSNR': psnr, 'SSIM': ssim}
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")
    
    # I-HAZE
    print("\n I-HAZE")
    dataset = EvalDepthDataset(
        hazy_dir='/home/barshikar.s/depth-aware-dehazing/data/ihaze/hazy',
        gt_dir='/home/barshikar.s/depth-aware-dehazing/data/ihaze/GT',
        depth_dir='/home/barshikar.s/depth-aware-dehazing/data/depth_cache/ihaze/hazy',
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    psnr, ssim = evaluate(model, dataloader, device)
    results['I-HAZE'] = {'PSNR': psnr, 'SSIM': ssim}
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")

    print("Summary")
    print(f"{'Dataset':<15} {'PSNR (dB)':<12} {'SSIM':<10}")
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['PSNR']:<12.2f} {metrics['SSIM']:<10.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['aodnet', 'ffanet'])
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    main(args)