import sys
sys.path.append('/home/barshikar.s/depth-aware-dehazing')

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

from src.models import AODNet
from src.datasets import SOTSDataset, SOTSDepthDataset
from src.datasets.transforms import get_test_transforms
from src.evaluation import calculate_psnr, calculate_ssim


def evaluate_per_image(model, dataloader, device, use_depth=False):
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            hazy = batch['hazy'].to(device)
            clean = batch['clean'].to(device)
            filename = batch['filename']
            
            if use_depth:
                depth = batch['depth'].to(device)
                inp = torch.cat([hazy, depth], dim=1)
            else:
                inp = hazy
            
            output = model(inp)
            
            for i in range(output.shape[0]):
                psnr = calculate_psnr(output[i:i+1], clean[i:i+1])
                ssim = calculate_ssim(output[i:i+1], clean[i:i+1])
                results.append({
                    'filename': filename[i],
                    'psnr': psnr,
                    'ssim': ssim
                })
    
    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    transform = get_test_transforms(image_size=256)
    
    print("\nLoading AOD-Net baseline")
    model_baseline = AODNet(in_channels=3)
    ckpt = torch.load('experiments/aodnet_baseline/checkpoints/best.pth', map_location=device)
    model_baseline.load_state_dict(ckpt['model_state_dict'])
    model_baseline = model_baseline.to(device)
    
    print("Loading AOD-Net + Depth Concat")
    model_depth = AODNet(in_channels=4)
    ckpt = torch.load('experiments/aodnet_depth_concat/checkpoints/best.pth', map_location=device)
    model_depth.load_state_dict(ckpt['model_state_dict'])
    model_depth = model_depth.to(device)

    dataset_base = SOTSDataset(
        root_dir='/home/barshikar.s/depth-aware-dehazing/data/reside/SOTS',
        split='outdoor',
        transform=transform
    )

    dataset_depth = SOTSDepthDataset(
        root_dir='/home/barshikar.s/depth-aware-dehazing/data/reside/SOTS',
        depth_dir='/home/barshikar.s/depth-aware-dehazing/data/depth_cache/SOTS/outdoor/hazy',
        split='outdoor',
        transform=transform
    )
    
    loader_base = DataLoader(dataset_base, batch_size=1, shuffle=False, num_workers=2)
    loader_depth = DataLoader(dataset_depth, batch_size=1, shuffle=False, num_workers=2)
    
    # Evaluate
    print("\nEvaluating baseline")
    results_baseline = evaluate_per_image(model_baseline, loader_base, device, use_depth=False)
    
    print("\nEvaluating depth concat")
    results_depth = evaluate_per_image(model_depth, loader_depth, device, use_depth=True)
    
    # Compare
    print("Per-Image Analysis")
    
    improved = []
    degraded = []
    
    for rb, rd in zip(results_baseline, results_depth):
        diff = rd['psnr'] - rb['psnr']
        if diff > 0.5:
            improved.append((rb['filename'], rb['psnr'], rd['psnr'], diff))
        elif diff < -0.5:
            degraded.append((rb['filename'], rb['psnr'], rd['psnr'], diff))
    
    print(f"\nImages improved by depth (>0.5 dB): {len(improved)}")
    print("-"*70)
    for fname, base_psnr, depth_psnr, diff in sorted(improved, key=lambda x: -x[3])[:10]:
        print(f"  {fname}: {base_psnr:.2f} -> {depth_psnr:.2f} (+{diff:.2f} dB)")
    
    print(f"\nImages degraded by depth (>0.5 dB): {len(degraded)}")
    print("-"*70)
    for fname, base_psnr, depth_psnr, diff in sorted(degraded, key=lambda x: x[3])[:10]:
        print(f"  {fname}: {base_psnr:.2f} -> {depth_psnr:.2f} ({diff:.2f} dB)")
    
    # Save to CSV
    output_path = '/home/barshikar.s/depth-aware-dehazing/outputs/per_image_analysis.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'baseline_psnr', 'depth_psnr', 'diff'])
        for rb, rd in zip(results_baseline, results_depth):
            diff = rd['psnr'] - rb['psnr']
            writer.writerow([rb['filename'], rb['psnr'], rd['psnr'], diff])
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()