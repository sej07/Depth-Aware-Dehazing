import sys
sys.path.append('/home/barshikar.s/depth-aware-dehazing')

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import AODNet, FFANet
from src.datasets import SOTSDataset, OHazeDataset, IHazeDataset
from src.datasets.transforms import get_test_transforms
from src.evaluation import calculate_psnr, calculate_ssim


def evaluate(model, dataloader, device):
    model.eval()
    
    psnr_list = []
    ssim_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            hazy = batch['hazy'].to(device)
            clean = batch['clean'].to(device)

            output = model(hazy)
            
            for i in range(output.shape[0]):
                pred = output[i:i+1]
                target = clean[i:i+1]
                
                psnr = calculate_psnr(pred, target)
                ssim = calculate_ssim(pred, target)
                
                psnr_list.append(psnr)
                ssim_list.append(ssim)
    
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    
    return avg_psnr, avg_ssim


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"\nLoading model: {args.model}")
    if args.model == 'aodnet':
        model = AODNet(in_channels=3)
    elif args.model == 'ffanet':
        model = FFANet(in_channels=3)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Parameters: {model.get_num_params():,}")
    
    transform = get_test_transforms(image_size=256)
    
    results = {}
    
    # SOTS Outdoor
    print("\n SOTS Outdoor")
    dataset = SOTSDataset(
        root_dir='/home/barshikar.s/depth-aware-dehazing/data/reside/SOTS',
        split='outdoor',
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    psnr, ssim = evaluate(model, dataloader, device)
    results['SOTS-Outdoor'] = {'PSNR': psnr, 'SSIM': ssim}
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")
    
    # O-HAZE
    print("\n O-HAZE")
    dataset = OHazeDataset(
        root_dir='/home/barshikar.s/depth-aware-dehazing/data/ohaze',
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    psnr, ssim = evaluate(model, dataloader, device)
    results['O-HAZE'] = {'PSNR': psnr, 'SSIM': ssim}
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")
    
    # I-HAZE
    print("\n I-HAZE")
    dataset = IHazeDataset(
        root_dir='/home/barshikar.s/depth-aware-dehazing/data/ihaze',
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    psnr, ssim = evaluate(model, dataloader, device)
    results['I-HAZE'] = {'PSNR': psnr, 'SSIM': ssim}
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")
    
    # Summary
    print("Summary")
    print(f"{'Dataset':<15} {'PSNR (dB)':<12} {'SSIM':<10}")
    for dataset_name, metrics in results.items():
        print(f"{dataset_name:<15} {metrics['PSNR']:<12.2f} {metrics['SSIM']:<10.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate dehazing model')
    parser.add_argument('--model', type=str, required=True, choices=['aodnet', 'ffanet'],
                        help='Model type')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    args = parser.parse_args()
    main(args)