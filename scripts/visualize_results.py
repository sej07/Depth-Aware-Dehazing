import sys
sys.path.append('/home/barshikar.s/depth-aware-dehazing')

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from src.models import AODNet, FFANet, DepthGuidedFFANet, JointDehazeDepthNet


def load_image(path, size=256):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)


def load_depth(path, size=256):
    depth = np.load(path)
    depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()  
    depth = torch.nn.functional.interpolate(depth, size=(size, size), mode='bilinear', align_corners=False)
    return depth


def tensor_to_image(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    return img


def visualize_comparison(hazy_path, gt_path, depth_path, models_dict, output_path, device='cuda'):

    # Load inputs
    hazy = load_image(hazy_path).to(device)
    gt = load_image(gt_path).to(device)
    depth = load_depth(depth_path).to(device)
    
    # Get outputs from each model
    outputs = {}
    with torch.no_grad():
        for name, (model, model_type) in models_dict.items():
            model.eval()
            
            if model_type == 'baseline':
                out = model(hazy)
            elif model_type == 'concat':
                inp = torch.cat([hazy, depth], dim=1)
                out = model(inp)
            elif model_type == 'attention':
                out = model(hazy, depth)
            elif model_type == 'joint':
                out, _ = model(hazy)
            
            outputs[name] = tensor_to_image(out)
    

    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models + 2, figsize=(3 * (n_models + 2), 3))
    
    axes[0].imshow(tensor_to_image(hazy))
    axes[0].set_title('Hazy Input')
    axes[0].axis('off')
    

    for i, (name, img) in enumerate(outputs.items()):
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(name)
        axes[i + 1].axis('off')

    axes[-1].imshow(tensor_to_image(gt))
    axes[-1].set_title('Ground Truth')
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    output_dir = '/home/barshikar.s/depth-aware-dehazing/outputs/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("Loading models")
    models_dict = {}
    
    # AOD-Net baseline
    model = AODNet(in_channels=3)
    ckpt = torch.load('experiments/aodnet_baseline/checkpoints/best.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    models_dict['AOD-Net'] = (model, 'baseline')
    print("  Loaded AOD-Net baseline")
    
    # FFA-Net baseline
    model = FFANet(in_channels=3)
    ckpt = torch.load('experiments/ffanet_baseline/checkpoints/best.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    models_dict['FFA-Net'] = (model, 'baseline')
    print("  Loaded FFA-Net baseline")
    
    # AOD-Net + Depth Concat
    model = AODNet(in_channels=4)
    ckpt = torch.load('experiments/aodnet_depth_concat/checkpoints/best.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    models_dict['AOD+Concat'] = (model, 'concat')
    print("  Loaded AOD-Net + Depth Concat")
    
    # Depth-Guided Attention
    try:
        model = DepthGuidedFFANet(in_channels=3, channels=64, num_groups=3, num_blocks=6,
                                   attention_type='learned', injection_points=['middle'])
        ckpt = torch.load('experiments/depth_attention/checkpoints/best.pth', map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        models_dict['Depth-Attn'] = (model, 'attention')
        print("  Loaded Depth-Guided Attention")
    except:
        print("  Depth-Guided Attention not found, skipping")
    
    # Joint Multi-Task
    try:
        model = JointDehazeDepthNet(in_channels=3, base_channels=32)
        ckpt = torch.load('experiments/depth_joint/checkpoints/best.pth', map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        models_dict['Joint'] = (model, 'joint')
        print("  Loaded Joint Multi-Task")
    except:
        print("  Joint Multi-Task not found, skipping")

    samples = [
        # SOTS samples
        {
            'name': 'sots_sample1',
            'hazy': '/home/barshikar.s/depth-aware-dehazing/data/reside/SOTS/outdoor/hazy/0001_0.8_0.2.jpg',
            'gt': '/home/barshikar.s/depth-aware-dehazing/data/reside/SOTS/outdoor/gt/0001.png',
            'depth': '/home/barshikar.s/depth-aware-dehazing/data/depth_cache/SOTS/outdoor/hazy/0001_0.8_0.2.npy'
        },
        # O-HAZE samples
        {
            'name': 'ohaze_sample1',
            'hazy': '/home/barshikar.s/depth-aware-dehazing/data/ohaze/hazy/01_outdoor_hazy.jpg',
            'gt': '/home/barshikar.s/depth-aware-dehazing/data/ohaze/GT/01_outdoor_GT.jpg',
            'depth': '/home/barshikar.s/depth-aware-dehazing/data/depth_cache/ohaze/hazy/01_outdoor_hazy.npy'
        },
        # I-HAZE samples
        {
            'name': 'ihaze_sample1',
            'hazy': '/home/barshikar.s/depth-aware-dehazing/data/ihaze/hazy/01_indoor_hazy.jpg',
            'gt': '/home/barshikar.s/depth-aware-dehazing/data/ihaze/GT/01_indoor_GT.jpg',
            'depth': '/home/barshikar.s/depth-aware-dehazing/data/depth_cache/ihaze/hazy/01_indoor_hazy.npy'
        }
    ]
    
    print("\nGenerating visualizations")
    for sample in samples:
        if not os.path.exists(sample['hazy']):
            print(f"  Skipping {sample['name']}: hazy image not found")
            continue
        if not os.path.exists(sample['gt']):
            print(f"  Skipping {sample['name']}: GT not found")
            continue
        if not os.path.exists(sample['depth']):
            print(f"  Skipping {sample['name']}: depth not found")
            continue
        
        output_path = os.path.join(output_dir, f"{sample['name']}.png")
        visualize_comparison(
            hazy_path=sample['hazy'],
            gt_path=sample['gt'],
            depth_path=sample['depth'],
            models_dict=models_dict,
            output_path=output_path,
            device=device
        )
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()