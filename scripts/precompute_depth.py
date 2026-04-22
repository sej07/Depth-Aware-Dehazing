import sys
sys.path.append('/home/barshikar.s/depth-aware-dehazing')

import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def precompute_depth_maps(image_dir, output_dir, model_type='MiDaS_small'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load MiDaS
    print(f"Loading MiDaS ({model_type})")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    midas = torch.hub.load('intel-isl/MiDaS', model_type, trust_repo=True)
    midas.eval().to(device)
    
    # Load transforms
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
    if model_type in ['DPT_Large', 'DPT_Hybrid']:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    
    # Get all images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Process each image
    for img_name in tqdm(image_files, desc='Computing depth'):
        # Check if already computed
        depth_name = os.path.splitext(img_name)[0] + '.npy'
        depth_path = os.path.join(output_dir, depth_name)
        
        if os.path.exists(depth_path):
            continue
        
        # Load image
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        # Transform and predict
        input_batch = transform(img_np).to(device)
        
        with torch.no_grad():
            depth = midas(input_batch)
            depth = depth.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        # Save
        np.save(depth_path, depth.astype(np.float32))
    
    print(f"Depth maps saved to {output_dir}")


if __name__ == '__main__':
    # Precompute for OTS hazy images
    print("\n OTS Hazy")
    precompute_depth_maps(
        image_dir='/home/barshikar.s/depth-aware-dehazing/data/reside/OTS/hazy',
        output_dir='/home/barshikar.s/depth-aware-dehazing/data/depth_cache/OTS/hazy'
    )
    
    # Precompute for SOTS outdoor hazy images
    print("\n SOTS Outdoor Hazy")
    precompute_depth_maps(
        image_dir='/home/barshikar.s/depth-aware-dehazing/data/reside/SOTS/outdoor/hazy',
        output_dir='/home/barshikar.s/depth-aware-dehazing/data/depth_cache/SOTS/outdoor/hazy'
    )
    
    # Precompute for O-HAZE
    print("\n O-HAZE")
    precompute_depth_maps(
        image_dir='/home/barshikar.s/depth-aware-dehazing/data/ohaze/hazy',
        output_dir='/home/barshikar.s/depth-aware-dehazing/data/depth_cache/ohaze/hazy'
    )
    
    # Precompute for I-HAZE
    print("\n I-HAZE")
    precompute_depth_maps(
        image_dir='/home/barshikar.s/depth-aware-dehazing/data/ihaze/hazy',
        output_dir='/home/barshikar.s/depth-aware-dehazing/data/depth_cache/ihaze/hazy'
    )
    
    print("\n All depth maps precomputed")