import sys
sys.path.append('/home/barshikar.s/depth-aware-dehazing')

import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def precompute_depth_maps(image_dir, output_dir, model_type='MiDaS_small'):
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    midas = torch.hub.load('intel-isl/MiDaS', model_type, trust_repo=True)
    midas.eval().to(device)
    
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
    transform = midas_transforms.small_transform
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images in {image_dir}")
    
    for img_name in tqdm(image_files, desc='Computing depth'):
        depth_name = os.path.splitext(img_name)[0] + '.npy'
        depth_path = os.path.join(output_dir, depth_name)
        
        if os.path.exists(depth_path):
            continue
        
        img_path = os.path.join(image_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        input_batch = transform(img_np).to(device)
        
        with torch.no_grad():
            depth = midas(input_batch)
            depth = depth.squeeze().cpu().numpy()
        
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        np.save(depth_path, depth.astype(np.float32))


if __name__ == '__main__':
    print("Computing depth from CLEAN images")
    precompute_depth_maps(
        image_dir='/home/barshikar.s/depth-aware-dehazing/data/reside/SOTS/outdoor/gt',
        output_dir='/home/barshikar.s/depth-aware-dehazing/data/depth_cache/SOTS/outdoor/clean'
    )
    print("Done")