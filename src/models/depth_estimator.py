import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthEstimator(nn.Module):
    
    def __init__(self, model_type='DPT_Hybrid'):
        super().__init__()
        
        self.model_type = model_type
        
        self.midas = torch.hub.load('intel-isl/MiDaS', model_type)
        self.midas.eval()

        for param in self.midas.parameters():
            param.requires_grad = False
            
        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        if model_type == 'DPT_Large' or model_type == 'DPT_Hybrid':
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
            
    def forward(self, x):
        B, C, H, W = x.shape
        with torch.no_grad():
            depth = self.midas(x)
            
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            
            for i in range(B):
                d = depth[i]
                d_min = d.min()
                d_max = d.max()
                if d_max - d_min > 0:
                    depth[i] = (d - d_min) / (d_max - d_min)
                else:
                    depth[i] = torch.zeros_like(d)
                    
        return depth
    
    
def precompute_depth(image_path, output_path, model_type='DPT_Hybrid', device='cuda'):
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    
    midas = torch.hub.load('intel-isl/MiDaS', model_type)
    midas.eval().to(device)
    
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    if model_type == 'DPT_Large' or model_type == 'DPT_Hybrid':
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    input_batch = transform(img_np).to(device)
    
    with torch.no_grad():
        depth = midas(input_batch)
        depth = depth.squeeze().cpu().numpy()

    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    np.save(output_path, depth.astype(np.float32))
    
    return depth