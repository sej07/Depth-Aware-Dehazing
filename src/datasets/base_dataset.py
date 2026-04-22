import os
from PIL import Image
from torch.utils.data import Dataset


class BaseDehazeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = []  # List of (hazy_path, gt_path) tuples
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        hazy_path, gt_path = self.pairs[idx]
        
        hazy_img = Image.open(hazy_path).convert('RGB')
        clean_img = Image.open(gt_path).convert('RGB')
        
        if self.transform:
            hazy_img = self.transform(hazy_img)
            clean_img = self.transform(clean_img)
        filename = os.path.basename(hazy_path)
        
        return {
            'hazy': hazy_img,
            'clean': clean_img,
            'filename': filename
        }