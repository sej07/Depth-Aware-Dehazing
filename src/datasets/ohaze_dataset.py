import os
from .base_dataset import BaseDehazeDataset


class OHazeDataset(BaseDehazeDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)
        hazy_dir = os.path.join(root_dir, 'hazy')
        gt_dir = os.path.join(root_dir, 'GT')
        
        hazy_files = sorted(os.listdir(hazy_dir))

        for hazy_name in hazy_files:
            gt_name = hazy_name.replace('_hazy', '_GT')
            
            hazy_path = os.path.join(hazy_dir, hazy_name)
            gt_path = os.path.join(gt_dir, gt_name)
            if os.path.exists(gt_path):
                self.pairs.append((hazy_path, gt_path))
            else:
                name, ext = os.path.splitext(gt_name)
                alt_ext = ext.upper() if ext.islower() else ext.lower()
                gt_path_alt = os.path.join(gt_dir, name + alt_ext)
                
                if os.path.exists(gt_path_alt):
                    self.pairs.append((hazy_path, gt_path_alt))
        
        print(f"O-HAZE: Found {len(self.pairs)} image pairs")