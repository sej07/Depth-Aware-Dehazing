import os
from .base_dataset import BaseDehazeDataset


class SOTSDataset(BaseDehazeDataset):
    def __init__(self, root_dir, split='outdoor', transform=None):
        super().__init__(root_dir, transform)
        self.split = split
        hazy_dir = os.path.join(root_dir, split, 'hazy')
        gt_dir = os.path.join(root_dir, split, 'gt')
        hazy_files = sorted(os.listdir(hazy_dir))
        for hazy_name in hazy_files:
            img_id = hazy_name.split('_')[0]
            gt_name_png = f"{img_id}.png"
            gt_name_jpg = f"{img_id}.jpg"
            
            gt_path_png = os.path.join(gt_dir, gt_name_png)
            gt_path_jpg = os.path.join(gt_dir, gt_name_jpg)
            
            if os.path.exists(gt_path_png):
                gt_path = gt_path_png
            elif os.path.exists(gt_path_jpg):
                gt_path = gt_path_jpg
            else:
                continue
            
            hazy_path = os.path.join(hazy_dir, hazy_name)
            self.pairs.append((hazy_path, gt_path))
        
        print(f"SOTS {split}: Found {len(self.pairs)} image pairs")
