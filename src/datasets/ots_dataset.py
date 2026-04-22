import os
from .base_dataset import BaseDehazeDataset


class OTSDataset(BaseDehazeDataset):
    
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)
        
        hazy_dir = os.path.join(root_dir, 'hazy')
        clear_dir = os.path.join(root_dir, 'clear')
        
        hazy_files = sorted(os.listdir(hazy_dir))
        
        for hazy_name in hazy_files:
            img_id = hazy_name.split('_')[0]
            
            clear_name = f"{img_id}.jpg"
            clear_path = os.path.join(clear_dir, clear_name)
            
            if os.path.exists(clear_path):
                hazy_path = os.path.join(hazy_dir, hazy_name)
                self.pairs.append((hazy_path, clear_path))
        
        print(f"OTS: Found {len(self.pairs)} image pairs")