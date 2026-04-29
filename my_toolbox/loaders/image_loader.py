import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

# ==========================================
# EXAMPLE USAGE
# ==========================================
# df = pd.DataFrame({
#     "image": ["train/img1.png", "train/img2.jpg"],
#     "mask": ["masks/mask1.png", None],    # Optional
#     "label": ["dog", "cat"]                # Optional
# })
#
# dataset = ImageDataset(df)
# first_item = dataset[0] # Returns (img_tensor, mask_tensor, label_tensor)

class ImageDataset(Dataset):
    """
    Consumes a DataFrame with 'image', and optional 'mask'/'label' columns.
    All paths are assumed to be relative to ./data
    """
    def __init__(self, df: pd.DataFrame, transform=None, augment=None):
        self.df = df
        self.transform = transform
        self.augment = augment
        self.base_path = Path("./data")
        
        # Mapping labels to integers starting at 0
        self.label_map = None
        if 'label' in self.df.columns:
            unique_labels = sorted(self.df['label'].unique())
            self.label_map = {name: i for i, name in enumerate(unique_labels)}

    def __len__(self):
        return len(self.df)

    def _load_img(self, relative_path, is_mask=False):
        """Loads file from ./data/relative_path"""
        full_path = self.base_path / str(relative_path).lstrip('/')
        
        if full_path.suffix == '.npy':
            data = np.load(full_path)
        else:
            data = cv2.imread(str(full_path))
            if data is None:
                raise FileNotFoundError(f"Missing file: {full_path}")
            # Standardize: RGB for images, Gray for masks
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY if is_mask else cv2.COLOR_BGR2RGB)
        return data

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load Image (Required)
        img = self._load_img(row['image'])
        
        # 2. Load Mask (Optional)
        mask = None
        if 'mask' in self.df.columns and pd.notna(row['mask']):
            mask = self._load_img(row['mask'], is_mask=True)

        # 3. Apply Augmentation/Transform
        if self.augment or self.transform:
            payload = {"image": img}
            if mask is not None:
                payload["mask"] = mask
                
            if self.augment:
                payload = self.augment(**payload)
            if self.transform:
                payload = self.transform(**payload)
                
            img = payload["image"]
            mask = payload.get("mask")

        # 4. Conversion to Tensors
        img_tensor = torch.from_numpy(img).float()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.permute(2, 0, 1) # HWC -> CHW
            
        results = (img_tensor,)

        # 5. Append Mask to Tuple
        if mask is not None:
            # Cast mask to Long (for classes) or Float (for binary)
            m_tensor = torch.from_numpy(mask).long() if mask.ndim == 2 else torch.from_numpy(mask).float()
            results += (m_tensor,)

        # 6. Append Label to Tuple (Int starting at 0)
        if self.label_map and 'label' in self.df.columns and pd.notna(row['label']):
            label_idx = self.label_map[row['label']]
            results += (torch.tensor(label_idx, dtype=torch.long),)

        return results