from typing import Any
from torch.utils.data import DataLoader
from .loaders.image_loader import ImageDataset
from .loaders.table_loader import TableDataset
from .loaders.text_loader import TextDataset

# ==========================================
# EXAMPLE USAGE
# ==========================================
# import pandas as pd
# from my_toolbox import ToolboxDataLoader
# df = pd.DataFrame({
#     "image": ["cat.png", "dog.png"],
#     "label": ["cat", "dog"]})

# loader = ToolboxDataLoader.get_loader(
#     data_type="image",
#     data=df,
#     batch_size=2)

class ToolboxDataLoader:
    """Entry point for generating DataLoaders for various tasks."""
    
    @staticmethod
    def get_loader(
        data_type: str, 
        data: Any,  # This can be a DataFrame (Image/Table) or Path (Text)
        batch_size: int = 32, 
        shuffle: bool = True,
        **kwargs) -> DataLoader:
        """
        Args:
            data_type: 'image', 'table', or 'text'
            data: The input data (DataFrame for Image/Table, file path for Text)
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            **kwargs: Extra args like 'transform', 'augment', or 'tokenizer'
        """
        
        if data_type == "image":
            # 'data' here is the DataFrame with columns: image, mask, label
            dataset = ImageDataset(df=data, **kwargs)
            
        elif data_type == "table":
            # 'data' here is the DataFrame or path to CSV
            dataset = TableDataset(data=data, **kwargs)
            
        elif data_type == "text":
            # 'data' here is the path to .txt, .csv, or .json
            dataset = TextDataset(path=data, **kwargs)
            
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=kwargs.get('num_workers', 0))