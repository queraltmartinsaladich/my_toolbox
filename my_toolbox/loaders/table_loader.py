import pandas as pd
from torch.utils.data import Dataset
import torch

# ==========================================
# EXAMPLE USAGE: TableDataset
# ==========================================
# 1. For Regression (e.g., Housing Prices):
# dataset = TableDataset(csv_path="prices.csv", task="regression", target_col="price")
# loader = DataLoader(dataset, batch_size=64)
# features, target = next(iter(loader))
# -> target.dtype will be torch.float32

# 2. For Classification (e.g., Churn Prediction):
# dataset = TableDataset(csv_path="users.csv", task="classification", target_col="churned")
# loader = DataLoader(dataset, batch_size=64)
# features, target = next(iter(loader))
# -> target.dtype will be torch.long (class indices)
# ==========================================

class TableDataset(Dataset):
    def __init__(self, csv_path, task, target_col):
        df = pd.read_csv(csv_path)
        self.X = df.drop(columns=[target_col]).values
        self.y = df[target_col].values
        self.task = task

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Regression targets usually need to be float32
        target_dtype = torch.float32 if self.task == "regression" else torch.long
        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=target_dtype)