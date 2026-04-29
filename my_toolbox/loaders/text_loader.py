from torch.utils.data import Dataset
import torch

# ==========================================
# EXAMPLE USAGE: TextDataset
# ==========================================

# Pre-requisite: A tokenizer (e.g., from HuggingFace)
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 1. UNSUPERVISED (.txt paragraphs)
# dataset = TextDataset(path="essays.txt", tokenizer=tokenizer, task="unsupervised")
# loader = DataLoader(dataset, batch_size=4)
# batch = next(iter(loader))
# -> Returns: {'input_ids': ..., 'attention_mask': ...} (No labels)

# 2. SUPERVISED (.csv with 'text' and 'label' columns)
# dataset = TextDataset(path="reviews.csv", tokenizer=tokenizer, task="classification")
# loader = DataLoader(dataset, batch_size=32)
# batch = next(iter(loader))
# -> Returns: {'input_ids': ..., 'attention_mask': ..., 'labels': ...}

# 3. STRUCTURED DATA (.json)
# dataset = TextDataset(path="data.json", tokenizer=tokenizer)
# loader = DataLoader(dataset, batch_size=16)
# ==========================================

import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path

class TextDataset(Dataset):
    """
    Handles .txt, .csv, and .json for supervised and unsupervised text tasks.
    """
    def __init__(self, path: str, tokenizer, task: str = "classification", max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.task = task
        self.data = []
        self.labels = []
        
        self._load_data(Path(path))

    def _load_data(self, path: Path):
        # Handle .txt (Unsupervised / Paragraphs)
        if path.suffix == ".txt":
            with open(path, 'r', encoding='utf-8') as f:
                # Split by double newline to get paragraphs
                self.data = [p.strip() for p in f.read().split('\n\n') if p.strip()]
                self.labels = None # Unsupervised
        
        # Handle .csv (Supervised)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
            self.data = df['text'].tolist()
            if 'label' in df.columns:
                self.labels = df['label'].tolist()
        
        # Handle .json (Flexible)
        elif path.suffix == ".json":
            with open(path, 'r') as f:
                json_data = json.load(f)
                # Assumes list of dicts: [{"text": "...", "label": 1}]
                self.data = [item['text'] for item in json_data]
                if 'label' in json_data[0]:
                    self.labels = [item['label'] for item in json_data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.data[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt')
        
        item = {'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()}
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item