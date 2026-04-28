import os
import re
import json
import unicodedata
import pandas as pd
import logging
import numpy as np
from typing import List, Union, Dict, Any
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, lowercase: bool = True, rem_numbers: bool = False, rem_punct: bool = False):
        self.lowercase = lowercase
        self.rem_numbers = rem_numbers
        self.rem_punct = rem_punct
        self.html_pattern = re.compile('<.*?>', re.DOTALL)

    def clean_text(self, text: str) -> str:
        """Core cleaning logic for a single string."""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Unicode Normalization (Fixes accents and weird symbols)
        text = unicodedata.normalize('NFKC', text)
        
        # Remove non-printable control characters
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
        
        # HTML removal
        text = re.sub(self.html_pattern, '', text)
        
        # Transformation Rules
        if self.lowercase:
            text = text.lower()
        if self.rem_numbers:
            text = re.sub(r'\d+', '', text)
        if self.rem_punct:        
            # Replaces punctuation with a space to avoid joining words incorrectly
            text = re.sub(r"[^\w\s]", " ", text)
            
        text = text.replace("_", " ")
        # Collapse multiple spaces and trim
        return re.sub(r"\s+", " ", text).strip()

    # --- LOADING LOGIC ---
    @staticmethod
    def load_any_file(file_path: str) -> Union[pd.DataFrame, str, List, Dict]:
        """Generic loader for different formats."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Path not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Extension {ext} not supported.")

    # --- ORCHESTRATION ---
    def process(self, input_data: Any, target_cols: List[str] = None) -> Any:
        """
        Recursive orchestration: handles DataFrames, Lists, Dicts, and Strings.
        """
        # 1. DataFrames
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
            # If no target_cols provided, clean all string (object) columns
            cols = target_cols if target_cols else df.select_dtypes(include=['object']).columns
            for col in cols:
                df[col] = df[col].astype(str).apply(self.clean_text)
            return df

        # 2. Lists
        elif isinstance(input_data, list):
            return [self.process(item, target_cols) for item in input_data]

        # 3. Dictionaries
        elif isinstance(input_data, dict):
            return {k: (self.process(v, target_cols) if isinstance(v, (str, list, dict)) else v) 
                    for k, v in input_data.items()}

        # 4. Raw Strings
        elif isinstance(input_data, str):
            return self.clean_text(input_data)

        return input_data

def prepare_labels(y: Union[List, pd.Series, np.ndarray]):
    """
    Converts categorical labels into integers and returns the encoder for inverse mapping.
    Example: ['A', 'B'] -> [0, 1]
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Extract mapping for logging
    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    logger.info(f"Label Mapping: {mapping}")
    
    return y_encoded, encoder