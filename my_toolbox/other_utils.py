"""
GENERAL UTILITIES MODULE (other_utils.py)
-----------------------------------------
Purpose: Supporting functions for configuration, timing, and system health.
Features: 
    - YAML/JSON Config Management
    - Execution Timing Decorators
    - Device and System Information
    - Data Auditing (Health Checks)
"""

import time
import json
import yaml
import torch
import logging
import platform
from functools import wraps
from pathlib import Path

logger = logging.getLogger(__name__)

# --- 1. PERFORMANCE MONITORING ---

def time_it(func):
    """Decorator to measure the execution time of any function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# --- 2. CONFIGURATION MANAGEMENT ---

def load_config(config_path: str):
    """Loads parameters from YAML or JSON files."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError("Unsupported config format. Use .yaml or .json")

# --- 3. SYSTEM & HARDWARE AUDIT ---

def get_system_info():
    """Returns a dictionary of hardware and environment details."""
    info = {
        "os": platform.system(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()}
    
    if info["cuda_available"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    
    for k, v in info.items():
        logger.info(f"{k.upper()}: {v}")
    return info

# --- 4. DATA INTEGRITY CHECKS ---

class DataAuditor:
    """Helper to check DataFrame health before training."""
    @staticmethod
    def audit_dataframe(df, label_col=None):
        """Checks for missing values, duplicates, and class balance."""
        report = {
            "total_rows": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
        }
        
        if label_col and label_col in df.columns:
            report["class_distribution"] = df[label_col].value_counts().to_dict()
        
        logger.info("--- Data Audit Report ---")
        logger.info(json.dumps(report, indent=4))
        return report

# --- 5. PATH FIXER ---

def ensure_dir(directory: str):
    """Thread-safe directory creation."""
    Path(directory).mkdir(parents=True, exist_ok=True)