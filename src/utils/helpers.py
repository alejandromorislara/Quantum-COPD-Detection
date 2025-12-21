"""
Utility functions for the Quantum Respiratory Classification project.
"""
import numpy as np
import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable
import torch

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import RANDOM_STATE


def set_seed(seed: int = RANDOM_STATE) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch not installed


def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Usage:
        @timer
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{func.__name__} executed in {elapsed:.2f} seconds")
        return result
    return wrapper


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Block"):
        self.name = name
        self.start_time = None
        self.elapsed = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        print(f"{self.name} executed in {self.elapsed:.2f} seconds")


def save_model(model: Any, path: Path) -> None:
    """
    Save a model to disk.
    
    Args:
        model: Model to save
        path: Path to save the model
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if str(path).endswith(".pt") or str(path).endswith(".pth"):
        # PyTorch model
        torch.save(model.state_dict() if hasattr(model, "state_dict") else model, path)
    else:
        # Pickle for sklearn and other models
        with open(path, "wb") as f:
            pickle.dump(model, f)
    
    print(f"Model saved to {path}")


def load_model(path: Path, model_class: Any = None) -> Any:
    """
    Load a model from disk.
    
    Args:
        path: Path to the saved model
        model_class: Optional model class for PyTorch models
        
    Returns:
        Loaded model
    """
    path = Path(path)
    
    if str(path).endswith(".pt") or str(path).endswith(".pth"):
        # PyTorch model
        if model_class is not None:
            model = model_class()
            model.load_state_dict(torch.load(path))
            return model
        else:
            return torch.load(path)
    else:
        # Pickle
        with open(path, "rb") as f:
            return pickle.load(f)


def normalize_to_range(x: np.ndarray, 
                       min_val: float = -np.pi, 
                       max_val: float = np.pi) -> np.ndarray:
    """
    Normalize array to a specific range (default: [-pi, pi] for quantum encoding).
    
    Args:
        x: Input array
        min_val: Minimum value of output range
        max_val: Maximum value of output range
        
    Returns:
        Normalized array
    """
    x_min = x.min()
    x_max = x.max()
    
    if x_max - x_min == 0:
        return np.zeros_like(x) + (min_val + max_val) / 2
    
    return (x - x_min) / (x_max - x_min) * (max_val - min_val) + min_val


def print_model_summary(model_name: str, params: dict) -> None:
    """Print a formatted summary of model parameters."""
    print("\n" + "=" * 50)
    print(f"Model: {model_name}")
    print("=" * 50)
    for key, value in params.items():
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")

