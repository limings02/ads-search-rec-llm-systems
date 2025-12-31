"""Reproducibility: seed management, hash computation, environment capture."""

import os
import random
import hashlib
import json
from typing import Dict, Any, Optional
import subprocess


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def compute_dict_hash(data: Dict[str, Any]) -> str:
    """Compute hash of a dictionary."""
    json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(json_str.encode()).hexdigest()


def get_python_version() -> str:
    """Get Python version."""
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_platform_info() -> str:
    """Get platform information."""
    import platform
    return platform.platform()


def get_cuda_info() -> Optional[str]:
    """Get CUDA version if available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_dependencies_summary() -> Dict[str, str]:
    """Get summary of key dependencies."""
    deps = {}
    
    packages = ["torch", "numpy", "pandas", "scikit-learn", "hydra-core"]
    
    for pkg in packages:
        try:
            mod = __import__(pkg.replace("-", "_"))
            if hasattr(mod, "__version__"):
                deps[pkg] = mod.__version__
        except ImportError:
            pass
    
    return deps
