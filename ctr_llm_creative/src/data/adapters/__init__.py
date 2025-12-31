"""Data adapter base class."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from contracts import DatasetManifest


class BaseAdapter(ABC):
    """Base class for dataset adapters."""
    
    def __init__(self, manifest: DatasetManifest):
        self.manifest = manifest
    
    @abstractmethod
    def load_split(self, split: str) -> Any:
        """
        Load data for a split.
        
        Args:
            split: 'train', 'valid', or 'test'
        
        Returns:
            Loaded data (format depends on implementation)
        """
        pass
    
    @abstractmethod
    def get_features(self) -> Dict[str, str]:
        """
        Get feature names and types.
        
        Returns:
            Dictionary of {feature_name: feature_type}
        """
        pass
