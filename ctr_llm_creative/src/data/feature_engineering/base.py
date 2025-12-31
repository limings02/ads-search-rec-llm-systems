"""Base feature engineer class."""

from abc import ABC, abstractmethod
from typing import Any, Dict
from contracts import FeatureMap


class BaseFeatureEngineer(ABC):
    """Base class for feature engineers."""
    
    def __init__(self, feature_map: FeatureMap):
        self.feature_map = feature_map
    
    @abstractmethod
    def fit(self, data: Any) -> None:
        """
        Fit feature engineer on training data.
        
        Args:
            data: Training data
        """
        pass
    
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """
        Transform data using fitted feature map.
        
        Args:
            data: Data to transform
        
        Returns:
            Transformed data
        """
        pass
    
    def fit_transform(self, data: Any) -> Any:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)
