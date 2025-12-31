"""Training callbacks: early stopping, checkpointing, logging."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class Callback(ABC):
    """Base callback class."""
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: dict) -> bool:
        """
        Called at end of epoch.
        
        Returns:
            True to continue, False to stop training
        """
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping callback."""
    
    def __init__(self, patience: int = 5, metric: str = "valid_loss"):
        self.patience = patience
        self.metric = metric
        self.best_value = float("inf")
        self.wait_count = 0
    
    def on_epoch_end(self, epoch: int, logs: dict) -> bool:
        """Check early stopping condition."""
        # Implementation will be added
        pass


class CheckpointCallback(Callback):
    """Save best model."""
    
    def __init__(self, save_dir: str, metric: str = "valid_auc", mode: str = "max"):
        self.save_dir = save_dir
        self.metric = metric
        self.mode = mode
    
    def on_epoch_end(self, epoch: int, logs: dict) -> bool:
        """Save checkpoint if better."""
        # Implementation will be added
        pass
