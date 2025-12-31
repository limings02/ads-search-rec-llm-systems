"""Contracts I/O: read/write and validate contracts."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Type, TypeVar

from contracts import (
    DatasetManifest,
    FeatureMap,
    AuctionStream,
    Metrics,
    RunMeta,
)


T = TypeVar("T")


class ContractsIO:
    """Utility class for reading and writing contracts."""
    
    @staticmethod
    def save_json(obj: Any, path: str) -> None:
        """Save object to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if hasattr(obj, "to_json"):
            content = obj.to_json()
        else:
            content = json.dumps(obj, indent=2, ensure_ascii=False)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    
    @staticmethod
    def load_json(path: str, target_class: Type[T]) -> T:
        """Load JSON file and convert to target class."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if hasattr(target_class, "from_json"):
            return target_class.from_json(content)
        else:
            data = json.loads(content)
            return target_class(**data)
    
    @staticmethod
    def save_dataset_manifest(manifest: DatasetManifest, path: str) -> None:
        """Save dataset manifest."""
        ContractsIO.save_json(manifest, path)
    
    @staticmethod
    def load_dataset_manifest(path: str) -> DatasetManifest:
        """Load dataset manifest."""
        return ContractsIO.load_json(path, DatasetManifest)
    
    @staticmethod
    def save_feature_map(feature_map: FeatureMap, path: str) -> None:
        """Save feature map."""
        ContractsIO.save_json(feature_map, path)
    
    @staticmethod
    def load_feature_map(path: str) -> FeatureMap:
        """Load feature map."""
        return ContractsIO.load_json(path, FeatureMap)
    
    @staticmethod
    def save_metrics(metrics: Metrics, path: str) -> None:
        """Save metrics."""
        ContractsIO.save_json(metrics, path)
    
    @staticmethod
    def load_metrics(path: str) -> Metrics:
        """Load metrics."""
        return ContractsIO.load_json(path, Metrics)
    
    @staticmethod
    def save_run_meta(run_meta: RunMeta, path: str) -> None:
        """Save run metadata."""
        ContractsIO.save_json(run_meta, path)
    
    @staticmethod
    def load_run_meta(path: str) -> RunMeta:
        """Load run metadata."""
        return ContractsIO.load_json(path, RunMeta)
