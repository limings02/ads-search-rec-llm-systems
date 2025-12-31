"""Feature map schema: defines feature transformations (hash, vocab, bucket, normalize, seq)."""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json


class FeatureType(str, Enum):
    """Feature data type enumeration."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    EMBEDDING = "embedding"
    SEQUENCE = "sequence"


class TransformType(str, Enum):
    """Feature transformation type enumeration."""
    NONE = "none"
    HASH = "hash"
    VOCAB = "vocab"
    BUCKET = "bucket"
    NORMALIZE = "normalize"
    LOG = "log"
    SEQUENCE = "sequence"


@dataclass
class TransformSpec:
    """Feature transformation specification."""
    type: TransformType
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Examples for common transforms:
    # HASH: {"size": 10000}
    # VOCAB: {"vocab_size": 1000, "vocab": [...]}
    # BUCKET: {"boundaries": [0, 10, 100, 1000]}
    # NORMALIZE: {"mean": 0.5, "std": 0.1}
    # SEQUENCE: {"max_len": 100, "sep": ","}


@dataclass
class FeatureSpec:
    """Individual feature specification."""
    name: str
    feature_type: FeatureType
    raw_type: str = "string"  # Original data type: string, int, float, etc.
    description: Optional[str] = None
    transforms: List[TransformSpec] = field(default_factory=list)
    output_size: Optional[int] = None  # For embeddings or hashed features
    
    # Field mapping (in case raw field name differs from feature name)
    raw_field_name: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureMap:
    """
    Feature map: defines all features and their transformations.
    Ensures consistent feature engineering across train/valid/test.
    """
    name: str
    version: str
    description: Optional[str] = None
    
    # Feature specifications
    features: List[FeatureSpec] = field(default_factory=list)
    
    # Total feature dimension after transformations
    total_dim: Optional[int] = None
    
    # Fitting metadata (from training data)
    fitted_at: Optional[str] = None  # Timestamp
    fitted_on: Optional[str] = None  # Dataset name
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.features:
            self.features = []
    
    def get_feature(self, name: str) -> Optional[FeatureSpec]:
        """Get feature by name."""
        for feature in self.features:
            if feature.name == name:
                return feature
        return None
    
    def add_feature(self, feature: FeatureSpec) -> None:
        """Add a feature."""
        self.features.append(feature)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        data = asdict(self)
        # Convert Enums to strings
        data["features"] = [
            {**asdict(f),
             "feature_type": f.feature_type.value,
             "transforms": [
                 {**asdict(t), "type": t.type.value}
                 for t in f.transforms
             ]}
            for f in self.features
        ]
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "FeatureMap":
        """Create from JSON string."""
        data = json.loads(json_str)
        if "features" in data:
            data["features"] = [
                FeatureSpec(
                    name=f["name"],
                    feature_type=FeatureType(f["feature_type"]),
                    raw_type=f.get("raw_type", "string"),
                    description=f.get("description"),
                    transforms=[
                        TransformSpec(
                            type=TransformType(t["type"]),
                            params=t.get("params", {})
                        )
                        for t in f.get("transforms", [])
                    ],
                    output_size=f.get("output_size"),
                    raw_field_name=f.get("raw_field_name"),
                    metadata=f.get("metadata", {})
                )
                for f in data["features"]
            ]
        return cls(**data)
