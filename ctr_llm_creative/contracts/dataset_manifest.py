"""Dataset manifest schema: defines dataset structure, tasks, splits, and labels."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json


class TaskType(str, Enum):
    """Task type enumeration."""
    CTR = "ctr"
    CVR = "cvr"
    CTCVR = "ctcvr"
    MULTITASK = "multitask"


class SplitType(str, Enum):
    """Data split type enumeration."""
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


@dataclass
class LabelSpec:
    """Label specification."""
    name: str
    task_type: TaskType
    positive_value: Any = 1
    negative_value: Any = 0
    description: Optional[str] = None


@dataclass
class TaskSpec:
    """Task specification."""
    name: str
    task_type: TaskType
    labels: List[LabelSpec] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class SplitSpec:
    """Data split specification."""
    name: SplitType
    path: str
    size: Optional[int] = None
    description: Optional[str] = None


@dataclass
class DatasetManifest:
    """
    Dataset manifest: defines the structure, tasks, splits, and metadata of a dataset.
    """
    name: str
    version: str
    dataset_type: str  # e.g., "avazu", "ali_ccp", "ipinyou", "criteo_attr"
    description: Optional[str] = None
    
    # Dataset location and format
    raw_path: Optional[str] = None
    processed_path: Optional[str] = None
    format: str = "csv"  # csv, parquet, arrow, etc.
    
    # Fields and schema
    feature_fields: List[str] = field(default_factory=list)
    label_fields: List[str] = field(default_factory=list)
    id_field: Optional[str] = None
    timestamp_field: Optional[str] = None
    
    # Tasks
    tasks: List[TaskSpec] = field(default_factory=list)
    
    # Splits
    splits: List[SplitSpec] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        data = asdict(self)
        # Convert Enums to strings
        data["tasks"] = [
            {**asdict(task), "task_type": task.task_type.value,
             "labels": [{**asdict(label), "task_type": label.task_type.value} 
                       for label in task.labels]}
            for task in self.tasks
        ]
        data["splits"] = [
            {**asdict(split), "name": split.name.value} 
            for split in self.splits
        ]
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "DatasetManifest":
        """Create from JSON string."""
        data = json.loads(json_str)
        # Reconstruct objects
        if "tasks" in data:
            data["tasks"] = [
                TaskSpec(
                    name=task["name"],
                    task_type=TaskType(task["task_type"]),
                    labels=[
                        LabelSpec(
                            name=label["name"],
                            task_type=TaskType(label["task_type"]),
                            positive_value=label.get("positive_value", 1),
                            negative_value=label.get("negative_value", 0),
                            description=label.get("description")
                        )
                        for label in task.get("labels", [])
                    ],
                    description=task.get("description")
                )
                for task in data["tasks"]
            ]
        if "splits" in data:
            data["splits"] = [
                SplitSpec(
                    name=SplitType(split["name"]),
                    path=split["path"],
                    size=split.get("size"),
                    description=split.get("description")
                )
                for split in data["splits"]
            ]
        return cls(**data)
