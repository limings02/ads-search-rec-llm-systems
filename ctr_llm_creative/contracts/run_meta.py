"""Run metadata schema: captures git, config, environment, seed, and hash info."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import json


@dataclass
class GitInfo:
    """Git repository information."""
    commit: str
    branch: str
    dirty: bool = False
    remotes: Dict[str, str] = field(default_factory=dict)


@dataclass
class EnvironmentInfo:
    """Environment and dependency information."""
    python_version: str
    platform: str
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    
    # Dependencies summary (sample of key packages)
    dependencies: Dict[str, str] = field(default_factory=dict)


@dataclass
class RunMeta:
    """
    Run metadata: captures all metadata for reproducibility.
    Stored in runs/{run_id}/run_meta.json
    """
    run_id: str
    timestamp: str  # ISO format
    
    # Git info
    git: Optional[GitInfo] = None
    
    # Environment
    environment: Optional[EnvironmentInfo] = None
    
    # Config references
    config_file: Optional[str] = None  # Path to the config used
    config_hash: Optional[str] = None  # SHA256 hash of config
    
    # Reproducibility
    seed: Optional[int] = None
    random_state: Optional[int] = None
    
    # Run type
    run_type: str = "experiment"  # experiment, debug, benchmark, etc.
    
    # Tags and notes
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        if self.git:
            data["git"] = asdict(self.git)
        if self.environment:
            data["environment"] = asdict(self.environment)
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "RunMeta":
        """Create from JSON string."""
        data = json.loads(json_str)
        
        # Reconstruct nested objects
        if data.get("git"):
            data["git"] = GitInfo(**data["git"])
        if data.get("environment"):
            data["environment"] = EnvironmentInfo(**data["environment"])
        
        return cls(**data)
