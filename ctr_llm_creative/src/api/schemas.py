"""API response schemas."""

from pydantic import BaseModel
from typing import Dict, Any, Optional, List


class RunInfo(BaseModel):
    """Run information."""
    run_id: str
    timestamp: str
    dataset: str
    model: str
    status: str


class MetricsResponse(BaseModel):
    """Metrics response."""
    offline: Optional[Dict[str, float]] = None
    simulation: Optional[Dict[str, float]] = None


class CompareResponse(BaseModel):
    """Run comparison response."""
    runs: List[RunInfo]
    metrics_comparison: Dict[str, Any]
