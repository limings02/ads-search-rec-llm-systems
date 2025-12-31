"""API routes for runs."""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("/")
async def list_runs() -> List[Dict[str, Any]]:
    """List all runs."""
    # Implementation will be added
    pass


@router.get("/{run_id}")
async def get_run(run_id: str) -> Dict[str, Any]:
    """Get run details."""
    # Implementation will be added
    pass


@router.get("/{run_id}/metrics")
async def get_metrics(run_id: str) -> Dict[str, Any]:
    """Get run metrics."""
    # Implementation will be added
    pass


@router.get("/{run_id}/artifacts")
async def list_artifacts(run_id: str) -> List[str]:
    """List run artifacts."""
    # Implementation will be added
    pass


@router.post("/compare")
async def compare_runs(run_ids: List[str]) -> Dict[str, Any]:
    """Compare multiple runs."""
    # Implementation will be added
    pass
