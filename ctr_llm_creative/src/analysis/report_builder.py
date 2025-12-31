"""Report builder: generates notes.md, tables, curve plots."""

from typing import Dict, Any, Optional
from pathlib import Path


class ReportBuilder:
    """Build comprehensive experiment report."""
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
    
    def build_report(
        self,
        metrics: Dict[str, Any],
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build full report.
        
        Returns:
            Path to generated notes.md
        """
        # Implementation will be added
        pass
    
    def save_tables(self, tables: Dict[str, str]) -> None:
        """Save metric tables."""
        # Implementation will be added
        pass
    
    def save_curves(self, curves: Dict[str, Any]) -> None:
        """Save curve plots."""
        # Implementation will be added
        pass
