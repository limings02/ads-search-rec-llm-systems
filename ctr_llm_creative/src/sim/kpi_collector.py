"""KPI collector: aggregates KPIs during simulation."""

from typing import Dict, Any


class KPICollector:
    """Collect KPIs during simulation."""
    
    def __init__(self):
        self.metrics = {
            "total_spend": 0.0,
            "total_impressions": 0,
            "total_clicks": 0,
            "total_conversions": 0,
            "total_revenue": 0.0,
        }
    
    def record_auction_result(
        self,
        winner: bool,
        impression: bool,
        click: bool,
        conversion: bool,
        spend: float,
        value: float,
    ) -> None:
        """Record an auction result."""
        if winner:
            self.metrics["total_spend"] += spend
            self.metrics["total_impressions"] += 1
        
        if impression:
            self.metrics["total_impressions"] += 1
        
        if click:
            self.metrics["total_clicks"] += 1
        
        if conversion:
            self.metrics["total_conversions"] += 1
            self.metrics["total_revenue"] += value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics."""
        metrics = dict(self.metrics)
        
        # Compute derived metrics
        if metrics["total_impressions"] > 0:
            metrics["ctr"] = metrics["total_clicks"] / metrics["total_impressions"]
            metrics["cvr"] = metrics["total_conversions"] / metrics["total_impressions"]
        
        if metrics["total_spend"] > 0:
            metrics["cpc"] = metrics["total_spend"] / max(metrics["total_clicks"], 1)
            metrics["cpa"] = metrics["total_spend"] / max(metrics["total_conversions"], 1)
        
        return metrics
