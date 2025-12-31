"""Metrics schema: defines offline, simulation, and CI metrics."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import json


@dataclass
class OfflineMetrics:
    """Offline metrics (computed on validation/test data)."""
    auc: Optional[float] = None
    logloss: Optional[float] = None
    ece: Optional[float] = None  # Expected Calibration Error
    mce: Optional[float] = None  # Maximum Calibration Error
    gini: Optional[float] = None
    ks: Optional[float] = None
    
    # Per-group metrics
    group_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Custom metrics
    custom: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationMetrics:
    """Simulation metrics (computed during auction simulation)."""
    total_spend: float = 0.0
    total_impressions: int = 0
    total_clicks: int = 0
    total_conversions: int = 0
    ctr: Optional[float] = None
    cvr: Optional[float] = None
    rpm: Optional[float] = None  # Revenue per mille
    ecpc: Optional[float] = None  # Effective cost per click
    ecpa: Optional[float] = None  # Effective cost per action
    
    # Budget related
    budget_utilization: Optional[float] = None
    budget_id: Optional[str] = None
    
    # Custom metrics
    custom: Dict[str, float] = field(default_factory=dict)


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval."""
    statistic: float  # Point estimate
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95
    n_bootstrap: int = 10000


@dataclass
class SignificanceTest:
    """Statistical significance test result."""
    baseline_metric: float
    treatment_metric: float
    metric_name: str
    p_value: float
    ci: Optional[BootstrapCI] = None
    significant: Optional[bool] = None  # Determined by p_value < alpha
    alpha: float = 0.05


@dataclass
class Metrics:
    """
    Unified metrics container for a single run.
    Contains offline metrics, simulation metrics, and CI/significance results.
    """
    run_id: str
    timestamp: str
    
    # Offline stage metrics
    offline: Optional[OfflineMetrics] = None
    
    # Simulation stage metrics
    simulation: Optional[SimulationMetrics] = None
    
    # Significance test results (comparing to baseline)
    significance_tests: List[SignificanceTest] = field(default_factory=list)
    
    # Additional metadata
    dataset_name: Optional[str] = None
    model_name: Optional[str] = None
    config_hash: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_significance_test(self, test: SignificanceTest) -> None:
        """Add a significance test result."""
        self.significance_tests.append(test)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        if self.offline:
            data["offline"] = asdict(self.offline)
        if self.simulation:
            data["simulation"] = asdict(self.simulation)
        if self.significance_tests:
            data["significance_tests"] = [
                {**asdict(test), 
                 "ci": asdict(test.ci) if test.ci else None}
                for test in self.significance_tests
            ]
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Metrics":
        """Create from JSON string."""
        data = json.loads(json_str)
        
        # Reconstruct nested objects
        if data.get("offline"):
            data["offline"] = OfflineMetrics(**data["offline"])
        if data.get("simulation"):
            data["simulation"] = SimulationMetrics(**data["simulation"])
        if data.get("significance_tests"):
            data["significance_tests"] = [
                SignificanceTest(
                    **{k: v for k, v in test.items() if k != "ci"},
                    ci=BootstrapCI(**test["ci"]) if test.get("ci") else None
                )
                for test in data["significance_tests"]
            ]
        
        return cls(**data)
