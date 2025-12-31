"""Budget manager: budget constraints, pacing, daily budgets."""

from typing import Optional, Dict


class BudgetManager:
    """Manage budget constraints during simulation."""
    
    def __init__(self, total_budget: float):
        self.total_budget = total_budget
        self.spent = 0.0
    
    def can_spend(self, amount: float) -> bool:
        """Check if budget allows this spend."""
        return (self.spent + amount) <= self.total_budget
    
    def record_spend(self, amount: float) -> None:
        """Record a spend."""
        if self.can_spend(amount):
            self.spent += amount
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        return self.total_budget - self.spent
    
    def get_utilization(self) -> float:
        """Get budget utilization ratio."""
        return self.spent / self.total_budget if self.total_budget > 0 else 0.0
