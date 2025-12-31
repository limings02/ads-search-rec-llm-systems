"""Registry: register dataset adapters, models, simulators."""

from typing import Dict, Type, Any, Callable, Optional


class Registry:
    """Simple registry for plugins."""
    
    def __init__(self):
        self._registry: Dict[str, Any] = {}
    
    def register(self, name: str, obj: Any) -> None:
        """Register an object."""
        if name in self._registry:
            raise ValueError(f"Already registered: {name}")
        self._registry[name] = obj
    
    def get(self, name: str) -> Any:
        """Get registered object."""
        if name not in self._registry:
            raise ValueError(f"Not registered: {name}")
        return self._registry[name]
    
    def list(self) -> Dict[str, Any]:
        """List all registered objects."""
        return dict(self._registry)
    
    def __contains__(self, name: str) -> bool:
        """Check if registered."""
        return name in self._registry


# Global registries
dataset_adapters = Registry()
models = Registry()
simulators = Registry()
