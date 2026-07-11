"""Base forcing functions interface.

Continuous forcing capabilities (OnOff, PerDose/Periodic, NDoses) are available
through the ``assign_forcing_function()`` model API and ``UnifiedForcingFactory``.
The ``InterpolatedForcing`` class provides programmatic instantiation for
data-driven interpolated forcing functions.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable


class ForcingFunction(ABC):
    """Abstract base class for forcing functions."""

    @abstractmethod
    def create_function(self, backend: str = "scipy") -> Callable:
        """Create the forcing function callable for the specified backend."""
        pass  # pragma: no cover

    @abstractmethod
    def get_switch_times(self, t_start: float, t_end: float) -> list[float]:
        """Get times when this forcing function changes behavior (discontinuities)."""
        pass  # pragma: no cover


class BackendAwareForcing(ForcingFunction):
    """Base class for forcing functions that cache compiled backend callables."""

    def __init__(self):
        self._cached_functions: dict[str, Callable] = {}

    def create_function(self, backend: str = "scipy") -> Callable:
        """Create and cache forcing function for the specified backend."""
        if backend not in self._cached_functions:
            self._cached_functions[backend] = self._create_backend_function(backend)
        return self._cached_functions[backend]

    def invalidate_cache(self) -> None:
        """Clear the backend function cache."""
        self._cached_functions.clear()

    @abstractmethod
    def _create_backend_function(self, backend: str) -> Callable:
        """Create the backend-specific function implementation."""
        pass  # pragma: no cover


def create_forcing_function(forcing_type: str, **kwargs) -> ForcingFunction:
    """
    Factory function to create forcing function objects.

    Currently supports 'interpolated'. For OnOff, PerDose/periodic, and
    NDoses/multidose continuous forcing, use ``assign_forcing_function()``
    on a model instance, or call ``UnifiedForcingFactory`` directly.

    Args:
        forcing_type: Type of forcing function. Currently: 'interpolated'.
        **kwargs: Arguments specific to the forcing function type.

    Returns:
        ForcingFunction instance.

    Examples:
        # Create interpolated forcing from arrays
        forcing = create_forcing_function('interpolated', times=[0, 1, 2], values=[10, 20, 30])

        # Create interpolated forcing from DataFrame
        forcing = create_forcing_function('interpolated',
                                          times=df['time'].values,
                                          values=df['bw'].values)
    """
    if forcing_type in ("interpolated", "interp"):
        from .interpolated import InterpolatedForcing

        return InterpolatedForcing(**kwargs)
    else:
        available_types = ["interpolated"]
        raise ValueError(
            f"Unknown forcing type: {forcing_type!r}. Available: {available_types}. "
            "For OnOff/PerDose/NDoses continuous forcing, use "
            "model.assign_forcing_function() or UnifiedForcingFactory directly."
        )


__all__ = [
    "BackendAwareForcing",
    "ForcingFunction",
    "create_forcing_function",
]
