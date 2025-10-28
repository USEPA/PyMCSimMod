"""Base forcing functions interface."""

from abc import ABC, abstractmethod
from collections.abc import Callable


class ForcingFunction(ABC):
    """Base class for forcing functions."""

    @abstractmethod
    def create_function(self, backend: str = "scipy") -> Callable:
        """Create the forcing function for the specified backend."""
        pass

    @abstractmethod
    def get_switch_times(self, t_start: float, t_end: float) -> list[float]:
        """Get times when this forcing function changes behavior."""
        pass


class BackendAwareForcing(ForcingFunction):
    """Base class for forcing functions that are backend-aware with caching."""

    def __init__(self):
        self._cached_functions = {}

    def create_function(self, backend: str = "scipy") -> Callable:
        """Create and cache forcing function for specified backend."""
        if backend not in self._cached_functions:
            self._cached_functions[backend] = self._create_backend_function(backend)
        return self._cached_functions[backend]

    @abstractmethod
    def _create_backend_function(self, backend: str) -> Callable:
        """Create the backend-specific function implementation."""
        pass


class OnOffForcing(BackendAwareForcing):
    """On-off forcing function."""

    def __init__(self, t0: float, t1: float, s: float = 10.0):
        super().__init__()
        self.t0 = t0
        self.t1 = t1
        self.s = s

    def _create_backend_function(self, backend: str) -> Callable:
        """Create the on-off function for the specified backend."""
        from .unified import UnifiedForcingFactory

        return UnifiedForcingFactory.create_onoff(self.t0, self.t1, self.s, backend)

    def get_switch_times(self, t_start: float, t_end: float) -> list[float]:
        """Get switch times for this forcing function."""
        times = []
        if t_start <= self.t0 <= t_end:
            times.append(self.t0)
        if t_start <= self.t1 <= t_end:
            times.append(self.t1)
        return times


class PeriodicForcing(BackendAwareForcing):
    """Periodic dosing forcing function."""

    def __init__(self, t0: float, duration: float, period: float, s: float = 10.0):
        super().__init__()
        self.t0 = t0
        self.duration = duration
        self.period = period
        self.s = s

    def _create_backend_function(self, backend: str) -> Callable:
        """Create the periodic function for the specified backend."""
        from .unified import UnifiedForcingFactory

        return UnifiedForcingFactory.create_perdose(
            self.t0, self.duration, self.period, self.s, backend
        )

    def get_switch_times(self, t_start: float, t_end: float) -> list[float]:
        """Get switch times for periodic dosing."""
        times = []
        n = 0
        while True:
            on = self.t0 + n * self.period
            off = on + self.duration
            if on > t_end:
                break
            if on >= t_start:
                times.append(on)
            if off >= t_start and off <= t_end:
                times.append(off)
            n += 1
        return times


class MultiDoseForcing(BackendAwareForcing):
    """Multiple discrete dose forcing function."""

    def __init__(self, t0_list: list[float], duration: float, s: float = 10.0):
        super().__init__()
        self.t0_list = t0_list
        self.duration = duration
        self.s = s

    def _create_backend_function(self, backend: str) -> Callable:
        """Create the multi-dose function for the specified backend."""
        from .unified import UnifiedForcingFactory

        return UnifiedForcingFactory.create_ndoses(self.t0_list, self.duration, self.s, backend)

    def get_switch_times(self, t_start: float, t_end: float) -> list[float]:
        """Get switch times for multiple doses."""
        times = []
        for t0 in self.t0_list:
            on = t0
            off = t0 + self.duration
            if on >= t_start and on <= t_end:
                times.append(on)
            if off >= t_start and off <= t_end:
                times.append(off)
        return times


def create_forcing_function(forcing_type: str, **kwargs) -> ForcingFunction:
    """
    Factory function to create forcing functions.

    Args:
        forcing_type: Type of forcing function ('onoff', 'perdose', 'ndoses', 'interpolated', etc.)
        **kwargs: Arguments specific to the forcing function type

    Returns:
        ForcingFunction instance

    Examples:
        # Create on-off forcing
        forcing = create_forcing_function('onoff', t0=1.0, t1=5.0, s=10.0)

        # Create periodic dosing
        forcing = create_forcing_function('perdose', t0=0.0, duration=1.0, period=24.0)

        # Create interpolated from arrays
        forcing = create_forcing_function('interpolated', times=[0, 1, 2], values=[10, 20, 30])
    """
    if forcing_type == "onoff":
        return OnOffForcing(**kwargs)
    elif forcing_type == "perdose" or forcing_type == "periodic":
        return PeriodicForcing(**kwargs)
    elif forcing_type == "ndoses" or forcing_type == "multidose":
        return MultiDoseForcing(**kwargs)
    elif forcing_type == "interpolated":
        from .interpolated import InterpolatedForcing

        return InterpolatedForcing(**kwargs)
    else:
        available_types = ["onoff", "perdose", "periodic", "ndoses", "multidose", "interpolated"]
        raise ValueError(f"Unknown forcing type: {forcing_type}. Available: {available_types}")


__all__ = [
    "BackendAwareForcing",
    "ForcingFunction",
    "MultiDoseForcing",
    "OnOffForcing",
    "PeriodicForcing",
    "create_forcing_function",
]
