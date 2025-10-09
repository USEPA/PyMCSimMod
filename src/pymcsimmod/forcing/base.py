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


class OnOffForcing(ForcingFunction):
    """On-off forcing function."""
    
    def __init__(self, t0: float, t1: float, s: float = 10.0):
        self.t0 = t0
        self.t1 = t1
        self.s = s
    
    def create_function(self, backend: str = "scipy") -> Callable:
        """Create the on-off function for the specified backend."""
        if backend == "scipy":
            from .scipy_functions import create_onoff
            return create_onoff(self.t0, self.t1, self.s)
        elif backend == "jax":
            from .jax_functions import create_onoff
            return create_onoff(self.t0, self.t1, self.s)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def get_switch_times(self, t_start: float, t_end: float) -> list[float]:
        """Get switch times for this forcing function."""
        times = []
        if t_start <= self.t0 <= t_end:
            times.append(self.t0)
        if t_start <= self.t1 <= t_end:
            times.append(self.t1)
        return times


class PeriodicForcing(ForcingFunction):
    """Periodic dosing forcing function."""
    
    def __init__(self, t0: float, duration: float, period: float, s: float = 10.0):
        self.t0 = t0
        self.duration = duration
        self.period = period
        self.s = s
    
    def create_function(self, backend: str = "scipy") -> Callable:
        """Create the periodic function for the specified backend."""
        if backend == "scipy":
            from .scipy_functions import create_perdose
            return create_perdose(self.t0, self.duration, self.period, self.s)
        elif backend == "jax":
            from .jax_functions import create_perdose
            return create_perdose(self.t0, self.duration, self.period, self.s)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
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


class MultiDoseForcing(ForcingFunction):
    """Multiple discrete dose forcing function."""
    
    def __init__(self, t0_list: list[float], duration: float, s: float = 10.0):
        self.t0_list = t0_list
        self.duration = duration
        self.s = s
    
    def create_function(self, backend: str = "scipy") -> Callable:
        """Create the multi-dose function for the specified backend."""
        if backend == "scipy":
            from .scipy_functions import create_ndoses
            return create_ndoses(self.t0_list, self.duration, self.s)
        elif backend == "jax":
            from .jax_functions import create_ndoses
            return create_ndoses(self.t0_list, self.duration, self.s)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
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


__all__ = ["ForcingFunction", "MultiDoseForcing", "OnOffForcing", "PeriodicForcing"]