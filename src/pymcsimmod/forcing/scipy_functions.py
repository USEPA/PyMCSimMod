"""Scipy-specific forcing function implementations (backward compatibility)."""

# Import unified implementations
from .unified import (
    create_constantfunc as unified_create_constantfunc,
)
from .unified import (
    create_ndoses as unified_create_ndoses,
)
from .unified import (
    create_onoff as unified_create_onoff,
)
from .unified import (
    create_perdose as unified_create_perdose,
)
from .unified import (
    create_zerofunc as unified_create_zerofunc,
)


def create_onoff(t0: float, t1: float, s: float = 10.0):
    """
    Create an on-off forcing function for scipy backend.

    Args:
        t0: Time when function turns on
        t1: Time when function turns off
        s: Smoothing parameter for transitions (default: 10.0)

    Returns:
        Function that takes time t and returns on/off value
    """
    return unified_create_onoff(t0, t1, s, backend="scipy")


def create_perdose(t0: float, duration: float, period: float, s: float = 10.0):
    """
    Create a periodic dosing forcing function for scipy backend.

    Args:
        t0: Time of first dose
        duration: Duration of each dose
        period: Period between doses
        s: Smoothing parameter for transitions (default: 10.0)

    Returns:
        Function that takes time t and returns dose value
    """
    return unified_create_perdose(t0, duration, period, s, backend="scipy")


def create_ndoses(t0_list: list[float], duration: float, s: float = 10.0):
    """
    Create a multiple discrete dose forcing function for scipy backend.

    Args:
        t0_list: List of dose start times
        duration: Duration of each dose
        s: Smoothing parameter for transitions (default: 10.0)

    Returns:
        Function that takes time t and returns total dose value
    """
    return unified_create_ndoses(t0_list, duration, s, backend="scipy")


def create_zerofunc():
    """
    Create a zero forcing function for scipy backend.

    Returns:
        Function that always returns 0.0
    """
    return unified_create_zerofunc(backend="scipy")


def create_constantfunc(val: float):
    """
    Create a forcing function with a constant value for scipy backend.

    Returns:
        Function that always returns val
    """
    return unified_create_constantfunc(val, backend="scipy")


__all__ = [
    "create_constantfunc",
    "create_ndoses",
    "create_onoff",
    "create_perdose",
    "create_zerofunc",
]
