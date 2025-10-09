"""Forcing functions."""

from .base import ForcingFunction, MultiDoseForcing, OnOffForcing, PeriodicForcing

try:
    from . import scipy_functions
except ImportError:
    scipy_functions = None

try:
    from . import jax_functions
except ImportError:
    jax_functions = None

__all__ = ["ForcingFunction", "MultiDoseForcing", "OnOffForcing", "PeriodicForcing"]