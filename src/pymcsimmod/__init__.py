"""Perform ODE simulations using MCSim models."""

from .ode import DiscreteEvent, JaxModel, JaxModelEqx, ScipyModel

__version__ = "0.0.1"

__all__ = [
    "DiscreteEvent",
    "JaxModel",
    "JaxModelEqx",
    "ScipyModel",
]
