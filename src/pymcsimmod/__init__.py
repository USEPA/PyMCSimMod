"""Perform ODE simulations using MCSim models."""

from pathlib import Path

# Maintain backward compatibility - import from old locations
from .models import (
    ComputedModel,
    DiscreteEvent,
    EqxModel,
    JaxModel,
    OdeModel,
    ScipyModel,
)

__version__ = "0.0.1"


def create_model(model_source: str | Path, backend: str = "scipy") -> OdeModel:
    """
    Create a model with the specified backend.

    Args:
        model_source: Path to model file or model string
        backend: 'scipy' or 'jax'

    Returns:
        OdeModel instance with the appropriate backend
    """
    if backend == "scipy":
        return ScipyModel(model_source)
    elif backend == "jax":
        return JaxModel(model_source)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'scipy' or 'jax'.")


__all__ = [
    "ComputedModel",
    "DiscreteEvent",
    "EqxModel",
    "JaxModel",
    "OdeModel",
    "ScipyModel",
    "create_model",
]
