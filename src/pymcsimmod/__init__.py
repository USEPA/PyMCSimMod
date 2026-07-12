"""Perform ODE simulations using MCSim models."""

from pathlib import Path

from pydantic import validate_call

from .config import BackendType
from .events import (
    DataFrameEventScheduler,
    DiscreteEvent,
    NDoses,
    OnOff,
    PerDose,
    PerDoses,
    create_event_scheduler,
)
from .models import (
    ComputedModel,
    EqxModel,
    JaxModel,
    OdeModel,
    ScipyModel,
)

__version__ = "0.0.1"


@validate_call
def create_model(model_source: str | Path, backend: BackendType = BackendType.SCIPY) -> OdeModel:
    """
    Create a model with the specified backend.

    Args:
        model_source: Path to model file or model string
        backend: Backend type ('scipy', 'jax', 'tensorflow', 'pytorch')

    Returns:
        OdeModel instance with the appropriate backend
    """
    if backend == BackendType.SCIPY:
        return ScipyModel(model_source)
    elif backend == BackendType.JAX:
        return JaxModel(model_source)
    elif backend == BackendType.TENSORFLOW:
        raise NotImplementedError("TensorFlow backend not yet implemented")
    elif backend == BackendType.PYTORCH:
        raise NotImplementedError("PyTorch backend not yet implemented")
    else:
        # This should never happen due to enum validation
        raise ValueError(f"Backend {backend} not implemented")


__all__ = [
    "BackendType",
    "ComputedModel",
    "DataFrameEventScheduler",
    "DiscreteEvent",
    "EqxModel",
    "JaxModel",
    "NDoses",
    "OdeModel",
    "OnOff",
    "PerDose",
    "PerDoses",
    "ScipyModel",
    "create_event_scheduler",
    "create_model",
]
