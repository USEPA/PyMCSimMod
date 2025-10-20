"""Backend detection and validation utilities."""

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, field_validator


class SupportedBackend(str, Enum):
    """Enumeration of supported backends."""

    SCIPY = "scipy"
    JAX = "jax"


class BackendRequest(BaseModel):
    """Pydantic model for backend validation."""

    backend: SupportedBackend

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend_not_none(cls, v):
        """Ensure backend is not None."""
        if v is None:
            raise TypeError("Backend cannot be None")
        return v


@runtime_checkable
class EventHandler(Protocol):
    """Protocol for handling discrete events."""

    def supports_events(self) -> bool:
        """Return True if this backend supports discrete events."""
        ...

    def apply_events(self, times, events) -> any:
        """Apply discrete events during ODE solving."""
        ...


def detect_available_backends() -> dict[str, bool]:
    """
    Detect which backends are available on the system.

    Returns:
        Dictionary mapping backend names to availability
    """
    import importlib.util

    backends = {"scipy": False, "jax": False}

    # Check scipy availability
    if importlib.util.find_spec("scipy.integrate") is not None:
        backends["scipy"] = True

    # Check JAX availability
    jax_modules = ["jax", "equinox", "diffrax"]
    if all(importlib.util.find_spec(module) is not None for module in jax_modules):
        backends["jax"] = True

    return backends


def validate_backend(backend: str) -> None:
    """
    Validate that a backend is available and supported.

    Args:
        backend: Backend name to validate

    Raises:
        ValueError: If backend is not supported
        ImportError: If backend dependencies are not available
        TypeError: If backend is None
    """
    # Use Pydantic validation for type checking and enum validation
    try:
        request = BackendRequest(backend=backend)
        backend_enum = request.backend
    except ValueError as e:
        # Convert Pydantic ValueError to our expected ValueError
        if "Input should be 'scipy' or 'jax'" in str(e):
            raise ValueError(
                f"Unsupported backend '{backend}'. Supported backends: {{'scipy', 'jax'}}"
            )
        raise
    except TypeError:
        # Re-raise TypeError for None values
        raise

    # Check if dependencies are available
    available = detect_available_backends()
    if not available[backend_enum.value]:
        if backend_enum == SupportedBackend.SCIPY:
            raise ImportError("Scipy backend requires: scipy")
        elif backend_enum == SupportedBackend.JAX:
            raise ImportError("JAX backend requires: jax, equinox, diffrax")


def get_backend_capabilities(backend: str) -> dict[str, bool]:
    """
    Get the capabilities of a specific backend.

    Args:
        backend: Backend name

    Returns:
        Dictionary of backend capabilities

    Raises:
        ValueError: If backend is unknown
    """
    # Validate backend using Pydantic
    try:
        request = BackendRequest(backend=backend)
        backend_enum = request.backend
    except (ValueError, TypeError):
        raise ValueError(f"Unknown backend '{backend}'. Supported: {list(SupportedBackend)}")

    capabilities = {
        SupportedBackend.SCIPY: {
            "discrete_events": True,
            "forcing_functions": True,
            "adaptive_stepping": True,
            "event_detection": True,
            "jit_compilation": False,
            "automatic_differentiation": False,
        },
        SupportedBackend.JAX: {
            "discrete_events": False,  # Currently unsupported
            "forcing_functions": True,
            "adaptive_stepping": True,
            "event_detection": False,
            "jit_compilation": True,
            "automatic_differentiation": True,
        },
    }

    return capabilities[backend_enum]


def recommend_backend(
    needs_events: bool = False,
    needs_jit: bool = False,
    needs_autodiff: bool = False,
) -> str:
    """
    Recommend a backend based on requirements.

    Args:
        needs_events: Whether discrete events are required
        needs_jit: Whether JIT compilation is preferred
        needs_autodiff: Whether automatic differentiation is needed

    Returns:
        Recommended backend name
    """
    available = detect_available_backends()

    # If events are required, only scipy supports them currently
    if needs_events:
        if available["scipy"]:
            return "scipy"
        else:
            raise RuntimeError("Discrete events require scipy backend, but scipy is not available")

    # If JAX features are needed and available, recommend JAX
    if (needs_jit or needs_autodiff) and available["jax"]:
        return "jax"

    # Default to scipy if available
    if available["scipy"]:
        return "scipy"

    # If only JAX is available
    if available["jax"]:
        return "jax"

    raise RuntimeError("No supported backends are available")


__all__ = [
    "EventHandler",
    "detect_available_backends",
    "get_backend_capabilities",
    "recommend_backend",
    "validate_backend",
]
