"""Backend detection and validation utilities."""

import importlib.util
from typing import Protocol, runtime_checkable

from ..config import BackendType


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
        Dictionary mapping backend names to availability.
    """
    backends = {
        "scipy": False,
        "jax": False,
        "tensorflow": False,
        "pytorch": False,
    }

    # Check scipy availability
    if importlib.util.find_spec("scipy.integrate") is not None:
        backends["scipy"] = True

    # Check JAX availability (requires jax, equinox, diffrax)
    jax_modules = ["jax", "equinox", "diffrax"]
    if all(importlib.util.find_spec(module) is not None for module in jax_modules):
        backends["jax"] = True

    # Check TensorFlow availability
    if importlib.util.find_spec("tensorflow") is not None:
        backends["tensorflow"] = True

    # Check PyTorch availability
    if importlib.util.find_spec("torch") is not None:
        backends["pytorch"] = True

    return backends


def validate_backend(backend: str) -> None:
    """
    Validate that a backend is supported and its dependencies are available.

    Args:
        backend: Backend name to validate.

    Raises:
        ValueError: If backend name is not recognised.
        ImportError: If backend dependencies are not installed.
        TypeError: If backend is None.
    """
    if backend is None:
        raise TypeError("Backend cannot be None")

    backend_lower = backend.lower()
    valid = {bt.value for bt in BackendType}
    if backend_lower not in valid:
        raise ValueError(f"Unsupported backend '{backend}'. Supported backends: {valid}")

    available = detect_available_backends()
    backend_key = backend_lower
    if backend_key in available and not available[backend_key]:
        install_hints = {
            "scipy": "pip install scipy",
            "jax": "pip install jax equinox diffrax",
            "tensorflow": "pip install tensorflow",
            "pytorch": "pip install torch",
        }
        hint = install_hints.get(backend_key, f"pip install {backend_key}")
        raise ImportError(
            f"{backend} backend requires additional dependencies. Install with: {hint}"
        )


def get_backend_capabilities(backend: str) -> dict[str, bool]:
    """
    Get the capabilities of a specific backend.

    Args:
        backend: Backend name (e.g., 'scipy', 'jax').

    Returns:
        Dictionary of backend capabilities.

    Raises:
        ValueError: If backend is unknown.
    """
    capabilities: dict[BackendType, dict[str, bool]] = {
        BackendType.SCIPY: {
            "discrete_events": True,
            "forcing_functions": True,
            "adaptive_stepping": True,
            "event_detection": True,
            "jit_compilation": False,
            "automatic_differentiation": False,
        },
        BackendType.JAX: {
            # JAX now supports discrete events via jax.lax.scan piece-wise integration
            "discrete_events": True,
            "forcing_functions": True,
            "adaptive_stepping": True,
            "event_detection": True,
            "jit_compilation": True,
            "automatic_differentiation": True,
        },
        # --- Future backend templates (not yet implemented) ---
        BackendType.TENSORFLOW: {
            "discrete_events": False,  # Not yet implemented
            "forcing_functions": False,  # Not yet implemented
            "adaptive_stepping": False,
            "event_detection": False,
            "jit_compilation": True,  # tf.function
            "automatic_differentiation": True,  # tf.GradientTape
        },
        BackendType.PYTORCH: {
            "discrete_events": False,
            "forcing_functions": False,
            "adaptive_stepping": False,
            "event_detection": False,
            "jit_compilation": False,  # torch.jit.script (limited)
            "automatic_differentiation": True,  # autograd
        },
    }

    try:
        bt = BackendType(backend.lower())
    except ValueError:
        valid = [bt.value for bt in BackendType]
        raise ValueError(f"Unknown backend '{backend}'. Supported: {valid}")

    return capabilities[bt]


def recommend_backend(
    needs_events: bool = False,
    needs_jit: bool = False,
    needs_autodiff: bool = False,
) -> str:
    """
    Recommend a backend based on requirements.

    Args:
        needs_events: Whether discrete events are required.
        needs_jit: Whether JIT compilation is preferred.
        needs_autodiff: Whether automatic differentiation is needed.

    Returns:
        Recommended backend name.
    """
    available = detect_available_backends()

    # If autodiff or JIT is needed, prefer JAX (it supports everything including events now)
    if (needs_autodiff or needs_jit) and available["jax"]:
        return "jax"

    # For events without autodiff/jit, scipy is a solid choice
    if needs_events and available["scipy"]:
        return "scipy"

    # Default to scipy if available
    if available["scipy"]:
        return "scipy"

    # Fall back to JAX
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
