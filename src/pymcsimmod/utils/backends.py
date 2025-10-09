"""Backend detection and validation utilities."""

from typing import Protocol, runtime_checkable


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
    backends = {"scipy": False, "jax": False}
    
    # Check scipy availability
    try:
        import scipy.integrate
        backends["scipy"] = True
    except ImportError:
        pass
    
    # Check JAX availability
    try:
        import jax
        import equinox
        import diffrax
        backends["jax"] = True
    except ImportError:
        pass
    
    return backends


def validate_backend(backend: str) -> None:
    """
    Validate that a backend is available and supported.
    
    Args:
        backend: Backend name to validate
        
    Raises:
        ValueError: If backend is not supported
        ImportError: If backend dependencies are not available
    """
    supported_backends = {"scipy", "jax"}
    
    if backend not in supported_backends:
        raise ValueError(f"Unsupported backend '{backend}'. Supported backends: {supported_backends}")
    
    available = detect_available_backends()
    if not available[backend]:
        if backend == "scipy":
            raise ImportError("Scipy backend requires: scipy")
        elif backend == "jax":
            raise ImportError("JAX backend requires: jax, equinox, diffrax")


def get_backend_capabilities(backend: str) -> dict[str, bool]:
    """
    Get the capabilities of a specific backend.
    
    Args:
        backend: Backend name
        
    Returns:
        Dictionary of backend capabilities
    """
    capabilities = {
        "scipy": {
            "discrete_events": True,
            "forcing_functions": True,
            "adaptive_stepping": True,
            "event_detection": True,
            "jit_compilation": False,
            "automatic_differentiation": False,
        },
        "jax": {
            "discrete_events": False,  # Currently unsupported
            "forcing_functions": True,
            "adaptive_stepping": True,
            "event_detection": False,
            "jit_compilation": True,
            "automatic_differentiation": True,
        }
    }
    
    if backend not in capabilities:
        raise ValueError(f"Unknown backend: {backend}")
    
    return capabilities[backend]


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