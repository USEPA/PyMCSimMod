"""Configuration and enums for PyMCSimMod package."""

from enum import StrEnum


class BackendType(StrEnum):
    """
    Supported computational backends for PyMCSimMod.

    This enum defines the available backends for ODE solving and forcing functions.
    String values allow seamless conversion from user-provided strings.
    """

    SCIPY = "scipy"
    JAX = "jax"
    TENSORFLOW = "tensorflow"  # Future implementation
    PYTORCH = "pytorch"  # Future implementation


__all__ = ["BackendType"]
