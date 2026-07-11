"""Utility functions."""

from .backends import (
    EventHandler,
    detect_available_backends,
    get_backend_capabilities,
    recommend_backend,
    validate_backend,
)
from .context import build_evaluation_context, merge_contexts, validate_context
from .switch_times import (
    combine_switch_times,
    validate_switch_times,
)

__all__ = [
    "EventHandler",
    "build_evaluation_context",
    "combine_switch_times",
    "detect_available_backends",
    "get_backend_capabilities",
    "merge_contexts",
    "recommend_backend",
    "validate_backend",
    "validate_context",
    "validate_switch_times",
]
