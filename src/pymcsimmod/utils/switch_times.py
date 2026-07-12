"""Switch time utilities for ODE solving with discontinuities.

Note: Forcing-function-specific switch time extraction has moved to
``pymcsimmod.forcing.unified.extract_forcing_switch_times``.
Event-specific switch time extraction has moved to
``pymcsimmod.events.utils.extract_event_times``.

This module retains only the generic combining/validation utilities.
"""


def combine_switch_times(
    *time_sets: list[float],
) -> list[float]:
    """
    Combine and sort switch times from multiple sources.

    Args:
        *time_sets: One or more sequences of switch times.

    Returns:
        Sorted list of unique switch times.
    """
    all_times: set[float] = set()
    for times in time_sets:
        all_times.update(times)
    return sorted(all_times)


def validate_switch_times(switch_times: list[float], t_start: float, t_end: float) -> list[float]:
    """
    Validate and filter switch times to ensure they're within bounds.

    Args:
        switch_times: List of switch times to validate
        t_start: Start time bound
        t_end: End time bound

    Returns:
        Filtered list of switch times within bounds
    """
    return [t for t in switch_times if t_start <= t <= t_end]


__all__ = [
    "combine_switch_times",
    "validate_switch_times",
]
