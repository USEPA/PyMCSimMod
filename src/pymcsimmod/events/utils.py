"""Event handling utilities inspired by deSolve's approach."""

import warnings

import numpy as np


def nearest_event_time(times: np.ndarray, event_times: np.ndarray) -> np.ndarray:
    """
    Find the nearest event time for each output time.

    Equivalent to deSolve's nearestEvent function.

    Args:
        times: Array of output times
        event_times: Array of event times

    Returns:
        Array of nearest event times for each output time
    """
    event_times = np.unique(event_times)  # Remove duplicates
    times = np.sort(times)
    event_times = np.sort(event_times)

    # Find index where each time would be inserted
    indices = np.searchsorted(event_times, times)

    # Handle boundary conditions
    lower_indices = np.maximum(indices - 1, 0)
    upper_indices = np.minimum(indices, len(event_times) - 1)

    lower = event_times[lower_indices]
    upper = event_times[upper_indices]

    # Choose the nearest event time
    nearest = np.where(np.abs(times - lower) <= np.abs(times - upper), lower, upper)

    return nearest


def clean_event_times(
    times: np.ndarray, event_times: np.ndarray, eps: float | None = None
) -> np.ndarray:
    """
    Remove output times that are numerically too close to event times.

    Equivalent to deSolve's cleanEventTimes function.

    Args:
        times: Array of output times
        event_times: Array of event times
        eps: Tolerance for considering times "too close" (default: 10 * machine epsilon)

    Returns:
        Cleaned array of output times with near-duplicates removed
    """
    if eps is None:
        eps = np.finfo(float).eps * 10

    times = np.sort(times)
    nearest = nearest_event_time(times, event_times)

    # Calculate relative difference, using the larger of the two numbers as denominator
    div = np.maximum(np.abs(times), np.abs(nearest))
    div = np.where(div == 0, 1, div)  # Handle zero case

    rel_diff = np.abs(times - nearest) / div
    too_close = rel_diff < eps

    return times[~too_close]


def check_events(
    events: list, times: np.ndarray, state_names: list[str]
) -> tuple[list, np.ndarray]:
    """
    Validate and process events, following deSolve's checkevents logic.

    Args:
        events: List of DiscreteEvent objects
        times: Array of output times
        state_names: List of state variable names

    Returns:
        Tuple of (validated_events, modified_times)
    """
    if not events:
        return [], times

    # Extract event times within simulation range
    event_times = []
    valid_events = []

    time_range = (times[0], times[-1])

    for event in events:
        if time_range[0] <= event.time <= time_range[1]:
            # Validate state variable
            if event.state_var not in state_names:
                raise ValueError(f"Unknown state variable in event: {event.state_var}")

            event_times.append(event.time)
            valid_events.append(event)

    if not event_times:
        return valid_events, times

    event_times = np.array(event_times)

    # Check if all event times are in output times
    missing_events = []
    for et in event_times:
        if not np.any(np.abs(times - et) < 1e-12):
            missing_events.append(et)

    if missing_events:
        # Provide specific guidance based on time range
        time_start, time_end = float(np.min(times)), float(np.max(times))
        step_size = (time_end - time_start) / (len(times) - 1)

        warnings.warn(
            f"Event times {missing_events} not found in time grid - automatically adding them to ensure accurate event handling. "
            f"To avoid this message, consider using: np.arange({time_start}, {time_end + step_size:.3f}, {step_size:.3f}) "
            f"or explicitly include event times in your time array.",
            stacklevel=2,
        )

        # Clean existing times that are too close to events
        unique_times = clean_event_times(times, event_times)

        if len(unique_times) < len(times):
            removed_count = len(times) - len(unique_times)
            warnings.warn(
                f"Removed {removed_count} time points that were within numerical precision of event times "
                f"to avoid numerical issues. This is normal behavior following deSolve's event handling.",
                stacklevel=2,
            )

        # Combine and sort times
        modified_times = np.sort(np.concatenate([unique_times, event_times]))
    else:
        modified_times = times

    # Sort events by time
    time_order = np.argsort([e.time for e in valid_events])
    valid_events = [valid_events[i] for i in time_order]

    return valid_events, modified_times


def apply_events_at_time(
    time: float, state_dict: dict, events: list, tolerance: float = 1e-12
) -> dict:
    """
    Apply all events that occur at a specific time.

    Args:
        time: Current time
        state_dict: Current state as dictionary
        events: List of all events
        tolerance: Tolerance for time matching

    Returns:
        Updated state dictionary
    """
    for event in events:
        if abs(event.time - time) < tolerance:
            state_dict = event.apply(state_dict, list(state_dict.keys()))

    return state_dict


def extract_event_times(events: list, t_start: float, t_end: float) -> list[float]:
    """
    Extract event times within the specified time range.

    Args:
        events: List of DiscreteEvent objects
        t_start: Start time
        t_end: End time

    Returns:
        List of event times within [t_start, t_end]
    """
    return [event.time for event in events if t_start <= event.time <= t_end]


__all__ = [
    "apply_events_at_time",
    "check_events",
    "clean_event_times",
    "extract_event_times",
    "nearest_event_time",
]
