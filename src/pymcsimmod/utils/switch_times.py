"""Switch time utilities for ODE solving with discontinuities."""

from typing import Any


def extract_forcing_switch_times(
    forcing_functions: dict[str, Any], t_start: float, t_end: float
) -> list[float]:
    """
    Extract switch times from forcing function specifications.

    Args:
        forcing_functions: Dictionary of forcing function specifications
        t_start: Start time for extraction
        t_end: End time for extraction

    Returns:
        Sorted list of switch times within [t_start, t_end]
    """
    switch_times = set()

    for ff in forcing_functions.values():
        if isinstance(ff, dict) and "function" in ff:
            func = ff["function"]
            kwargs = ff.get("kwargs", {})

            if func == "PerDose":
                switch_times.update(_extract_perdose_times(kwargs, t_start, t_end))
            elif func == "NDoses":
                switch_times.update(_extract_ndoses_times(kwargs, t_start, t_end))
            elif func == "OnOff":
                switch_times.update(_extract_onoff_times(kwargs, t_start, t_end))

    return sorted(switch_times)


def _extract_perdose_times(kwargs: dict, t_start: float, t_end: float) -> set[float]:
    """Extract switch times for PerDose forcing function."""
    switch_times = set()
    t0 = kwargs.get("t0", 0.0)
    duration = kwargs.get("duration", 1.0)
    period = kwargs.get("period", 8.0)

    n = 0
    while True:
        on_time = t0 + n * period
        off_time = on_time + duration

        if on_time > t_end:
            break

        if on_time >= t_start:
            switch_times.add(on_time)
        if off_time >= t_start and off_time <= t_end:
            switch_times.add(off_time)

        n += 1

    return switch_times


def _extract_ndoses_times(kwargs: dict, t_start: float, t_end: float) -> set[float]:
    """Extract switch times for NDoses forcing function."""
    switch_times = set()
    t0_list = kwargs.get("t0_list", [])
    duration = kwargs.get("duration", 1.0)

    for t0 in t0_list:
        on_time = t0
        off_time = t0 + duration

        if on_time >= t_start and on_time <= t_end:
            switch_times.add(on_time)
        if off_time >= t_start and off_time <= t_end:
            switch_times.add(off_time)

    return switch_times


def _extract_onoff_times(kwargs: dict, t_start: float, t_end: float) -> set[float]:
    """Extract switch times for OnOff forcing function."""
    switch_times = set()
    t0 = kwargs.get("t0", 0.0)
    t1 = kwargs.get("t1", 1.0)

    if t0 >= t_start and t0 <= t_end:
        switch_times.add(t0)
    if t1 >= t_start and t1 <= t_end:
        switch_times.add(t1)

    return switch_times


def extract_event_times(events: list[Any], t_start: float, t_end: float) -> list[float]:
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


def combine_switch_times(
    forcing_times: list[float],
    event_times: list[float],
    additional_times: list[float] | None = None,
) -> list[float]:
    """
    Combine and sort all switch times from different sources.

    Args:
        forcing_times: Switch times from forcing functions
        event_times: Switch times from discrete events
        additional_times: Additional switch times (optional)

    Returns:
        Sorted list of unique switch times
    """
    all_times = set(forcing_times)
    all_times.update(event_times)

    if additional_times:
        all_times.update(additional_times)

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
    "extract_event_times",
    "extract_forcing_switch_times",
    "validate_switch_times",
]
