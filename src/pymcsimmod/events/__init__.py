"""Events submodule.

Provides:
  - DiscreteEvent: Atomic discrete state-change event
  - BaseEventScheduler: Abstract base for event schedulers
  - NDoses: Multi-dose discrete event scheduler
  - PerDose: Periodic dose event scheduler (PerDoses is an alias)
  - OnOff: On/off interval event scheduler
  - DataFrameEventScheduler: DataFrame-driven event scheduler
  - create_event_scheduler: Factory function
  - Event utilities: check_events, apply_events_at_time, etc.
"""

from .base import (
    BaseEventScheduler,
    DataFrameEventScheduler,
    DiscreteEvent,
    NDoses,
    OnOff,
    PerDose,
    PerDoses,
    create_event_scheduler,
)
from .utils import (
    apply_events_at_time,
    check_events,
    clean_event_times,
    extract_event_times,
    nearest_event_time,
)

__all__ = [
    "BaseEventScheduler",
    "DataFrameEventScheduler",
    "DiscreteEvent",
    "NDoses",
    "OnOff",
    "PerDose",
    "PerDoses",
    "apply_events_at_time",
    "check_events",
    "clean_event_times",
    "create_event_scheduler",
    "extract_event_times",
    "nearest_event_time",
]
