"""Base event schedulers and factory implementation."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..models.events import DiscreteEvent


class BaseEventScheduler(ABC):
    """Base class for event schedulers."""

    def __init__(self, method: str = "add"):
        self.method = method

    @abstractmethod
    def get_events(self, state_var: str, t_start: float, t_end: float) -> list[DiscreteEvent]:
        """Generate a list of DiscreteEvent objects for the given state variable."""
        pass  # pragma: no cover


class NDoses(BaseEventScheduler):
    """Multiple discrete doses event scheduler."""

    def __init__(
        self,
        t0_list: list[float] | Sequence[float],
        value: float | list[float] | Sequence[float],
        method: str = "add",
    ):
        super().__init__(method=method)
        self.t0_list = sorted(list(t0_list))
        self.value = value

    def get_events(self, state_var: str, t_start: float, t_end: float) -> list[DiscreteEvent]:
        events = []
        for i, t0 in enumerate(self.t0_list):
            if t_start <= t0 <= t_end:
                val = (
                    self.value[i]
                    if isinstance(self.value, (list, tuple, np.ndarray))
                    else self.value
                )
                events.append(
                    DiscreteEvent(
                        time=float(t0),
                        state_var=state_var,
                        value=float(val),
                        method=self.method,
                    )
                )
        return events


class PerDoses(BaseEventScheduler):
    """Periodic dosing event scheduler."""

    def __init__(
        self,
        t0: float,
        period: float,
        value: float,
        method: str = "add",
        n: int | None = None,
        t_end: float | None = None,
    ):
        super().__init__(method=method)
        self.t0 = t0
        self.period = period
        self.value = value
        self.n = n
        self.t_end_limit = t_end

    def get_events(self, state_var: str, t_start: float, t_end: float) -> list[DiscreteEvent]:
        events = []
        n = 0
        limit_t = t_end
        if self.t_end_limit is not None:
            limit_t = min(limit_t, self.t_end_limit)

        while True:
            t = self.t0 + n * self.period
            if t > limit_t:
                break
            if self.n is not None and n >= self.n:
                break
            if t >= t_start:
                events.append(
                    DiscreteEvent(
                        time=float(t),
                        state_var=state_var,
                        value=float(self.value),
                        method=self.method,
                    )
                )
            n += 1
        return events


class OnOff(BaseEventScheduler):
    """On-Off event scheduler (triggers at start and end of interval)."""

    def __init__(
        self,
        t0: float,
        t1: float,
        value: float,
        method: str = "add",
        value_off: float | None = None,
    ):
        super().__init__(method=method)
        self.t0 = t0
        self.t1 = t1
        self.value = value
        self.value_off = value_off

    def get_events(self, state_var: str, t_start: float, t_end: float) -> list[DiscreteEvent]:
        events = []
        if t_start <= self.t0 <= t_end:
            events.append(
                DiscreteEvent(
                    time=float(self.t0),
                    state_var=state_var,
                    value=float(self.value),
                    method=self.method,
                )
            )
        if t_start <= self.t1 <= t_end:
            if self.value_off is not None:
                off_val = self.value_off
                off_method = self.method
            else:
                if self.method == "add":
                    off_val = -self.value
                    off_method = "add"
                elif self.method == "replace":
                    off_val = 0.0
                    off_method = "replace"
                elif self.method == "multiply":
                    off_val = 1.0 / self.value if self.value != 0.0 else 0.0
                    off_method = "multiply"
                else:
                    off_val = 0.0
                    off_method = self.method
            events.append(
                DiscreteEvent(
                    time=float(self.t1),
                    state_var=state_var,
                    value=float(off_val),
                    method=off_method,
                )
            )
        return events


class DataFrameEventScheduler(BaseEventScheduler):
    """Event scheduler using a pandas DataFrame."""

    def __init__(
        self,
        df: pd.DataFrame,
        state_var: str | None = None,
        time_col: str = "time",
        value_col: str = "value",
        method_col: str = "method",
        method: str = "add",
    ):
        super().__init__(method=method)
        self.df = df
        self.state_var = state_var
        self.time_col = time_col
        self.value_col = value_col
        self.method_col = method_col

    def get_events(self, state_var: str, t_start: float, t_end: float) -> list[DiscreteEvent]:
        events = []
        df_filtered = self.df

        if self.time_col in df_filtered.columns:
            df_filtered = df_filtered[
                (df_filtered[self.time_col] >= t_start) & (df_filtered[self.time_col] <= t_end)
            ]

        if "state_var" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["state_var"] == state_var]
        elif self.state_var is not None:
            if self.state_var != state_var:
                return []

        for _, row in df_filtered.iterrows():
            t = float(row[self.time_col])
            val = float(row[self.value_col])
            meth = (
                str(row[self.method_col])
                if self.method_col in df_filtered.columns
                else self.method
            )
            s_var = str(row["state_var"]) if "state_var" in df_filtered.columns else state_var
            events.append(
                DiscreteEvent(
                    time=t,
                    state_var=s_var,
                    value=val,
                    method=meth,
                )
            )
        return events


def create_event_scheduler(scheduler_type: str, *args, **kwargs) -> BaseEventScheduler:
    """Factory function to create event schedulers."""
    t = scheduler_type.lower()
    if t == "ndoses" or t == "multidose":
        return NDoses(*args, **kwargs)
    elif t == "perdoses" or t == "periodic":
        return PerDoses(*args, **kwargs)
    elif t == "onoff":
        return OnOff(*args, **kwargs)
    elif t == "dataframe" or t == "df":
        return DataFrameEventScheduler(*args, **kwargs)
    else:
        available = ["NDoses", "PerDoses", "OnOff", "DataFrame"]
        raise ValueError(f"Unknown event scheduler type: {scheduler_type}. Available: {available}")
