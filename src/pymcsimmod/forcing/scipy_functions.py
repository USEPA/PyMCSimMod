"""Scipy-specific forcing function implementations."""

import numpy as np


def create_onoff(t0: float, t1: float, s: float = 10.0):
    """
    Create an on-off forcing function for scipy backend.

    Args:
        t0: Time when function turns on
        t1: Time when function turns off
        s: Smoothing parameter for transitions (default: 10.0)

    Returns:
        Function that takes time t and returns on/off value
    """

    def onoff_func(t):
        """
        On-off forcing function implementation.

        Args:
            t: current time

        Returns:
            Value between 0 and 1 representing on/off state.
        """
        y = (np.tanh(s * (t - t0)) - np.tanh(s * (t - t1))) / 2
        return y

    return onoff_func


def create_perdose(t0: float, duration: float, period: float, s: float = 10.0):
    """
    Create a periodic dosing forcing function for scipy backend.

    Args:
        t0: Time of first dose
        duration: Duration of each dose
        period: Period between doses
        s: Smoothing parameter for transitions (default: 10.0)

    Returns:
        Function that takes time t and returns dose value
    """

    def perdose_func(t):
        """
        Periodic dosing function implementation.

        Args:
            t: current time

        Returns:
            Dose value at time t
        """
        if t < t0:
            return 0.0
        n = int((t - t0) // period)
        start = t0 + n * period
        stop = start + duration
        # Use the onoff function for each dose period
        y = (np.tanh(s * (t - start)) - np.tanh(s * (t - stop))) / 2
        return y

    return perdose_func


def create_ndoses(t0_list: list[float], duration: float, s: float = 10.0):
    """
    Create a multiple discrete dose forcing function for scipy backend.

    Args:
        t0_list: List of dose start times
        duration: Duration of each dose
        s: Smoothing parameter for transitions (default: 10.0)

    Returns:
        Function that takes time t and returns total dose value
    """

    def ndoses_func(t):
        """
        Multiple doses function implementation.

        Args:
            t: current time

        Returns:
            Sum of all active dose values at time t
        """
        total = 0.0
        for t0 in t0_list:
            t1 = t0 + duration
            dose_value = (np.tanh(s * (t - t0)) - np.tanh(s * (t - t1))) / 2
            total += dose_value
        return total

    return ndoses_func


def create_zerofunc():
    """
    Create a zero forcing function for scipy backend.

    Returns:
        Function that always returns 0.0
    """

    def zero_func(t):
        """
        Zero function implementation.

        Args:
            t: current time (unused)

        Returns:
            Always returns 0.0
        """
        return 0.0

    return zero_func


def create_constantfunc(val: float):
    """
    Create a forcing function with a constant value for scipy backend.

    Returns:
        Function that always returns val
    """

    def constant_func(t):
        """
        Zero function implementation.

        Args:
            t: current time (unused)

        Returns:
            Always returns val
        """
        return float(val)

    return constant_func


__all__ = [
    "create_constantfunc",
    "create_ndoses",
    "create_onoff",
    "create_perdose",
    "create_zerofunc",
]
