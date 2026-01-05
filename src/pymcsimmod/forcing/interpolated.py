"""Interpolated forcing functions for time-varying inputs from data."""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .base import BackendAwareForcing


class InterpolatedForcing(BackendAwareForcing):
    """
    Forcing function that interpolates between discrete time points from a DataFrame.

    Similar to deSolve's forcing function approach where users provide a dataframe
    with time and value columns, and the function interpolates between points.

    Examples:
        # From DataFrame
        df = pd.DataFrame({'time': [0, 1, 2, 5], 'bodyweight': [20, 22, 24, 30]})
        bw_forcing = InterpolatedForcing.from_dataframe(df, 'time', 'bodyweight')

        # From arrays
        times = [0, 1, 2, 5]
        values = [20, 22, 24, 30]
        bw_forcing = InterpolatedForcing(times, values)
    """

    def __init__(
        self,
        times: list | np.ndarray,
        values: list | np.ndarray,
        interpolation_method: str = "linear",
        bounds_error: bool = False,
        fill_value: float | str = "extrapolate",
    ):
        """
        Initialize interpolated forcing function.

        Args:
            times: Array of time points
            values: Array of corresponding values
            interpolation_method: Interpolation method ('linear', 'cubic', 'nearest', etc.)
            bounds_error: If True, raise error when extrapolating beyond data range
            fill_value: Value to use for extrapolation ('extrapolate' or a number)
        """
        super().__init__()
        self.times = np.asarray(times)
        self.values = np.asarray(values)
        self.interpolation_method = interpolation_method
        self.bounds_error = bounds_error
        self.fill_value = fill_value

        # Validate inputs
        if len(self.times) != len(self.values):
            raise ValueError("times and values must have the same length")

        if len(self.times) < 2:
            raise ValueError("At least 2 time points required for interpolation")

        # Sort by time to ensure proper interpolation
        sorted_indices = np.argsort(self.times)
        self.times = self.times[sorted_indices]
        self.values = self.values[sorted_indices]

        # Check for duplicate times
        if len(np.unique(self.times)) != len(self.times):
            raise ValueError("Duplicate time points are not allowed")

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, time_col: str, value_col: str, **kwargs
    ) -> "InterpolatedForcing":
        """
        Create interpolated forcing function from a pandas DataFrame.

        Args:
            df: DataFrame containing time and value data
            time_col: Name of the time column
            value_col: Name of the value column
            **kwargs: Additional arguments passed to InterpolatedForcing constructor

        Returns:
            InterpolatedForcing instance

        Example:
            df = pd.DataFrame({
                'time': [0, 1, 2, 5, 10],
                'bodyweight': [20, 22, 24, 30, 35]
            })
            bw_forcing = InterpolatedForcing.from_dataframe(df, 'time', 'bodyweight')
        """
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in DataFrame")
        if value_col not in df.columns:
            raise ValueError(f"Value column '{value_col}' not found in DataFrame")

        # Sort by time and remove any NaN values
        df_clean = df[[time_col, value_col]].dropna().sort_values(time_col)

        if len(df_clean) < 2:
            raise ValueError("At least 2 valid data points required after cleaning")

        return cls(times=df_clean[time_col].values, values=df_clean[value_col].values, **kwargs)

    @classmethod
    def from_dict(
        cls, data: dict, time_key: str = "time", value_key: str = "value", **kwargs
    ) -> "InterpolatedForcing":
        """
        Create interpolated forcing function from a dictionary.

        Args:
            data: Dictionary with time and value arrays
            time_key: Key for time data (default: 'time')
            value_key: Key for value data (default: 'value')
            **kwargs: Additional arguments passed to InterpolatedForcing constructor

        Returns:
            InterpolatedForcing instance
        """
        if time_key not in data:
            raise ValueError(f"Time key '{time_key}' not found in data")
        if value_key not in data:
            raise ValueError(f"Value key '{value_key}' not found in data")

        return cls(times=data[time_key], values=data[value_key], **kwargs)

    def _create_backend_function(self, backend: str):
        """Create the interpolation function for the specified backend."""
        if backend == "scipy":
            return self._create_scipy_function()
        elif backend == "jax":
            return self._create_jax_function()
        else:
            # For other backends, we could implement interpolation like this:
            # elif backend == "tensorflow":
            #     return self._create_tensorflow_function()
            # elif backend == "pytorch":
            #     return self._create_pytorch_function()
            #
            # For now, we only support scipy and jax for interpolation
            raise ValueError(
                f"Interpolation not yet supported for backend: {backend}. Supported: scipy, jax"
            )

    def _create_scipy_function(self):
        """Create scipy-compatible interpolation function."""
        from .unified import ScipyBackend

        backend = ScipyBackend()

        # Create scipy interpolator
        interpolator = interp1d(
            self.times,
            self.values,
            kind=self.interpolation_method,
            bounds_error=self.bounds_error,
            fill_value=self.fill_value,
        )

        def interpolation_func(t):
            """
            Interpolation function for scipy backend.

            Args:
                t: Time point(s) to evaluate

            Returns:
                Interpolated value(s)
            """
            # Convert input using backend
            t = backend.asarray(t)

            # Handle scalar input
            if np.isscalar(t):
                return float(interpolator(t))

            # Handle array input
            return interpolator(t)

        return backend.compile_function(interpolation_func)

    def _create_jax_function(self):
        """Create JAX-compatible interpolation function."""
        from .unified import JAXBackend

        backend = JAXBackend()

        # Convert to JAX arrays
        times_jax = backend.asarray(self.times)
        values_jax = backend.asarray(self.values)

        def jax_interpolation_func(t):
            """
            JAX-compatible interpolation function using linear interpolation.

            Args:
                t: Time point to evaluate

            Returns:
                Interpolated value
            """
            # Use JAX's interp function for linear interpolation
            return backend.jnp.interp(t, times_jax, values_jax)

        return backend.compile_function(jax_interpolation_func)

    def get_switch_times(self, t_start: float, t_end: float) -> list[float]:
        """
        Get all data time points within the simulation range.

        Args:
            t_start: Start time of simulation
            t_end: End time of simulation

        Returns:
            List of time points where data is available
        """
        # Return all time points that fall within the simulation range
        mask = (self.times >= t_start) & (self.times <= t_end)
        return self.times[mask].tolist()

    def get_data_range(self) -> tuple[float, float]:
        """
        Get the time range of the available data.

        Returns:
            Tuple of (min_time, max_time)
        """
        return float(self.times.min()), float(self.times.max())

    def get_value_range(self) -> tuple[float, float]:
        """
        Get the value range of the available data.

        Returns:
            Tuple of (min_value, max_value)
        """
        return float(self.values.min()), float(self.values.max())

    def plot_data(
        self, ax=None, show_points=True, show_interpolation=True, n_interp_points=100, **kwargs
    ):
        """
        Plot the data and interpolation function.

        Args:
            ax: Matplotlib axes to plot on (creates new if None)
            show_points: Whether to show the original data points
            show_interpolation: Whether to show the interpolated curve
            n_interp_points: Number of points for interpolation curve
            **kwargs: Additional keyword arguments for matplotlib

        Returns:
            Matplotlib axes object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        # Plot original data points
        if show_points:
            ax.scatter(
                self.times, self.values, color="red", s=50, zorder=3, label="Data points", **kwargs
            )

        # Plot interpolation curve
        if show_interpolation:
            t_interp = np.linspace(self.times.min(), self.times.max(), n_interp_points)
            interpolator = self._create_scipy_function()
            v_interp = [interpolator(t) for t in t_interp]
            ax.plot(
                t_interp,
                v_interp,
                "b-",
                linewidth=2,
                label=f"{self.interpolation_method} interpolation",
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title("Interpolated Forcing Function")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def __repr__(self):
        """String representation of the interpolated forcing function."""
        return (
            f"InterpolatedForcing(n_points={len(self.times)}, "
            f"time_range=({self.times.min():.3f}, {self.times.max():.3f}), "
            f"method='{self.interpolation_method}')"
        )


# Convenience function for creating interpolated forcing from common data formats
def create_interpolated_forcing(
    data: pd.DataFrame | dict | tuple,
    time_col: str | None = None,
    value_col: str | None = None,
    **kwargs,
) -> InterpolatedForcing:
    """
    Convenience function to create interpolated forcing from various data formats.

    Args:
        data: Data in various formats:
            - pandas DataFrame with time and value columns
            - dict with 'time' and 'value' keys (or specified by time_col/value_col)
            - tuple of (times, values) arrays
        time_col: Name of time column (for DataFrame) or time key (for dict)
        value_col: Name of value column (for DataFrame) or value key (for dict)
        **kwargs: Additional arguments for InterpolatedForcing

    Returns:
        InterpolatedForcing instance

    Examples:
        # From DataFrame
        df = pd.DataFrame({'t': [0, 1, 2], 'bw': [20, 22, 24]})
        forcing = create_interpolated_forcing(df, 't', 'bw')

        # From dict
        data = {'time': [0, 1, 2], 'value': [20, 22, 24]}
        forcing = create_interpolated_forcing(data)

        # From tuple
        forcing = create_interpolated_forcing(([0, 1, 2], [20, 22, 24]))
    """
    if isinstance(data, pd.DataFrame):
        if time_col is None or value_col is None:
            raise ValueError("time_col and value_col must be specified for DataFrame input")
        return InterpolatedForcing.from_dataframe(data, time_col, value_col, **kwargs)

    elif isinstance(data, dict):
        time_key = time_col if time_col is not None else "time"
        value_key = value_col if value_col is not None else "value"
        return InterpolatedForcing.from_dict(data, time_key, value_key, **kwargs)

    elif isinstance(data, tuple | list) and len(data) == 2:
        times, values = data
        return InterpolatedForcing(times, values, **kwargs)

    else:
        raise ValueError("Unsupported data format. Use DataFrame, dict, or (times, values) tuple.")
