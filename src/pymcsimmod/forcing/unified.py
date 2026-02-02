"""Unified forcing function backend system."""

import importlib.util
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ..config import BackendType


def _check_backend_availability(module_name: str, backend_name: str, install_command: str):
    """
    Check if a backend module is available and raise informative error if not.

    Args:
        module_name: Name of the module to check (e.g., 'jax', 'tensorflow')
        backend_name: Human-readable name of the backend (e.g., 'JAX', 'TensorFlow')
        install_command: Installation command to include in error message

    Raises:
        ImportError: If the module is not available with installation instructions
    """
    if importlib.util.find_spec(module_name) is None:
        raise ImportError(
            f"{backend_name} is required for {backend_name} backend. "
            f"Install with: {install_command}"
        )


class ForcingBackend(ABC):
    """Abstract base class for forcing function backends."""

    @abstractmethod
    def tanh(self, x):
        """Backend-specific tanh implementation."""
        pass  # pragma: no cover

    @abstractmethod
    def floor(self, x):
        """Backend-specific floor implementation."""
        pass  # pragma: no cover

    @abstractmethod
    def sum(self, x, axis=None):
        """Backend-specific sum implementation."""
        pass  # pragma: no cover

    @abstractmethod
    def asarray(self, x):
        """Backend-specific array conversion."""
        pass  # pragma: no cover

    @abstractmethod
    def compile_function(self, func):
        """Backend-specific function compilation (e.g., jit)."""
        pass  # pragma: no cover


class ScipyBackend(ForcingBackend):
    """Scipy/NumPy backend implementation."""

    def tanh(self, x):
        return np.tanh(x)

    def floor(self, x):
        return np.floor(x)

    def sum(self, x, axis=None):
        return np.sum(x, axis=axis)

    def asarray(self, x):
        return np.asarray(x)

    def compile_function(self, func):
        # No compilation for scipy
        return func


class JAXBackend(ForcingBackend):
    """JAX backend implementation."""

    def __init__(self):
        # Check if JAX is available
        _check_backend_availability("jax", "JAX", "pip install jax jaxlib")

        import jax
        import jax.numpy as jnp

        self.jax = jax
        self.jnp = jnp

    def tanh(self, x):
        return self.jnp.tanh(x)

    def floor(self, x):
        return self.jnp.floor(x)

    def sum(self, x, axis=None):
        return self.jnp.sum(x, axis=axis)

    def asarray(self, x):
        return self.jnp.asarray(x)

    def compile_function(self, func):
        return self.jax.jit(func)


class TensorFlowBackend(ForcingBackend):  # pragma: no cover
    """TensorFlow backend implementation (future extension)."""

    def __init__(self):
        # Check if TensorFlow is available
        _check_backend_availability("tensorflow", "TensorFlow", "pip install tensorflow")

        import tensorflow as tf

        self.tf = tf

    def tanh(self, x):
        return self.tf.tanh(x)

    def floor(self, x):
        return self.tf.floor(x)

    def sum(self, x, axis=None):
        return self.tf.reduce_sum(x, axis=axis)

    def asarray(self, x):
        return self.tf.convert_to_tensor(x)

    def compile_function(self, func):
        return self.tf.function(func)


class PyTorchBackend(ForcingBackend):  # pragma: no cover
    """PyTorch backend implementation (future extension)."""

    def __init__(self):
        # Check if PyTorch is available
        _check_backend_availability("torch", "PyTorch", "pip install torch")

        import torch

        self.torch = torch

    def tanh(self, x):
        return self.torch.tanh(x)

    def floor(self, x):
        return self.torch.floor(x)

    def sum(self, x, axis=None):
        if axis is None:
            return self.torch.sum(x)
        return self.torch.sum(x, dim=axis)

    def asarray(self, x):
        return self.torch.tensor(x)

    def compile_function(self, func):
        # PyTorch doesn't have built-in JIT compilation like JAX
        # Could use torch.jit.script here if needed
        return func


class UnifiedForcingFactory:
    """Factory for creating forcing functions with different backends."""

    _backends: ClassVar[dict[BackendType, type[ForcingBackend]]] = {
        BackendType.SCIPY: ScipyBackend,
        BackendType.JAX: JAXBackend,
        BackendType.TENSORFLOW: TensorFlowBackend,
        BackendType.PYTORCH: PyTorchBackend,
    }

    @classmethod
    def register_backend(cls, name: BackendType, backend_class: type[ForcingBackend]):
        """Register a new backend."""
        cls._backends[name] = backend_class

    @classmethod
    def create_onoff(cls, t0: float, t1: float, s: float = 10.0, backend: BackendType = BackendType.SCIPY):
        """Create an on-off forcing function for specified backend."""
        backend_impl = cls._get_backend(backend)

        def onoff_func(t):
            t = backend_impl.asarray(t)
            t0_arr = backend_impl.asarray(t0)
            t1_arr = backend_impl.asarray(t1)
            return (backend_impl.tanh(s * (t - t0_arr)) - backend_impl.tanh(s * (t - t1_arr))) / 2

        return backend_impl.compile_function(onoff_func)

    @classmethod
    def create_perdose(
        cls, t0: float, duration: float, period: float, s: float = 10.0, backend: BackendType = BackendType.SCIPY
    ):
        """Create a periodic dosing forcing function for specified backend."""
        backend_impl = cls._get_backend(backend)
        t0 = float(t0)
        duration = float(duration)
        period = float(period)

        def perdose_func(t):
            t = backend_impl.asarray(t)
            n = backend_impl.floor((t - t0) / period)
            start = t0 + n * period
            stop = start + duration
            return (backend_impl.tanh(s * (t - start)) - backend_impl.tanh(s * (t - stop))) / 2

        return backend_impl.compile_function(perdose_func)

    @classmethod
    def create_ndoses(
        cls, t0_list: list[float], duration: float, s: float = 10.0, backend: BackendType = BackendType.SCIPY
    ):
        """Create a multiple discrete dose forcing function for specified backend."""
        backend_impl = cls._get_backend(backend)
        t0_arr = backend_impl.asarray(t0_list)
        duration = float(duration)

        def ndoses_func(t):
            t = backend_impl.asarray(t)
            t1_arr = t0_arr + duration
            
            # Use natural broadcasting - backends handle scalar/array automatically
            dose_values = (
                backend_impl.tanh(s * (t[..., None] - t0_arr)) - 
                backend_impl.tanh(s * (t[..., None] - t1_arr))
            ) / 2
            return backend_impl.sum(dose_values, axis=-1)

        return backend_impl.compile_function(ndoses_func)

    @classmethod
    def create_zerofunc(cls, backend: BackendType = BackendType.SCIPY):
        """Create a zero forcing function for specified backend."""
        backend_impl = cls._get_backend(backend)

        def zero_func(t):
            return 0.0

        return backend_impl.compile_function(zero_func)

    @classmethod
    def create_constantfunc(cls, val: float, backend: BackendType = BackendType.SCIPY):
        """Create a constant forcing function for specified backend."""
        backend_impl = cls._get_backend(backend)

        def constant_func(t):
            return float(val)

        return backend_impl.compile_function(constant_func)

    @classmethod
    def create_interpolated(
        cls,
        backend: BackendType = BackendType.SCIPY,
        dataframe=None,
        data_dict=None,
        time_col: str = "time",
        value_col: str = "value",
        **kwargs
    ):
        """
        Create an interpolated forcing function using existing class methods.
        
        Args:
            backend: Backend to use ('scipy', 'jax', etc.)
            dataframe: pandas DataFrame with time and value columns (deSolve style)
            data_dict: Dictionary with time and value data
            time_col: Name of time column (for DataFrame)
            value_col: Name of value column (for DataFrame) 
            **kwargs: Additional parameters for InterpolatedForcing
            
        Returns:
            Compiled interpolated forcing function for the specified backend
            
        Raises:
            ValueError: If neither dataframe nor data_dict is provided
            ImportError: If required backend is not available
            
        Example:
            # Using DataFrame (deSolve style)
            func = UnifiedForcingFactory.create_interpolated(
                backend="scipy",
                dataframe=df,
                time_col="time",
                value_col="concentration",
                interpolation_method="cubic"
            )
            
            # Using dict
            func = UnifiedForcingFactory.create_interpolated(
                backend="jax",
                data_dict={"time": [0,1,2], "value": [10,20,30]}
            )
        """
        from ..forcing.interpolated import InterpolatedForcing
        
        if dataframe is not None:
            # Use from_dataframe class method for deSolve compatibility
            forcing = InterpolatedForcing.from_dataframe(
                dataframe, time_col, value_col, **kwargs
            )
        elif data_dict is not None:
            # Use from_dict class method
            forcing = InterpolatedForcing.from_dict(data_dict, **kwargs)
        else:
            raise ValueError(
                "Must provide either 'dataframe' or 'data_dict' parameter for interpolated forcing"
            )
            
        return forcing.create_function(backend)

    @classmethod
    def create_forcing_function(cls, func_name: str, backend: BackendType = BackendType.SCIPY, args=(), **kwargs):
        """
        Create a forcing function by name with the specified backend.
        
        Args:
            func_name: Name of the forcing function ('OnOff', 'PerDose', 'NDoses', 'InterpolatedForcing', etc.)
            backend: Backend to use ('scipy', 'jax', etc.)
            args: Positional arguments for the forcing function (mainly for InterpolatedForcing)
            **kwargs: Parameters for the forcing function
            
        Returns:
            Compiled forcing function for the specified backend
            
        Raises:
            ValueError: If func_name is unknown or required parameters are missing
        """
        if func_name == "OnOff":
            t0 = kwargs.get("t0")
            t1 = kwargs.get("t1")
            s = kwargs.get("s", 10.0)
            if t0 is None or t1 is None:
                raise ValueError(f"OnOff forcing function requires 't0' and 't1' parameters")
            return cls.create_onoff(t0, t1, s, backend)
            
        elif func_name == "PerDose":
            t0 = kwargs.get("t0")
            duration = kwargs.get("duration")
            period = kwargs.get("period")
            s = kwargs.get("s", 10.0)
            if any(param is None for param in [t0, duration, period]):
                raise ValueError(f"PerDose forcing function requires 't0', 'duration', and 'period' parameters")
            return cls.create_perdose(t0, duration, period, s, backend)
            
        elif func_name == "NDoses":
            t0_list = kwargs.get("t0_list")
            duration = kwargs.get("duration")
            s = kwargs.get("s", 10.0)
            if t0_list is None or duration is None:
                raise ValueError(f"NDoses forcing function requires 't0_list' and 'duration' parameters")
            return cls.create_ndoses(t0_list, duration, s, backend)
            
        elif func_name == "ZeroFunc":
            return cls.create_zerofunc(backend)
            
        elif func_name == "ConstFunc":
            value = kwargs.get("value")
            if value is None:
                raise ValueError(f"ConstFunc forcing function requires 'value' parameter")
            return cls.create_constantfunc(value, backend)
            
        elif func_name == "InterpolatedForcing" or func_name == "Interpolate":
            # Use the new create_interpolated method
            # If args are provided (times, values), convert to data_dict
            if args and len(args) >= 2:
                times, values = args[0], args[1]
                kwargs['data_dict'] = {'time': times, 'value': values}
            # Handle times/values in kwargs for interpolation
            elif 'times' in kwargs and 'values' in kwargs:
                times, values = kwargs.pop('times'), kwargs.pop('values')
                kwargs['data_dict'] = {'time': times, 'value': values}
            return cls.create_interpolated(backend=backend, **kwargs)
            
        else:
            raise ValueError(f"Unknown forcing function type: '{func_name}'. Available: OnOff, PerDose, NDoses, ZeroFunc, ConstFunc, InterpolatedForcing, Interpolate")

    @classmethod
    def _get_backend(cls, backend: BackendType) -> ForcingBackend:
        """Get backend implementation instance."""
        if backend not in cls._backends:
            available = [b.value for b in BackendType]
            raise ValueError(f"Unknown backend: {backend}. Available: {available}")
        return cls._backends[backend]()


# Convenience functions that maintain backward compatibility
def create_onoff(t0: float, t1: float, s: float = 10.0, backend: BackendType = BackendType.SCIPY):
    """Create an on-off forcing function."""
    return UnifiedForcingFactory.create_onoff(t0, t1, s, backend)


def create_perdose(
    t0: float, duration: float, period: float, s: float = 10.0, backend: BackendType = BackendType.SCIPY
):
    """Create a periodic dosing forcing function."""
    return UnifiedForcingFactory.create_perdose(t0, duration, period, s, backend)


def create_ndoses(t0_list: list[float], duration: float, s: float = 10.0, backend: BackendType = BackendType.SCIPY):
    """Create a multiple discrete dose forcing function."""
    return UnifiedForcingFactory.create_ndoses(t0_list, duration, s, backend)


def create_zerofunc(backend: BackendType = BackendType.SCIPY):
    """Create a zero forcing function."""
    return UnifiedForcingFactory.create_zerofunc(backend)


def create_constantfunc(val: float, backend: BackendType = BackendType.SCIPY):
    """Create a constant forcing function."""
    return UnifiedForcingFactory.create_constantfunc(val, backend)


__all__ = [
    "ForcingBackend",
    "JAXBackend",
    "PyTorchBackend",
    "ScipyBackend",
    "TensorFlowBackend",
    "UnifiedForcingFactory",
    "create_constantfunc",
    "create_ndoses",
    "create_onoff",
    "create_perdose",
    "create_zerofunc",
]
