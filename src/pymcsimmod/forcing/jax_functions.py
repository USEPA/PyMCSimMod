"""JAX-specific forcing function implementations."""

import jax
import jax.numpy as jnp


def create_onoff(t0: float, t1: float, s: float = 10.0):
    """
    Create an on-off forcing function for JAX backend.
    
    Args:
        t0: Time when function turns on
        t1: Time when function turns off
        s: Smoothing parameter (default: 10.0)
        
    Returns:
        JAX-compiled function that takes time t and returns on/off value
    """
    @jax.jit
    def onoff_func(t):
        """
        JAX-compiled on-off forcing function implementation.
        
        Args:
            t: current time
            
        Returns:
            Value between 0 and 1 representing on/off state.
        """
        t = jnp.asarray(t)
        t0_arr = jnp.asarray(t0)
        t1_arr = jnp.asarray(t1)
        return (jnp.tanh(s * (t - t0_arr)) - jnp.tanh(s * (t - t1_arr))) / 2
    
    return onoff_func


def create_perdose(t0: float, duration: float, period: float, s: float = 10.0):
    """
    Create a periodic dosing forcing function for JAX backend.
    
    Args:
        t0: Time of first dose
        duration: Duration of each dose
        period: Period between doses
        s: Smoothing parameter (default: 10.0)
        
    Returns:
        JAX-compiled function that takes time t and returns dose value
    """
    t0 = float(t0)
    duration = float(duration)
    period = float(period)
    
    @jax.jit
    def perdose_func(t):
        """
        JAX-compiled periodic dosing function implementation.
        
        Args:
            t: current time
            
        Returns:
            Dose value at time t
        """
        t = jnp.asarray(t)
        n = jnp.floor((t - t0) / period)
        start = t0 + n * period
        stop = start + duration
        return (jnp.tanh(s * (t - start)) - jnp.tanh(s * (t - stop))) / 2
    
    return perdose_func


def create_ndoses(t0_list: list[float], duration: float, s: float = 10.0):
    """
    Create a multiple discrete dose forcing function for JAX backend.
    
    Args:
        t0_list: List of dose start times
        duration: Duration of each dose
        s: Smoothing parameter (default: 10.0)
        
    Returns:
        JAX-compiled function that takes time t and returns total dose value
    """
    t0_arr = jnp.array(t0_list)
    duration = float(duration)
    
    @jax.jit
    def ndoses_func(t):
        """
        JAX-compiled multiple doses function implementation.
        
        Args:
            t: current time
            
        Returns:
            Sum of all active dose values at time t
        """
        t = jnp.asarray(t)
        t1_arr = t0_arr + duration
        dose_values = (jnp.tanh(s * (t - t0_arr)) - jnp.tanh(s * (t - t1_arr))) / 2
        return jnp.sum(dose_values, axis=-1)
    
    return ndoses_func


def create_zerofunc():
    """
    Create a zero forcing function for JAX backend.
    
    Returns:
        JAX-compiled function that always returns 0.0
    """
    @jax.jit
    def zero_func(t):
        """
        JAX-compiled zero function implementation.
        
        Args:
            t: current time (unused)
            
        Returns:
            Always returns 0.0
        """
        return 0.0
    
    return zero_func


__all__ = ["create_ndoses", "create_onoff", "create_perdose", "create_zerofunc"]
