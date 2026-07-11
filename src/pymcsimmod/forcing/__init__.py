"""Forcing functions.

Continuous forcing types (OnOff, PerDose, NDoses) are accessible via
``model.assign_forcing_function()`` or directly through ``UnifiedForcingFactory``.
For data-driven interpolated forcing, use ``InterpolatedForcing``.
"""

from .base import (
    BackendAwareForcing,
    ForcingFunction,
    create_forcing_function,
)
from .interpolated import InterpolatedForcing, create_interpolated_forcing
from .unified import (
    ForcingBackend,
    JAXBackend,
    PyTorchBackend,
    ScipyBackend,
    TensorFlowBackend,
    UnifiedForcingFactory,
    create_constantfunc,
    create_ndoses,
    create_onoff,
    create_perdose,
    create_zerofunc,
    extract_forcing_switch_times,
)

__all__ = [
    "BackendAwareForcing",
    "ForcingBackend",
    "ForcingFunction",
    "InterpolatedForcing",
    "JAXBackend",
    "PyTorchBackend",
    "ScipyBackend",
    "TensorFlowBackend",
    "UnifiedForcingFactory",
    "create_constantfunc",
    "create_forcing_function",
    "create_interpolated_forcing",
    "create_ndoses",
    "create_onoff",
    "create_perdose",
    "create_zerofunc",
    "extract_forcing_switch_times",
]
