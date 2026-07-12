"""Tests for PyMC ODE utility functions."""

import numpy as np
import pytest

pytest.importorskip("jax")

from pymcsimmod.models.jax_model import JaxModel
from pymcsimmod.pymc.utils import build_jax_solve_function, validate_pymc_config


@pytest.fixture
def exponential_model_str():
    """Simple exponential decay model string."""
    return """
    States = {
        A
    };

    ke = 0.5;

    Initialize {
        A = 1.0;
    }
    Dynamics {
        dt(A) = - A * ke;
    }
    End.
    """


def test_validate_pymc_config(exponential_model_str):
    """Test parameter, Y0, and observed variable validation logic."""
    model = JaxModel(exponential_model_str)

    # Valid configs should pass without exception
    validate_pymc_config(model, sampled_params=["ke"], sampled_y0=["A"], observed_vars=["A"])
    validate_pymc_config(model, sampled_params=None, sampled_y0=None, observed_vars=None)

    # Invalid parameter name
    with pytest.raises(KeyError, match="Parameter 'invalid_param' not found"):
        validate_pymc_config(
            model, sampled_params=["invalid_param"], sampled_y0=None, observed_vars=None
        )

    # Invalid state variable name
    with pytest.raises(KeyError, match="Initial condition 'invalid_state' not found"):
        validate_pymc_config(
            model, sampled_params=None, sampled_y0=["invalid_state"], observed_vars=None
        )

    # Invalid observed variable name
    with pytest.raises(KeyError, match="Observed variable 'invalid_observed' not found"):
        validate_pymc_config(
            model, sampled_params=None, sampled_y0=None, observed_vars=["invalid_observed"]
        )


def test_build_jax_solve_function(exponential_model_str):
    """Test direct building and evaluation of JAX solve functions."""
    model = JaxModel(exponential_model_str)
    times = np.linspace(0, 4, 5)

    eqx_model = model._to_eqx()
    eqx_model.compile_forcing_functions()

    solve_fn = build_jax_solve_function(
        eqx_model=eqx_model,
        times=times,
        sampled_param_names=["ke"],
        sampled_y0_names=["A"],
        observed_var_names=["A"],
        solver_config={},
    )

    # Evaluate solve function directly: ke=0.5, A=2.0
    res = solve_fn(0.5, 2.0)
    expected = 2.0 * np.exp(-times * 0.5)
    np.testing.assert_allclose(res, expected, rtol=1e-5)
