"""Tests for JAX-based discrete event integration."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pymcsimmod.models.events import DiscreteEvent
from pymcsimmod.models.jax_model import JaxModel
from pymcsimmod.models.scipy_model import ScipyModel


def test_jax_single_event_add():
    """Test a single add event with JaxModel."""
    model_str = """
    States = {
        A
    };
    Initialize {
        A = 0.0;
    }
    Dynamics {
        dt(A) = -0.1 * A;
    }
    End.
    """
    model = JaxModel(model_str)
    model.add_event(time=5.0, state_var="A", value=10.0, method="add")

    times = np.linspace(0, 10, 11)
    result = model.run_model(times)

    # Check times include 5.0
    assert 5.0 in result.times

    # Find the index of t=5.0 in result.times
    idx_5 = np.where(np.abs(result.times - 5.0) < 1e-12)[0][0]

    # At t=5.0, the state should be updated with the event (+10.0)
    # Solve analytically: A(5-) = 0, A(5+) = 10, A(10) = 10 * exp(-0.1 * 5) = 6.065
    np.testing.assert_allclose(result.states[idx_5, 0], 10.0, rtol=1e-5)
    
    idx_10 = np.where(np.abs(result.times - 10.0) < 1e-12)[0][0]
    expected_10 = 10.0 * np.exp(-0.1 * 5.0)
    np.testing.assert_allclose(result.states[idx_10, 0], expected_10, rtol=1e-3)


def test_jax_single_event_replace():
    """Test a single replace event with JaxModel."""
    model_str = """
    States = {
        A
    };
    Initialize {
        A = 5.0;
    }
    Dynamics {
        dt(A) = 0.0;  # constant state
    }
    End.
    """
    model = JaxModel(model_str)
    model.add_event(time=3.0, state_var="A", value=12.0, method="replace")

    times = [0.0, 2.0, 3.0, 5.0]
    result = model.run_model(times)

    # Before t=3: state should be 5.0
    # At and after t=3: state should be 12.0
    np.testing.assert_allclose(result.states[0, 0], 5.0)
    np.testing.assert_allclose(result.states[1, 0], 5.0)
    np.testing.assert_allclose(result.states[2, 0], 12.0)
    np.testing.assert_allclose(result.states[3, 0], 12.0)


def test_jax_single_event_multiply():
    """Test a single multiply event with JaxModel."""
    model_str = """
    States = {
        A
    };
    Initialize {
        A = 5.0;
    }
    Dynamics {
        dt(A) = 0.0;  # constant state
    }
    End.
    """
    model = JaxModel(model_str)
    model.add_event(time=3.0, state_var="A", value=2.5, method="multiply")

    times = [0.0, 2.0, 3.0, 5.0]
    result = model.run_model(times)

    # Before t=3: state should be 5.0
    # At and after t=3: state should be 5.0 * 2.5 = 12.5
    np.testing.assert_allclose(result.states[0, 0], 5.0)
    np.testing.assert_allclose(result.states[1, 0], 5.0)
    np.testing.assert_allclose(result.states[2, 0], 12.5)
    np.testing.assert_allclose(result.states[3, 0], 12.5)


def test_jax_multiple_events():
    """Test multiple sequential discrete events with JaxModel."""
    model_str = """
    States = {
        A
    };
    Initialize {
        A = 0.0;
    }
    Dynamics {
        dt(A) = 0.0;
    }
    End.
    """
    model = JaxModel(model_str)
    model.add_event(time=2.0, state_var="A", value=3.0, method="add")
    model.add_event(time=5.0, state_var="A", value=2.0, method="multiply")
    model.add_event(time=8.0, state_var="A", value=10.0, method="replace")

    times = [0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0]
    result = model.run_model(times)

    # Expected trajectory:
    # t=0, 1: 0.0
    # t=2, 4: 3.0
    # t=5, 7: 3.0 * 2.0 = 6.0
    # t=8, 10: 10.0
    expected = [0.0, 0.0, 3.0, 3.0, 6.0, 6.0, 10.0, 10.0]
    np.testing.assert_allclose(result.states[:, 0], expected)


def test_jax_scipy_equivalence():
    """Test that JaxModel events produce equivalent results to ScipyModel events."""
    model_str = """
    States = {
        A
    };
    Initialize {
        A = 10.0;
    }
    Dynamics {
        dt(A) = -0.5 * A;
    }
    End.
    """
    jax_model = JaxModel(model_str)
    scipy_model = ScipyModel(model_str)

    # Add same events to both models
    for model in [jax_model, scipy_model]:
        model.add_event(time=2.0, state_var="A", value=5.0, method="add")
        model.add_event(time=4.0, state_var="A", value=0.5, method="multiply")
        model.add_event(time=7.0, state_var="A", value=1.0, method="replace")

    times = np.linspace(0, 10, 101)
    
    jax_res = jax_model.run_model(times)
    scipy_res = scipy_model.run_model(times)

    np.testing.assert_allclose(jax_res.times, scipy_res.times, rtol=1e-6, atol=1e-6)
    
    # Filter out event times from states comparison as JAX reports post-event states 
    # at boundary times whereas SciPy reports pre-event states.
    mask = ~np.isin(jax_res.times, [2.0, 4.0, 7.0])
    np.testing.assert_allclose(jax_res.states[mask], scipy_res.states[mask], rtol=1e-2)


def test_jax_events_differentiability():
    """Test that JaxModel is fully differentiable with respect to event values using JAX."""
    import equinox as eqx

    model_str = """
    States = {
        A
    };
    Initialize {
        A = 0.0;
    }
    Dynamics {
        dt(A) = -0.1 * A;
    }
    End.
    """
    model = JaxModel(model_str)
    eqx_model = model._to_eqx()

    def solve_and_sum(event_val):
        # We can construct the event inside the differentiable function
        ev = DiscreteEvent(time=5.0, state_var="A", value=event_val, method="add")

        # Pass events directly to run_model
        sol, _, _ = eqx_model.run_model([0.0, 5.0, 10.0], events=[ev])
        # Return sum of states at the final time point as a scalar
        return jnp.sum(sol.ys)

    # Compute gradient with respect to event_val
    grad_fn = jax.grad(solve_and_sum)
    g = grad_fn(10.0)

    # Assert gradient is not NaN and is non-zero
    assert not jnp.isnan(g)
    assert g != 0.0
    
    # Analytical verification:
    # A(0) = 0
    # A(5-) = 0
    # A(5+) = event_val
    # A(10) = event_val * exp(-0.1 * 5) = event_val * exp(-0.5)
    # Sum of states = A(0) + A(5) + A(10) = 0 + event_val + event_val * exp(-0.5) = event_val * (1 + exp(-0.5))
    # d(Sum)/d(event_val) = 1 + exp(-0.5) = 1 + 0.60653 = 1.60653
    expected_grad = 1.0 + np.exp(-0.5)
    np.testing.assert_allclose(g, expected_grad, rtol=1e-4)
