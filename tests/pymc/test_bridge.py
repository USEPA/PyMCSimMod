"""Tests for PyMC integration and bridge."""

import numpy as np
import pytest

pytest.importorskip("pymc")
pytest.importorskip("pytensor")
pytest.importorskip("jax")

import pymc as pm
import pytensor

from pymcsimmod.models.jax_model import JaxModel
from pymcsimmod.pymc.bridge import BayesianODEModel, create_pymc_op


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


def test_op_creation_and_forward(exponential_model_str):
    """Test that MCSimModOp is successfully created and returns correct forward predictions."""
    model = JaxModel(exponential_model_str)
    times = np.linspace(0, 4, 5)  # [0, 1, 2, 3, 4]

    # Create Op with ke as sampled parameter
    sol_op = create_pymc_op(model, times, sampled_params=["ke"], observed_vars=["A"])

    # Expect shape (5,)
    assert sol_op._output_shape == (5,)

    # Evaluate the JAX solve function directly for ke=0.5
    res = sol_op._jitted_solve(0.5)
    expected = np.exp(-times * 0.5)
    np.testing.assert_allclose(res, expected, rtol=1e-5)


def test_grad_verification(exponential_model_str):
    """Verify gradients using PyTensor's verify_grad helper."""
    model = JaxModel(exponential_model_str)
    times = np.linspace(0, 2, 3)

    sol_op = create_pymc_op(model, times, sampled_params=["ke"], observed_vars=["A"])

    # Test verify_grad
    # verify_grad expects a function, input values as tuple, and RNG
    pytensor.gradient.verify_grad(sol_op, (np.array(0.5),), rng=np.random.default_rng())


def test_end_to_end_sampling(exponential_model_str):
    """Test standard pm.sample integration end-to-end."""
    model = JaxModel(exponential_model_str)
    times = np.linspace(0, 4, 5)

    # True parameter value
    true_ke = 0.3
    model.update_constants(ke=true_ke)
    true_sol = model.run_model(times).states.squeeze()

    # Generate synthetic noisy data
    np.random.seed(42)
    observed_data = true_sol + np.random.normal(0, 0.02, size=len(times))

    # Reset model to default parameters
    model.update_constants(reset_to_defaults=True)

    sol_op = model.create_pymc_op(times, sampled_params=["ke"], observed_vars=["A"])

    with pm.Model():
        ke_prior = pm.HalfNormal("ke", sigma=1.0)
        predictions = sol_op(ke_prior)
        sigma = pm.HalfNormal("sigma", sigma=0.1)
        pm.Normal("obs", mu=predictions, sigma=sigma, observed=observed_data)

        # Draw a small number of samples to verify it runs successfully
        idata = pm.sample(draws=100, tune=100, chains=2, random_seed=42)

    # Assert posterior has sampled
    assert "ke" in idata.posterior
    assert idata.posterior["ke"].values.shape == (2, 100)


def test_y0_as_random_variable(exponential_model_str):
    """Test that initial conditions (Y0) can be passed as random variables."""
    model = JaxModel(exponential_model_str)
    times = np.linspace(0, 4, 5)

    # Sample both parameters (ke) and initial conditions (A)
    sol_op = model.create_pymc_op(
        times, sampled_params=["ke"], sampled_y0=["A"], observed_vars=["A"]
    )

    # Call Op with both values
    res = sol_op._jitted_solve(0.5, 2.0)
    expected = 2.0 * np.exp(-times * 0.5)
    np.testing.assert_allclose(res, expected, rtol=1e-5)

    # Verify gradients with respect to both inputs
    pytensor.gradient.verify_grad(
        sol_op, (np.array(0.5), np.array(2.0)), rng=np.random.default_rng()
    )


def test_high_level_api(exponential_model_str):
    """Test the high-level BayesianODEModel API wrapper."""
    model = JaxModel(exponential_model_str)
    times = np.linspace(0, 4, 5)

    bayes = BayesianODEModel(model, times)

    with pm.Model():
        ke = pm.HalfNormal("ke", sigma=1.0)
        y0 = pm.Normal("A_init", mu=1.0, sigma=0.1)

        predictions = bayes.solve(params={"ke": ke}, y0={"A": y0}, observed_vars=["A"])

        sigma = pm.HalfNormal("sigma", sigma=0.1)
        pm.Normal("obs", mu=predictions, sigma=sigma, observed=np.zeros(5))

        idata = pm.sample(draws=50, tune=50, chains=1, random_seed=42)

    # Verify posterior predictive projection
    computed = bayes.posterior_predictive(idata)
    preds = computed.sample_predictions(num_samples=10)
    assert preds.shape == (10, 5)

    # Test plotting
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    computed.plot_predictive(var_name="A", ax=ax, num_samples=10)
    plt.close(fig)


def test_complex_pk_model_inference(complex_pk_model_str):
    """Test inference on a multi-state complex PK model observing a CalcOutput."""
    model = JaxModel(complex_pk_model_str)
    times = np.linspace(0, 10, 11)

    # Sample elimination rate (ke), absorption rate (ka), and initial amount (A0)
    sol_op = model.create_pymc_op(
        times=times,
        sampled_params=["ke", "ka"],
        sampled_y0=["A0"],
        observed_vars=["C"],  # C is a CalcOutput (A1 / V)
    )

    # Evaluate op directly: ke=0.15, ka=0.8, A0=100.0
    res = sol_op._jitted_solve(0.15, 0.8, 100.0)
    assert res.shape == (11,)

    # Verify gradients with relaxed tolerance due to numeric ODE solver error
    pytensor.gradient.verify_grad(
        sol_op,
        (np.array(0.15), np.array(0.8), np.array(100.0)),
        rng=np.random.default_rng(),
        abs_tol=1e-3,
        rel_tol=1e-3,
    )
