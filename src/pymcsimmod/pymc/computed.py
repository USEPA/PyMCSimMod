"""BayesianComputedModel class for storing and post-processing PyMC inference results."""

import inspect
from typing import Any

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .bridge import create_pymc_op


class BayesianComputedModel:
    """Results container for time-course Bayesian ODE inference.

    Wraps PyMC/ArviZ InferenceData and provides convenience methods
    to project time-course predictions and plot probabilistic curves.
    """

    def __init__(
        self,
        idata: Any,
        model: Any,
        times: np.ndarray,
        observed_vars: list[str],
        sampled_params: list[str],
        sampled_y0: list[str],
        param_mapping: dict[str, str] | None = None,
        y0_mapping: dict[str, str] | None = None,
        solver_kwargs: dict[str, Any] | None = None,
    ):
        self.idata = idata
        self.model = model
        self.times = times
        self.observed_vars = observed_vars
        self.sampled_params = sampled_params
        self.sampled_y0 = sampled_y0
        self.param_mapping = (
            param_mapping if param_mapping is not None else {n: n for n in sampled_params}
        )
        self.y0_mapping = y0_mapping if y0_mapping is not None else {n: n for n in sampled_y0}
        self.solver_kwargs = solver_kwargs if solver_kwargs is not None else {}

    def sample_predictions(
        self, num_samples: int = 100, times: np.ndarray | None = None
    ) -> np.ndarray:
        """Project time-course trajectories from posterior parameter draws.

        Uses JAX parallelized vmap solve for high performance.
        """
        if times is None:
            times = self.times
        else:
            if hasattr(self.model, "events") and self.model.events:
                from ..events.utils import check_events
                _, times = check_events(
                    self.model.events, np.asarray(times), list(self.model.state_names)
                )

        posterior = self.idata.posterior
        if hasattr(posterior, "to_dataset"):
            posterior = posterior.to_dataset()
        flat_posterior = posterior.stack(sample=("chain", "draw"))
        total_samples = flat_posterior.sizes["sample"]

        # Randomly select a subset of draws from the posterior
        rng = np.random.default_rng()
        indices = rng.choice(total_samples, size=min(num_samples, total_samples), replace=False)

        # Retrieve parameter arrays using mapped names
        param_arrays = [
            flat_posterior[self.param_mapping.get(name, name)].isel(sample=indices).values
            for name in self.sampled_params
        ]
        y0_arrays = [
            flat_posterior[self.y0_mapping.get(name, name)].isel(sample=indices).values
            for name in self.sampled_y0
        ]

        # Re-create Op for the target times
        op = create_pymc_op(
            model=self.model,
            times=times,
            sampled_params=self.sampled_params,
            sampled_y0=self.sampled_y0,
            observed_vars=self.observed_vars,
            **self.solver_kwargs,
        )

        solve_fn = op._jax_solve_fn
        vmap_solve = jax.vmap(solve_fn)

        args = [jnp.array(arr) for arr in param_arrays] + [jnp.array(arr) for arr in y0_arrays]
        predictions = jax.jit(vmap_solve)(*args)
        return np.asarray(predictions)

    def plot_predictive(
        self,
        var_name: str,
        ax: Any = None,
        hdi_prob: float = 0.94,
        num_samples: int = 100,
        times: np.ndarray | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot posterior predictive median and credible intervals (HDI)."""
        if ax is None:
            _, ax = plt.subplots()

        plot_times = times if times is not None else self.times
        if hasattr(self.model, "events") and self.model.events:
            from ..events.utils import check_events
            _, plot_times = check_events(
                self.model.events, np.asarray(plot_times), list(self.model.state_names)
            )
        predictions = self.sample_predictions(num_samples=num_samples, times=plot_times)

        if len(self.observed_vars) == 1:
            if self.observed_vars[0] != var_name:
                raise ValueError(f"Variable '{var_name}' was not observed.")
            y_samples = predictions
        else:
            if var_name not in self.observed_vars:
                raise ValueError(f"Variable '{var_name}' was not observed.")
            idx = self.observed_vars.index(var_name)
            y_samples = predictions[:, :, idx]

        median = np.median(y_samples, axis=0)

        sig = inspect.signature(az.hdi)
        hdi_kwargs = {}
        if "prob" in sig.parameters:
            hdi_kwargs["prob"] = hdi_prob
        else:
            hdi_kwargs["hdi_prob"] = hdi_prob

        hdi = az.hdi(y_samples, axis=0, **hdi_kwargs)

        ax.plot(plot_times, median, label="Median Fit", color="C0", **kwargs)
        ax.fill_between(
            plot_times,
            hdi[:, 0],
            hdi[:, 1],
            alpha=0.3,
            color="C0",
            label=f"{int(hdi_prob * 100)}% Credible Interval",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel(var_name)
        ax.legend()
        return ax
