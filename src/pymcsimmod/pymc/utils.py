"""Utility functions for PyMC and JAX ODE integration."""

from collections.abc import Callable
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from ..config import BackendType
from ..forcing.unified import UnifiedForcingFactory


def validate_pymc_config(
    model: Any,
    sampled_params: list[str] | None,
    sampled_y0: list[str] | None,
    observed_vars: list[str] | None,
) -> None:
    """Validate that requested parameters, initial conditions, and observed variables exist."""
    if sampled_params is None:
        sampled_params = []
    if sampled_y0 is None:
        sampled_y0 = []
    if observed_vars is None:
        observed_vars = model.state_names

    # Validate parameters
    for p in sampled_params:
        if p not in model.parameters:
            raise KeyError(f"Parameter '{p}' not found in model parameters.")

    # Validate Y0
    for y in sampled_y0:
        if y not in model.state_names:
            raise KeyError(f"Initial condition '{y}' not found in state variables.")

    # Validate observed variables
    for v in observed_vars:
        if v not in model.state_names and v not in model.outputs:
            raise KeyError(f"Observed variable '{v}' not found in states or CalcOutputs.")


def build_jax_solve_function(
    eqx_model: Any,
    times: np.ndarray,
    sampled_param_names: list[str],
    sampled_y0_names: list[str],
    observed_var_names: list[str],
    solver_config: dict[str, Any],
) -> Callable[..., jnp.ndarray]:
    """Build a pure JAX function that solves the ODE model.

    Args:
        eqx_model: The EqxModel instance.
        times: Array of time points.
        sampled_param_names: List of parameter names that are PyMC random variables.
        sampled_y0_names: List of state names whose initial conditions are PyMC random variables.
        observed_var_names: List of observed state/calculated variable names to return.
        solver_config: Dict containing solver settings (solver, rtol, atol, dt0, max_steps).

    Returns:
        A JAX-compatible function solve_fn(*args) -> observed_outputs.
    """
    fixed_params = {k: v for k, v in eqx_model.parameters.items() if k not in sampled_param_names}
    fixed_y0 = {k: v for k, v in eqx_model.Y0.items() if k not in sampled_y0_names}

    t0 = float(times[0])
    t1 = float(times[-1])

    def solve_fn(*args: Any) -> jnp.ndarray:
        n_params = len(sampled_param_names)
        param_vals = args[:n_params]
        y0_vals = args[n_params:]

        # Build parameters dictionary
        params = {k: jnp.asarray(v) for k, v in fixed_params.items()}
        for name, val in zip(sampled_param_names, param_vals, strict=False):
            params[name] = val

        # Build Y0 dictionary
        y0 = {k: jnp.asarray(v) for k, v in fixed_y0.items()}
        for name, val in zip(sampled_y0_names, y0_vals, strict=False):
            y0[name] = val

        # Build initial state array
        y_init = jnp.stack([y0[state] for state in eqx_model.state_names])

        # Compile/rebuild forcing functions
        compiled_forcings = {}
        for input_name, ff in eqx_model.forcing_functions.items():
            if isinstance(ff, dict) and "function" in ff:
                func_name = ff["function"]
                kwargs = ff.get("kwargs", {}).copy()
                resolved_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, str) and v in params:
                        resolved_kwargs[k] = params[v]
                    elif k in params:
                        resolved_kwargs[k] = params[k]
                    else:
                        resolved_kwargs[k] = v

                compiled_forcings[input_name] = UnifiedForcingFactory.create_forcing_function(
                    func_name, backend=BackendType.JAX, **resolved_kwargs
                )
            else:
                compiled_forcings[input_name] = ff

        # Define the JAX ODE derivative function
        def ode_rhs(t: float, y: jnp.ndarray, args_diffrax: Any) -> jnp.ndarray:
            forcing_values = {name: ff(t) for name, ff in compiled_forcings.items()}

            from ..model import Approach
            from ..utils.context import build_evaluation_context

            context = build_evaluation_context(
                state_vals=y,
                state_names=eqx_model.state_names,
                parameters=params,
                forcing_values=forcing_values,
                dynamic_calcs=eqx_model.model_tree.dynamic_calcs,
                approach=Approach.JAX,
            )

            if hasattr(eqx_model.model_tree, "calc_outputs"):
                for var, expr in eqx_model.model_tree.calc_outputs.items():
                    context[var] = expr.evaluate(context, Approach.JAX)

            dydt = [
                eqx_model.model_tree.dynamics[state].evaluate(context, Approach.JAX)
                for state in eqx_model.state_names
            ]
            return jnp.stack(dydt)

        ode_term = diffrax.ODETerm(ode_rhs)

        # Solver configuration
        solver = solver_config.get("solver")
        if solver is None:
            solver = diffrax.Kvaerno5()
        rtol = solver_config.get("rtol", 1e-8)
        atol = solver_config.get("atol", 1e-8)
        dt0 = solver_config.get("dt0", 0.001)
        max_steps = solver_config.get("max_steps", 100000)

        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        saveat = diffrax.SaveAt(ts=jnp.array(times))

        sol = diffrax.diffeqsolve(
            ode_term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y_init,
            saveat=saveat,
            max_steps=max_steps,
            stepsize_controller=stepsize_controller,
        )

        # Map state trajectories and CalcOutputs to observed variables
        def get_observed(y_val: jnp.ndarray, t_val: float) -> jnp.ndarray:
            forcing_values = {name: ff(t_val) for name, ff in compiled_forcings.items()}
            from ..model import Approach
            from ..utils.context import build_evaluation_context

            context = build_evaluation_context(
                state_vals=y_val,
                state_names=eqx_model.state_names,
                parameters=params,
                forcing_values=forcing_values,
                dynamic_calcs=eqx_model.model_tree.dynamic_calcs,
                approach=Approach.JAX,
            )

            if hasattr(eqx_model.model_tree, "calc_outputs"):
                for var, expr in eqx_model.model_tree.calc_outputs.items():
                    context[var] = expr.evaluate(context, Approach.JAX)

            return jnp.stack([context[name] for name in observed_var_names])

        observed_trajectory = jax.vmap(get_observed)(sol.ys, sol.ts)

        if len(observed_var_names) == 1:
            observed_trajectory = observed_trajectory.squeeze(axis=-1)

        return observed_trajectory

    return solve_fn


def build_jax_vjp_function(
    solve_fn: Callable[..., jnp.ndarray], n_args: int
) -> Callable[..., tuple[jnp.ndarray, ...]]:
    """Build VJP function from solve function using jax.vjp."""

    def vjp_fn(*args: Any) -> tuple[jnp.ndarray, ...]:
        solve_args = args[:n_args]
        gz = args[n_args]
        _, vjp_func = jax.vjp(solve_fn, *solve_args)
        return vjp_func(gz)

    return vjp_fn
