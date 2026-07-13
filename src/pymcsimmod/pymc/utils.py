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
    validated_events: list[Any] | None = None,
) -> Callable[..., jnp.ndarray]:
    """Build a pure JAX function that solves the ODE model, with event handling awareness.

    Args:
        eqx_model: The EqxModel instance.
        times: Array of time points.
        sampled_param_names: List of parameter names that are PyMC random variables.
        sampled_y0_names: List of state names whose initial conditions are PyMC random variables.
        observed_var_names: List of observed state/calculated variable names to return.
        solver_config: Dict containing solver settings (solver, rtol, atol, dt0, max_steps).
        validated_events: Pre-validated discrete events to apply.

    Returns:
        A JAX-compatible function solve_fn(*args) -> observed_outputs.
    """
    fixed_params = {k: v for k, v in eqx_model.parameters.items() if k not in sampled_param_names}
    fixed_y0 = {k: v for k, v in eqx_model.Y0.items() if k not in sampled_y0_names}

    t0 = float(times[0])
    t1 = float(times[-1])

    if validated_events is None:
        validated_events = []

    def solve_fn(*args: Any) -> jnp.ndarray:
        float_type = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

        n_params = len(sampled_param_names)
        param_vals = args[:n_params]
        y0_vals = args[n_params:]

        # Build parameters dictionary
        params = {k: jnp.asarray(v, dtype=float_type) for k, v in fixed_params.items()}
        for name, val in zip(sampled_param_names, param_vals, strict=False):
            params[name] = jnp.asarray(val, dtype=float_type)

        # Build Y0 dictionary
        y0 = {k: jnp.asarray(v, dtype=float_type) for k, v in fixed_y0.items()}
        for name, val in zip(sampled_y0_names, y0_vals, strict=False):
            y0[name] = jnp.asarray(val, dtype=float_type)

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

        if not validated_events:
            saveat = diffrax.SaveAt(ts=jnp.array(times, dtype=float_type))
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
            all_ys = sol.ys
            all_ts = sol.ts
        else:
            # Map event methods to integers for JAX compatibility
            method_map = {"add": 0, "replace": 1, "multiply": 2}

            event_times = jnp.array([e.time for e in validated_events], dtype=float_type)
            event_indices = jnp.array(
                [eqx_model.state_names.index(e.state_var) for e in validated_events], dtype=jnp.int32
            )
            event_methods = jnp.array(
                [method_map[e.method] for e in validated_events], dtype=jnp.int32
            )

            # Resolve event values against PyMC parameter tracers
            resolved_event_vals = []
            for e in validated_events:
                val = e.value
                if isinstance(val, str):
                    neg = False
                    if val.startswith("-"):
                        val = val[1:]
                        neg = True
                    inv = False
                    if val.startswith("1.0/"):
                        val = val[4:]
                        inv = True

                    if val in params:
                        resolved_val = params[val]
                    else:
                        resolved_val = jnp.array(float(val), dtype=float_type)

                    if neg:
                        resolved_val = -resolved_val
                    if inv:
                        resolved_val = 1.0 / resolved_val
                else:
                    resolved_val = jnp.array(val, dtype=float_type)
                resolved_event_vals.append(resolved_val)

            event_values = jnp.stack(resolved_event_vals)

            def apply_events_jax(t, y, ev_vals):
                num_events = len(validated_events)
                for i in range(num_events):
                    idx = event_indices[i]
                    val = ev_vals[i]
                    method = event_methods[i]
                    ev_t = event_times[i]

                    # Use a tolerance appropriate for float32 time grids
                    is_event_t = jnp.isclose(ev_t, t, rtol=0.0, atol=1e-6)

                    val_add = y[idx] + val
                    val_replace = val
                    val_multiply = y[idx] * val

                    new_val = jnp.where(
                        method == 0,
                        val_add,
                        jnp.where(method == 1, val_replace, val_multiply),
                    )

                    target_val = jnp.where(is_event_t, new_val, y[idx])
                    y = y.at[idx].set(target_val)
                return y

            # Apply events at the start time t0 if any
            y_init = apply_events_jax(t0, y_init, event_values)

            # Calculate unique segment boundaries based on event times
            unique_event_times = sorted(list(set(e.time for e in validated_events)))
            segment_boundaries = [t0] + [t for t in unique_event_times if t0 < t < t1] + [t1]
            segment_boundaries = sorted(list(set(segment_boundaries)))

            segment_starts = segment_boundaries[:-1]
            segment_ends = segment_boundaries[1:]

            segment_ys = []
            carry_y = y_init
            for i in range(len(segment_starts)):
                t_start = float(segment_starts[i])
                t_end = float(segment_ends[i])

                sol_seg = diffrax.diffeqsolve(
                    ode_term,
                    solver,
                    t0=t_start,
                    t1=t_end,
                    dt0=dt0,
                    y0=carry_y,
                    saveat=diffrax.SaveAt(dense=True),
                    max_steps=max_steps,
                    stepsize_controller=stepsize_controller,
                )

                # Evaluate dense interpolation for all times in this segment
                clipped_ts = jnp.clip(times, t_start, t_end)
                ys_seg = jax.vmap(sol_seg.interpolation.evaluate)(clipped_ts)

                # Apply event at t_end
                y_end = sol_seg.interpolation.evaluate(t_end)
                carry_y = apply_events_jax(t_end, y_end, event_values)

                # Apply mask: B_i <= t < B_i+1 (or B_i <= t <= B_i+1 for last segment)
                if i < len(segment_starts) - 1:
                    mask = (times >= t_start) & (times < t_end)
                    ys_to_use = ys_seg
                else:
                    mask = (times >= t_start) & (times <= t_end)
                    is_t1 = (times == t_end)
                    ys_to_use = jnp.where(jnp.expand_dims(is_t1, axis=-1), carry_y, ys_seg)

                mask = jnp.expand_dims(mask, axis=-1)
                segment_ys.append(jnp.where(mask, ys_to_use, 0.0))

            all_ys = sum(segment_ys)
            all_ts = jnp.array(times, dtype=float_type)

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

        observed_trajectory = jax.vmap(get_observed)(all_ys, all_ts)

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
