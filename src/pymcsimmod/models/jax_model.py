"""JAX-based model implementations using diffrax and equinox."""

from collections.abc import Sequence
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..model import Approach
from .base import OdeModel
from .computed import ComputedModel


class EqxModel(eqx.Module):
    """Modern JAX model using equinox with event handling awareness."""

    parameters: dict = eqx.field()  # Constants
    forcing_functions: dict = eqx.field()  # Forcing functions for inputs
    Y0: dict = eqx.field()  # State variable initial conditions
    events: list = eqx.field()  # Discrete events
    model_tree: object = eqx.field(static=True)
    state_names: tuple = eqx.field()
    output_names: tuple = eqx.field()

    @staticmethod
    @jax.jit
    def OnOff(t, t0, t1, s=10.0):
        """JAX-compiled on-off forcing function."""
        t = jnp.asarray(t)
        t0 = jnp.asarray(t0)
        t1 = jnp.asarray(t1)
        return (jnp.tanh(s * (t - t0)) - jnp.tanh(s * (t - t1))) / 2

    @staticmethod
    def PerDose(t0, duration, period, s=10.0):
        """JAX-compiled periodic dosing function."""
        t0 = float(t0)
        duration = float(duration)
        period = float(period)

        @jax.jit
        def func(t):
            t = jnp.asarray(t)
            n = jnp.floor((t - t0) / period)
            start = t0 + n * period
            stop = start + duration
            return EqxModel.OnOff(t, start, stop, s)

        return func

    @staticmethod
    def NDoses(t0_list, duration, s=10.0):
        """JAX-compiled multiple dosing function."""
        t0_arr = jnp.array(t0_list)
        duration = float(duration)

        @jax.jit
        def func(t):
            t = jnp.asarray(t)
            return jnp.sum(EqxModel.OnOff(t, t0_arr, t0_arr + duration, s), axis=-1)

        return func

    @staticmethod
    def InterpolatedForcing(times, values, **kwargs):
        """
        Create an interpolated forcing function from time-value data for JAX.

        Args:
            times: Array-like of time points.
            values: Array-like of corresponding values.
            **kwargs: Additional parameters for InterpolatedForcing (e.g., interpolation_method).

        Returns:
            JAX-compiled callable function for the interpolated forcing.
        """
        from ..forcing.interpolated import InterpolatedForcing

        forcing = InterpolatedForcing(times, values, **kwargs)
        return forcing.create_function("jax")

    @staticmethod
    def ZeroFunc():
        """JAX-compiled zero function."""

        @jax.jit
        def func(t):
            return 0.0

        return func

    @staticmethod
    def ConstFunc(value):
        """JAX-compiled constant function."""
        value = float(value)  # Ensure value is a float for JAX compatibility

        @jax.jit
        def func(t):
            return value

        return func

    def compile_forcing_functions(self):
        """Convert all dict-based forcing functions to JIT-compiled callables."""
        # Create a new dict to avoid mutating during iteration
        compiled_functions = {}
        for input_name, ff in self.forcing_functions.items():
            # Check if it's a dict-like forcing function specification
            # Use duck typing with specific attribute checks instead of isinstance
            if hasattr(ff, "get") and hasattr(ff, "__getitem__") and "function" in ff:
                # It's a forcing function specification dict
                func_name = ff["function"]
                args = ff.get("args", ())
                kwargs = ff.get("kwargs", {})
                func_factory = getattr(self, func_name, None)
                if func_factory is None or not callable(func_factory):
                    raise AttributeError(f"Forcing function '{func_name}' not found in EqxModel.")
                compiled_functions[input_name] = func_factory(*args, **kwargs)
            else:
                # It's already a compiled function or other callable
                compiled_functions[input_name] = ff

        # Replace forcing_functions with compiled version
        # forcing_functions must be immutable, so use object.__setattr__
        object.__setattr__(self, "forcing_functions", compiled_functions)

    def build_context(self, state_vals, t):
        """Build context dictionary for expression evaluation (JAX-compatible)."""
        # Note: We cannot use the general context utility here because JAX requires
        # fixed structure and pure functions for JIT compilation
        context = {name: state_vals[i] for i, name in enumerate(self.state_names)}
        context.update(self.parameters)
        for input_name, ff in self.forcing_functions.items():
            context[input_name] = ff(t)
        # Evaluate all dynamic calcs
        for var, expr in self.model_tree.dynamic_calcs.items():
            context[var] = expr.evaluate(context, Approach.JAX)
        if hasattr(self.model_tree, "calc_outputs"):
            for var, expr in self.model_tree.calc_outputs.items():
                context[var] = expr.evaluate(context, Approach.JAX)
        return context

    @eqx.filter_jit
    def model(self, t, y):
        """JAX-compiled ODE right-hand side function."""
        context = self.build_context(y, t)
        dydt = [
            self.model_tree.dynamics[state].evaluate(context, Approach.JAX)
            for state in self.state_names
        ]
        return jnp.stack(dydt)

    def run_model(self, times, max_steps=100000, dt0=0.01, solver=None):
        """
        Run the JAX model with event handling checks.

        Args:
            times: Sequence of time points at which to solve the ODE system.
            max_steps: Maximum number of solver steps (default: 100000).
            dt0: Initial step size for the solver (default: 0.01).
            solver: Diffrax solver to use. If None, uses Dopri8() (default: None).
        """
        # Check for events and warn
        if self.events:
            raise NotImplementedError(
                "Discrete events are not yet supported for JAX-based models. "
                "Please use ScipyModel for models with discrete events, or consider "
                "implementing events as continuous forcing functions."
            )

        # Compile forcing functions before running ODE solve
        self.compile_forcing_functions()
        t0 = float(times[0])
        t_end = float(times[-1])
        y_init = jnp.asarray([self.Y0[state] for state in self.state_names], dtype=jnp.float32)

        @eqx.filter_jit
        def ode_rhs(t, y, args):
            return self.model(t, y)

        ode_term = diffrax.ODETerm(ode_rhs)
        # Use provided solver or default to Dopri8
        if solver is None:
            solver = diffrax.Dopri8()
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t_end, len(times)))
        sol = diffrax.diffeqsolve(
            ode_term, solver, t0=t0, t1=t_end, dt0=dt0, y0=y_init, saveat=saveat,
            max_steps=max_steps, args=()
        )

        @eqx.filter_jit
        def calc_outputs_single(state_vals, t):
            context = self.build_context(state_vals, t)
            return jnp.array([context[name] for name in self.output_names], dtype=jnp.float32)

        calc_outputs = jax.vmap(calc_outputs_single, in_axes=(0, 0))(sol.ys, sol.ts)
        # Return the compiled forcing functions for plotting
        return sol, calc_outputs, dict(self.forcing_functions)


class JaxModel(OdeModel):
    """JAX-based ODE model implementation."""

    def __init__(self, model: str | Path):
        """
        Initialize a JaxModel from a model string or file.

        Args:
            model: Path to model file or model string.
        """
        super().__init__(model=model)

    def _get_approach(self) -> Approach:
        """Get the evaluation approach for JaxModel."""
        return Approach.JAX

    def model(self, t: float, y, args) -> object:
        """Placeholder - actual implementation is in EqxModel."""
        raise NotImplementedError("This method should be implemented in equinox module class.")

    def run_model(self, times: Sequence[int, float], max_steps=100000, dt0=0.01, solver=None) -> ComputedModel:
        """
        Solve the ODE system using diffrax (JAX backend) and return a ComputedModel.

        Args:
            times: Sequence of time points at which to solve the ODE system.
            max_steps: Maximum number of solver steps (default: 100000).
            dt0: Initial step size for the solver (default: 0.01).
            solver: Diffrax solver to use. If None, uses Dopri8() (default: None).

        Returns:
            ComputedModel instance containing the solution.
        """
        eqx_model = self._to_eqx()
        sol, calc_outputs, input_functions = eqx_model.run_model(times, max_steps=max_steps, dt0=dt0, solver=solver)
        return ComputedModel(
            times=np.asarray(sol.ts),
            states=np.asarray(sol.ys),
            var_names=self.state_names,
            aux_outputs=np.asarray(calc_outputs),
            aux_names=self.outputs,
            input_functions=input_functions,
        )

    def _to_eqx(self):
        """Convert to EqxModel for JAX computation."""
        # Use tuples for state_names/output_names for JAX compatibility
        return EqxModel(
            parameters=self.parameters.copy(),
            forcing_functions=self.forcing_functions.copy(),
            Y0=self.Y0.copy(),
            events=self.events.copy(),
            model_tree=self.model_tree,
            state_names=tuple(self.state_names),
            output_names=tuple(self.outputs),
        )


__all__ = ["EqxModel", "JaxModel"]
