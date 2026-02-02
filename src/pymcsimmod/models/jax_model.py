"""JAX-based model implementations using diffrax and equinox."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..config import BackendType
from ..model import Approach
from .base import OdeModel
from .computed import ComputedModel


class EqxModel(eqx.Module):
    """Modern JAX model using equinox with event handling awareness."""

    parameters: dict[str, float] = eqx.field()  # Constants
    forcing_functions: dict[str, Callable] = eqx.field()  # Forcing functions for inputs
    Y0: dict[str, float] = eqx.field()  # State variable initial conditions
    events: list[Any] = eqx.field()  # Discrete events
    model_tree: Any = eqx.field(static=True)
    state_names: tuple[str, ...] = eqx.field()
    output_names: tuple[str, ...] = eqx.field()
    backend: BackendType = eqx.field(static=True, default=BackendType.JAX)

    def compile_forcing_functions(self) -> None:
        """Convert all dict-based forcing functions to JIT-compiled callables using unified backend."""
        from ..forcing.unified import UnifiedForcingFactory

        # Create a new dict to avoid mutating during iteration
        compiled_functions = {}
        for input_name, ff in self.forcing_functions.items():
            # Check if it's a dict-like forcing function specification
            # Use duck typing with specific attribute checks instead of isinstance
            if hasattr(ff, "get") and hasattr(ff, "__getitem__") and "function" in ff:
                # It's a forcing function specification dict
                func_name = ff["function"]
                kwargs = ff.get("kwargs", {})

                # Use unified forcing function factory for all forcing functions
                compiled_functions[input_name] = UnifiedForcingFactory.create_forcing_function(
                    func_name, backend=self.backend, **kwargs
                )
            else:
                # It's already a compiled function or other callable
                compiled_functions[input_name] = ff

        # Replace forcing_functions with compiled version
        # forcing_functions must be immutable, so use object.__setattr__
        object.__setattr__(self, "forcing_functions", compiled_functions)

    def build_context(self, state_vals: jnp.ndarray, t: float) -> dict[str, Any]:
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
    def model(self, t: float, y: jnp.ndarray) -> jnp.ndarray:
        """JAX-compiled ODE right-hand side function."""
        context = self.build_context(y, t)
        dydt = [
            self.model_tree.dynamics[state].evaluate(context, Approach.JAX)
            for state in self.state_names
        ]
        return jnp.stack(dydt)

    def run_model(
        self,
        times: Sequence[float],
        max_steps: int = 100000,
        dt0: float = 0.001,
        solver: diffrax.AbstractSolver | None = None,
        rtol: float = 1e-10,
        atol: float = 1e-10,
    ) -> tuple[diffrax.Solution, jnp.ndarray, dict[str, Callable]]:
        """
        Run the JAX model with event handling checks.

        Args:
            times: Sequence of time points at which to solve the ODE system.
            max_steps: Maximum number of solver steps (default: 100000).
            dt0: Initial step size for the solver (default: 0.001).
            solver: Diffrax solver to use. If None, uses Dopri8() (default: None).
            rtol: Relative tolerance for adaptive step size control (default: 1e-10).
            atol: Absolute tolerance for adaptive step size control (default: 1e-10).

        Returns:
            Tuple of (solution, calculated outputs, input functions dictionary).
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
            solver = diffrax.Kvaerno5()

        # Use adaptive step size control with specified tolerances for better accuracy
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        saveat = diffrax.SaveAt(ts=jnp.array(times))
        sol = diffrax.diffeqsolve(
            ode_term,
            solver,
            t0=t0,
            t1=t_end,
            dt0=dt0,
            y0=y_init,
            saveat=saveat,
            max_steps=max_steps,
            stepsize_controller=stepsize_controller,
            args=(),
        )

        # Check for NaN values in the solution and provide helpful guidance
        if jnp.any(jnp.isnan(sol.ys)):
            raise ValueError(
                f"JAX model integration produced NaN values. This often indicates numerical "
                f"instability. Try the following fixes:\n"
                f"1. Use a smaller initial step size: run_model(times, dt0={dt0 / 100:.6f})\n"
                f"2. Try a different solver: run_model(times, solver=diffrax.Dopri5())\n"
                f"3. For stiff systems, consider: run_model(times, solver=diffrax.Kvaerno5())\n"
                f"4. If the issue persists, try the ScipyModel backend instead.\n"
                f"Current settings: dt0={dt0}, solver={type(solver).__name__}"
            )

        @eqx.filter_jit
        def calc_outputs_single(state_vals: jnp.ndarray, t: float) -> jnp.ndarray:
            context = self.build_context(state_vals, t)
            return jnp.array([context[name] for name in self.output_names], dtype=jnp.float32)

        calc_outputs = jax.vmap(calc_outputs_single, in_axes=(0, 0))(sol.ys, sol.ts)

        # Check for NaN values in calculated outputs too
        if jnp.any(jnp.isnan(calc_outputs)):
            raise ValueError(
                f"JAX model calculated outputs contain NaN values. This may indicate issues "
                f"with auxiliary calculations (e.g., division by zero). Consider:\n"
                f"1. Checking for zero denominators in your model equations\n"
                f"2. Using a smaller step size: dt0={dt0 / 100:.6f}\n"
                f"3. Switching to ScipyModel for better numerical stability"
            )
        # Return the compiled forcing functions for plotting
        return sol, calc_outputs, dict(self.forcing_functions)


class JaxModel(OdeModel):
    """JAX-based ODE model implementation."""

    backend = BackendType.JAX

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

    def model(self, t: float, y: jnp.ndarray, args: Any) -> object:
        """Placeholder - actual implementation is in EqxModel."""
        raise NotImplementedError("This method should be implemented in equinox module class.")

    def run_model(
        self,
        times: Sequence[int | float],
        max_steps: int = 100000,
        dt0: float = 0.001,
        solver: diffrax.AbstractSolver | None = None,
        rtol: float = 1e-9,
        atol: float = 1e-11,
    ) -> ComputedModel:
        """
        Solve the ODE system using diffrax (JAX backend) and return a ComputedModel.

        Args:
            times: Sequence of time points at which to solve the ODE system.
            max_steps: Maximum number of solver steps (default: 100000).
            dt0: Initial step size for the solver (default: 0.001).
            solver: Diffrax solver to use. If None, uses Dopri8() (default: None).
            rtol: Relative tolerance for adaptive step size control (default: 1e-9).
            atol: Absolute tolerance for adaptive step size control (default: 1e-11).

        Returns:
            ComputedModel instance containing the solution.

        Raises:
            ValueError: If the integration produces NaN values, indicating numerical
                instability. See error message for suggested fixes including smaller
                dt0 values or different solvers.

        Note:
            JAX models may be more sensitive to numerical instability than SciPy models.
            If you encounter NaN values, try reducing dt0 (e.g., dt0=0.0001) or using
            a different solver. For very stiff systems, consider using ScipyModel instead.

            For optimal mass balance in PBPK models:
            - Default tolerances (rtol=1e-9, atol=1e-11) work well for most cases
            - For high doses or long simulations, try tighter tolerances (rtol=1e-10, atol=1e-12)
            - If mass balance is critical, compare with ScipyModel results
        """
        eqx_model = self._to_eqx()
        sol, calc_outputs, input_functions = eqx_model.run_model(
            times, max_steps=max_steps, dt0=dt0, solver=solver, rtol=rtol, atol=atol
        )
        return ComputedModel(
            times=np.asarray(sol.ts),
            states=np.asarray(sol.ys),
            var_names=self.state_names,
            aux_outputs=np.asarray(calc_outputs),
            aux_names=self.outputs,
            input_functions=input_functions,
        )

    def _to_eqx(self) -> EqxModel:
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
