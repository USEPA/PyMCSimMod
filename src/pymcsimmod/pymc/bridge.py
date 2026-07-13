"""PyMC integration bridge for MCSimMod JAX ODE models."""

from typing import Any

import jax
import numpy as np
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify

from .utils import build_jax_solve_function, build_jax_vjp_function, validate_pymc_config


class MCSimModOp(Op):
    """PyTensor Op wrapping an MCSimMod JAX ODE solver.

    Handles:
    - Running the JAX solver during PyMC model compilation
    - Providing gradient support via VJP Op
    - Enabling JAX JIT and sampler compilation
    """

    __props__ = ("_n_params", "_n_y0", "_output_shape", "_name")

    def __init__(
        self,
        jax_solve_fn: Any,
        jax_vjp_fn: Any,
        n_params: int,
        n_y0: int,
        output_shape: tuple[int, ...],
        name: str,
    ):
        self._jax_solve_fn = jax_solve_fn
        self._jax_vjp_fn = jax_vjp_fn
        self._jitted_solve = jax.jit(jax_solve_fn)
        self._n_params = n_params
        self._n_y0 = n_y0
        self._output_shape = output_shape
        self._name = name

    def make_node(self, *args: Any) -> Apply:
        inputs = [pt.as_tensor_variable(a) for a in args]
        if len(self._output_shape) == 1:
            outputs = [pt.vector(dtype="float64", shape=self._output_shape)]
        else:
            outputs = [pt.matrix(dtype="float64", shape=self._output_shape)]
        return Apply(self, inputs, outputs)

    def perform(
        self, node: Apply, inputs: list[np.ndarray], outputs: list[list[np.ndarray]]
    ) -> None:
        result = self._jitted_solve(*inputs)
        outputs[0][0] = np.asarray(result, dtype="float64")

    def grad(self, inputs: list[Any], output_gradients: list[Any]) -> list[Any]:
        (gz,) = output_gradients
        vjp_op = MCSimModVJPOp(self._jax_vjp_fn, self._n_params, self._n_y0)
        vjp_node = vjp_op.make_node(*inputs, gz)
        return vjp_node.outputs


class MCSimModVJPOp(Op):
    """VJP computation Op for MCSimModOp."""

    __props__ = ("_n_params", "_n_y0")

    def __init__(self, jax_vjp_fn: Any, n_params: int, n_y0: int):
        self._jax_vjp_fn = jax_vjp_fn
        self._jitted_vjp = jax.jit(jax_vjp_fn)
        self._n_params = n_params
        self._n_y0 = n_y0

    def make_node(self, *args_and_gz: Any) -> Apply:
        inputs = [pt.as_tensor_variable(a) for a in args_and_gz]
        n_inputs = self._n_params + self._n_y0
        outputs = [inputs[i].type() for i in range(n_inputs)]
        return Apply(self, inputs, outputs)

    def perform(
        self, node: Apply, inputs: list[np.ndarray], outputs: list[list[np.ndarray]]
    ) -> None:
        n_inputs = self._n_params + self._n_y0
        solve_inputs = inputs[:n_inputs]
        gz = inputs[n_inputs]
        grads = self._jitted_vjp(*solve_inputs, gz)
        # Handle single vs multiple inputs in VJP tuple unpacking
        if n_inputs == 1:
            outputs[0][0] = np.asarray(
                grads[0] if isinstance(grads, tuple | list) else grads, dtype="float64"
            )
        else:
            for i, g in enumerate(grads):
                outputs[i][0] = np.asarray(g, dtype="float64")


# Register dispatch for PyTensor JAX Linker
@jax_funcify.register(MCSimModOp)
def mcsimmod_op_jax_funcify(op: MCSimModOp, **kwargs: Any) -> Any:
    return op._jax_solve_fn


@jax_funcify.register(MCSimModVJPOp)
def mcsimmod_vjp_op_jax_funcify(op: MCSimModVJPOp, **kwargs: Any) -> Any:
    return op._jax_vjp_fn


def create_pymc_op(
    model: Any,
    times: np.ndarray,
    sampled_params: list[str] | None = None,
    sampled_y0: list[str] | None = None,
    observed_vars: list[str] | None = None,
    solver: Any = None,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    dt0: float = 0.001,
    max_steps: int = 100000,
) -> MCSimModOp:
    """Create a PyTensor Op from an MCSimMod JaxModel for use in PyMC.

    Args:
        model: A JaxModel instance.
        times: Array of time points.
        sampled_params: List of parameter names that will be PyMC random variables.
        sampled_y0: List of state variable names whose initial conditions will be RVs.
        observed_vars: Which state or CalcOutputs variables to return.
        solver: Diffrax solver to use (default: Kvaerno5).
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        dt0: Initial step size.
        max_steps: Max step limit.

    Returns:
        MCSimModOp to use inside pm.Model().
    """
    from ..models.jax_model import JaxModel

    if not isinstance(model, JaxModel):
        raise TypeError("model must be a JaxModel instance.")

    validate_pymc_config(model, sampled_params, sampled_y0, observed_vars)

    if sampled_params is None:
        sampled_params = []
    if sampled_y0 is None:
        sampled_y0 = []
    if observed_vars is None:
        observed_vars = model.state_names

    eqx_model = model._to_eqx()
    eqx_model.compile_forcing_functions()

    # Pre-validate events against times
    from ..events.utils import check_events
    if hasattr(eqx_model, "events") and eqx_model.events:
        validated_events, modified_times = check_events(
            eqx_model.events, np.asarray(times), list(eqx_model.state_names)
        )
    else:
        validated_events = []
        modified_times = np.asarray(times)

    solver_config = {
        "solver": solver,
        "rtol": rtol,
        "atol": atol,
        "dt0": dt0,
        "max_steps": max_steps,
    }

    solve_fn = build_jax_solve_function(
        eqx_model=eqx_model,
        times=modified_times,
        sampled_param_names=sampled_params,
        sampled_y0_names=sampled_y0,
        observed_var_names=observed_vars,
        solver_config=solver_config,
        validated_events=validated_events,
    )

    n_args = len(sampled_params) + len(sampled_y0)
    vjp_fn = build_jax_vjp_function(solve_fn, n_args)

    n_times = len(modified_times)
    if len(observed_vars) == 1:
        output_shape = (n_times,)
    else:
        output_shape = (n_times, len(observed_vars))

    return MCSimModOp(
        jax_solve_fn=solve_fn,
        jax_vjp_fn=vjp_fn,
        n_params=len(sampled_params),
        n_y0=len(sampled_y0),
        output_shape=output_shape,
        name=f"MCSimModOp_{model.__class__.__name__}",
    )


class BayesianODEModel:
    """High-level wrapper class for Bayesian inference with MCSimMod models."""

    def __init__(self, model: Any, times: np.ndarray, **solver_kwargs: Any):
        self.model = model
        self.times = times
        self.solver_kwargs = solver_kwargs
        self._op = None
        self._sampled_params = []
        self._sampled_y0 = []
        self._observed_vars = []
        self._param_mapping = {}
        self._y0_mapping = {}

    def solve(
        self,
        params: dict[str, Any],
        y0: dict[str, Any] | None = None,
        observed_vars: list[str] | None = None,
    ) -> Any:
        """Create PyTensor Op and call it with PyMC random variables."""
        if y0 is None:
            y0 = {}

        sampled_params = list(params.keys())
        sampled_y0 = list(y0.keys())

        self._sampled_params = sampled_params
        self._sampled_y0 = sampled_y0

        self._param_mapping = {
            p: params[p].name for p in sampled_params if hasattr(params[p], "name")
        }
        self._y0_mapping = {
            y: y0[y].name for y in sampled_y0 if hasattr(y0[y], "name")
        }

        if observed_vars is None:
            observed_vars = self.model.state_names
        self._observed_vars = observed_vars

        # Run check_events to update self.times if events exist
        if hasattr(self.model, "events") and self.model.events:
            from ..events.utils import check_events
            _, self.times = check_events(
                self.model.events, np.asarray(self.times), list(self.model.state_names)
            )

        self._op = create_pymc_op(
            model=self.model,
            times=self.times,
            sampled_params=sampled_params,
            sampled_y0=sampled_y0,
            observed_vars=observed_vars,
            **self.solver_kwargs,
        )

        args = [params[p] for p in sampled_params] + [y0[y] for y in sampled_y0]
        return self._op(*args)

    def posterior_predictive(self, idata: Any) -> Any:
        """Generate posterior predictive ODE trajectories."""
        from .computed import BayesianComputedModel

        return BayesianComputedModel(
            idata=idata,
            model=self.model,
            times=self.times,
            observed_vars=self._observed_vars,
            sampled_params=self._sampled_params,
            sampled_y0=self._sampled_y0,
            param_mapping=self._param_mapping,
            y0_mapping=self._y0_mapping,
            solver_kwargs=self.solver_kwargs,
        )
