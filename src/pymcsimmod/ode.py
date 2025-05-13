import operator
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import diffrax
import jax.numpy as jnp
import numpy as np
import scipy.integrate as sci
from pydantic import BaseModel

from pymcsimmod.extra_typing import NumericArray

from .model import (
    Identifier,
    MathematicalExpression,
    Number,
    ParenthesizedExpression,
    SignedExpression,
)
from .parser import ModelParser


class ComputedModel(BaseModel):
    """
    Adapter class for ODE solution results from either JaxModel (diffrax) or ScipyModel (solve_ivp).
    Provides a unified interface for accessing time, state, and plotting results.
    """

    times: NumericArray
    states: NumericArray
    var_names: list[str]
    backend: str  # 'jax' or 'scipy'
    raw: object  # The raw solution object (diffrax.Solution or OdeResult)

    @classmethod
    def from_scipy(cls, sol, var_names: list[str]):
        """
        Construct a ComputedModel from a scipy.integrate.OdeResult.
        Args:
            sol: OdeResult from scipy.integrate.solve_ivp
            var_names: List of variable names (state variable order)
        """
        return cls(
            times=sol.t,
            states=sol.y.T,  # shape (n_times, n_states)
            var_names=var_names,
            backend="scipy",
            raw=sol,
        )

    @classmethod
    def from_jax(cls, sol, var_names: list[str]):
        """
        Construct a ComputedModel from a diffrax solution.
        Args:
            sol: diffrax.Solution
            var_names: List of variable names (state variable order)
        """
        return cls(
            times=np.asarray(sol.ts),
            states=np.asarray(sol.ys),
            var_names=var_names,
            backend="jax",
            raw=sol,
        )

    def plot_results(self, ax=None, show=True, legend=True, **kwargs):
        """
        Plot the ODE solution results for all state variables.
        Args:
            ax: Optional matplotlib axis to plot on.
            show: Whether to call plt.show().
            legend: Whether to show legend.
            **kwargs: Additional arguments to plt.plot.
        Returns:
            The matplotlib axis object.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        for i, name in enumerate(self.var_names):
            ax.plot(self.times, self.states[:, i], label=name, **kwargs)
        ax.set_xlabel("Time")
        ax.set_ylabel("State")
        if legend:
            ax.legend()
        if show:
            plt.show()
        return ax

    def __getitem__(self, key):
        """
        Allow access to state variable arrays by name or index.
        """
        if isinstance(key, int):
            return self.states[:, key]
        elif isinstance(key, str):
            idx = self.var_names.index(key)
            return self.states[:, idx]
        else:
            raise KeyError(f"Invalid key: {key}")

    def __repr__(self):
        return (
            f"ComputedModel(backend={self.backend!r}, times=shape{self.times.shape}, "
            f"states=shape{self.states.shape}, var_names={self.var_names})"
        )


class OdeModel(ABC):
    def __init__(self, model: str | Path):
        """Load a model from a file path. This is a placeholder for the actual implementation."""
        model_str = model.read_text() if isinstance(model, Path) else model

        parser = ModelParser()
        parsed_model = parser.parse(model_str)
        self.model_tree = parsed_model.model_copy()

        # Once model is loaded, initialize the model parameters and initial conditions
        self.calc_outputs = []  # calculated outputs from CalcOutputs Section
        self._init_parameters()

        self.dep_var_names = list(self.Y0.keys())
        self.dep_var_indices = {name: i for i, name in enumerate(self.dep_var_names)}

    def _init_parameters(self) -> None:
        """Assign the parameters and initial conditions (Y0) from the model tree to the model instance."""
        self.parameters = self.model_tree.parameters
        self.Y0 = self.model_tree.Y0  # dict(state_var_name: value)

    def update_constants(self, **parameters: float | int) -> None:
        """Update any constants in the model tree in place. If a key passed in the
        parameters dictionary does not exist in the model tree, it will raise an exception.

        Args:
            **parameters: Keyword arguments where keys are parameter names and values are the new values.

        Raises:
            KeyError: If a parameter name does not exist in the model tree.
        """
        missing = [key for key in parameters if key not in self.parameters]
        if missing:
            raise KeyError(f"Parameter(s) '{', '.join(missing)}' do not exist in the model tree.")

        for key, value in parameters.items():
            self.parameters[key] = value

    def update_Y0(self, **Y0: float | int) -> None:
        """Update any initial conditions in the model tree in place. If a key passed in the
        Y0 dictionary does not exist in the model tree, it will raise an exception.

        Args:
            **parameters: Keyword arguments where keys are parameter names and values are the new values.

        Raises:
            KeyError: If a parameter name does not exist in the model tree.
        """
        missing = [key for key in Y0 if key not in self.Y0]
        if missing:
            raise KeyError(
                f"Initial condition(s) '{', '.join(missing)}' do not exist in the model tree."
            )

        for key, value in Y0.items():
            self.Y0[key] = value

    @abstractmethod
    def model(self, t: float, y, args) -> object:
        """Abstract ODE right-hand side function for subclass implementation."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def evaluate_expression(self, expr, y) -> object:
        """Abstract expression evaluator for subclass implementation."""
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def run_model(self, times: Sequence) -> ComputedModel:
        """Abstract ODE solver runner for subclass implementation."""
        raise NotImplementedError("This method should be implemented in a subclass.")


class JaxModel(OdeModel):
    def __init__(self, model: str | Path):
        super().__init__(model=model)
        # Will be set in from_model
        self.all_var_names = []
        self.all_var_indices = {}

        self.state_names = list(self.Y0.keys())
        self.param_names = list(self.parameters.keys())
        self.calc_names = list(self.model_tree.dynamic_calcs.keys())
        self.all_var_names = self.state_names + self.param_names + self.calc_names
        self.all_var_indices = {name: i for i, name in enumerate(self.all_var_names)}

    def evaluate_expression(
        self, expr: MathematicalExpression, all_vars: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Recursively evaluate an expression using the provided flat JAX array of variables.
        This is for evaluating expressions for a jax-specific ODE model.
        """
        idx = self.all_var_indices
        if isinstance(expr, Identifier):
            name = expr.name
            return all_vars[idx[name]]
        elif isinstance(expr, Number):
            return jnp.asarray(expr.value)
        elif isinstance(expr, SignedExpression):
            val = self.evaluate_expression(expr.expression, all_vars)
            return val if expr.sign == "+" else -val
        elif isinstance(expr, MathematicalExpression):
            lhs = self.evaluate_expression(expr.lhs, all_vars)
            rhs = self.evaluate_expression(expr.rhs, all_vars)

            expression_map = {
                "+": operator.add,
                "-": operator.sub,
                "*": operator.mul,
                "/": operator.truediv,
                "pow": jnp.power,
            }
            if expr.operator not in expression_map:
                raise ValueError(f"Unknown operator '{expr.operator}' in expression.")
            return expression_map[expr.operator](lhs, rhs)
        elif hasattr(expr, "expression"):
            return self.evaluate_expression(expr.expression, all_vars)
        elif hasattr(expr, "condition") and hasattr(expr, "if_true") and hasattr(expr, "if_false"):
            cond = expr.condition
            lhs = self.evaluate_expression(cond.lhs, all_vars)
            rhs = self.evaluate_expression(cond.rhs, all_vars)
            condition_map = {
                "==": operator.eq,
                "!=": operator.ne,
                "<": operator.lt,
                ">": operator.gt,
                "<=": operator.le,
                ">=": operator.ge,
            }
            if cond.operator not in condition_map:
                raise ValueError(f"Unknown condition '{cond.operator}' in expression.")
            result = condition_map[cond.operator](lhs, rhs)
            return (
                self.evaluate_expression(expr.if_true, all_vars)
                if result
                else self.evaluate_expression(expr.if_false, all_vars)
            )
        elif hasattr(expr, "func") and hasattr(expr, "args"):
            if expr.func == "pow":
                args = [self.evaluate_expression(arg, all_vars) for arg in expr.args]
                return jnp.power(*args)
            else:
                raise ValueError(f"Unknown function: {expr.func}")
        else:
            raise TypeError(f"Unsupported expression type: {type(expr)}")

    def model(self, t: float, y: jnp.ndarray, args: tuple[jnp.ndarray, ...]) -> jnp.ndarray:
        """
        ODE right-hand side function for use with JAX-based solvers (e.g., diffrax).

        This function computes the time derivatives (dydt) for the system of ODEs defined in the model.
        It builds a flat JAX array of all variables (state, parameters, calculated variables), then evaluates
        the ODE right-hand sides using the model's expression tree. This function is passed as the vector field
        to diffrax's ODETerm.

        Args:
            t: Current time (ignored for autonomous systems, but required by diffrax signature).
            y: Current state vector (JAX array).
            args: Tuple containing parameter values as a JAX array.

        Returns:
            dydt: JAX array of time derivatives for each state variable.
        """
        # args: tuple (param_vals,)
        param_vals = args[0]
        # y: state variables (jnp array)
        all_vars = y
        all_vars = jnp.concatenate([all_vars, param_vals])
        # Dynamic calcs (compute in order, using current all_vars)
        for name in self.calc_names:
            expr = self.model_tree.dynamic_calcs[name]
            val = self.evaluate_expression(expr, all_vars)
            all_vars = jnp.concatenate([all_vars, jnp.atleast_1d(val)])
        dydt = [
            self.evaluate_expression(self.model_tree.dynamics[state], all_vars)
            for state in self.state_names
        ]
        return jnp.stack(dydt)

    def run_model(self, times: Sequence) -> ComputedModel:
        ode_term = diffrax.ODETerm(self.model)
        t0 = times[0]
        t_end = times[-1]
        y_init = jnp.array([self.Y0[state] for state in self.dep_var_names])

        # Store parameter values as a jnp array for passing to ODE after all updates are done
        param_vals = jnp.array([self.parameters[name] for name in self.param_names])

        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t_end, len(times)))
        # Pass param_vals as args
        solution = diffrax.diffeqsolve(
            ode_term, solver, t0=t0, t1=t_end, dt0=0.1, y0=y_init, saveat=saveat, args=(param_vals,)
        )
        return ComputedModel.from_jax(solution, self.dep_var_names)


class ScipyModel(OdeModel):
    def __init__(self, model: str | Path):
        super().__init__(model=model)

    def model(self, t: float, y: np.ndarray, args: None = None) -> np.ndarray:
        """
        ODE right-hand side function for use with scipy.integrate.solve_ivp.

        This function computes the time derivatives (dydt) for the system of ODEs defined in the model.
        It builds a context from the current state vector `y`, model parameters, and any calculated variables
        (from the Dynamics section), then evaluates the ODE right-hand sides using the model's expression tree.
        This function is passed as the `fun` argument to solve_ivp.

        Args:
            t: Current time (ignored for autonomous systems, but required by solve_ivp signature).
            y: Current state vector (NumPy array).
            args: Optional extra arguments (not used, included for compatibility with JAX interface).

        Returns:
            dydt: Array of time derivatives for each state variable.
        """
        # Build context: state variables, parameters, and calculated variables
        context = {name: y[i] for i, name in enumerate(self.dep_var_names)}
        context.update(self.parameters)
        # Compute dynamic_calcs (e.g., C) and store them in context
        for var, expr in self.model_tree.dynamic_calcs.items():
            context[var] = self.evaluate_expression(expr, context)
        # Compute dydt, using context (which now includes calc vars)
        dydt = []
        for state in self.dep_var_names:
            expr = self.model_tree.dynamics[state]
            val = self.evaluate_expression(expr, context)
            dydt.append(val)
        return np.array(dydt)

    def evaluate_expression(
        self, expr: MathematicalExpression, context: dict[str, float | int]
    ) -> float | int:
        import operator

        # Identifier
        if isinstance(expr, Identifier):
            name = expr.name
            if name in context:
                return context[name]
            raise KeyError(f"Unknown identifier '{name}' in expression.")
        # Number
        elif isinstance(expr, Number):
            return expr.value
        # SignedExpression (must be checked before generic hasattr checks)
        elif isinstance(expr, SignedExpression):
            val = self.evaluate_expression(expr.expression, context)
            return val if expr.sign == "+" else -val
        # MathematicalExpression
        elif isinstance(expr, MathematicalExpression):
            lhs = self.evaluate_expression(expr.lhs, context)
            rhs = self.evaluate_expression(expr.rhs, context)
            expression_map = {
                "+": operator.add,
                "-": operator.sub,
                "*": operator.mul,
                "/": operator.truediv,
                "pow": pow,
            }
            if expr.operator in expression_map:
                return expression_map[expr.operator](lhs, rhs)
            else:
                raise ValueError(f"Unknown operator '{expr.operator}' in expression.")
        # PowFunction support
        elif hasattr(expr, "func") and hasattr(expr, "args"):
            if getattr(expr, "func", None) == "pow":
                args = [self.evaluate_expression(arg, context) for arg in expr.args]
                return pow(*args)
            else:
                raise ValueError(f"Unknown function: {getattr(expr, 'func', None)}")
        # ParenthesizedExpression
        elif isinstance(expr, ParenthesizedExpression):
            return self.evaluate_expression(expr.expression, context)
        # TernaryExpression
        elif hasattr(expr, "condition") and hasattr(expr, "if_true") and hasattr(expr, "if_false"):
            cond = expr.condition
            lhs = self.evaluate_expression(cond.lhs, context)
            rhs = self.evaluate_expression(cond.rhs, context)
            condition_map = {
                "==": operator.eq,
                "!=": operator.ne,
                "<": operator.lt,
                ">": operator.gt,
                "<=": operator.le,
                ">=": operator.ge,
            }
            if cond.operator in condition_map:
                result = condition_map[cond.operator](lhs, rhs)
            else:
                raise ValueError(f"Unknown condition operator: {cond.operator}")
            return (
                self.evaluate_expression(expr.if_true, context)
                if result
                else self.evaluate_expression(expr.if_false, context)
            )
        # SpecialFunction (e.g., BetaRandom)
        elif hasattr(expr, "func") and hasattr(expr, "args"):
            # For now, just return 0 or raise (implement as needed)
            raise NotImplementedError(f"Special function {expr.func} not implemented.")
        else:
            raise TypeError(f"Unsupported expression type: {type(expr)}")

    def run_model(self, times: Sequence) -> ComputedModel:
        """Use the tuple of dydt to build the module-specific model call"""
        times = np.array(times)
        y_init = np.array([self.Y0[state] for state in self.dep_var_names])
        t_span = np.array([times[0], times[-1]])
        # Use solve_ivp instead of odeint
        sol = sci.solve_ivp(
            fun=self.model,
            t_span=t_span,
            y0=y_init,
            t_eval=times,
        )
        return ComputedModel.from_scipy(sol, self.dep_var_names)
