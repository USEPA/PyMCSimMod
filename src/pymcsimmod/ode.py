import operator
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import diffrax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as sci
from matplotlib.axes import Axes
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

    Attributes:
        times: Array of time points at which the solution is evaluated.
        states: Array of state variable values at each time point (shape: [n_times, n_states]).
        var_names: List of state variable names (order matches columns of states).
        backend: String indicating the backend used ('jax' or 'scipy').
        raw: The raw solution object from the backend solver (diffrax.Solution or OdeResult).
        calc_dyn: Array of calculated dynamic variable values at each time point (shape: [n_times, n_calcs]).
        calc_names: List of calculated dynamic variable names (order matches columns of calc_dyn).
    """

    times: NumericArray
    states: NumericArray
    var_names: list[str]
    aux_outputs: NumericArray  # shape (n_times, n_calcs)
    aux_names: list[str]

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame with columns for time, state variables, and calculated dynamic variables.
        """
        data = {"time": self.times}
        for i, name in enumerate(self.var_names):
            data[name] = self.states[:, i]
        if self.aux_outputs is not None and self.aux_names is not None:
            for i, name in enumerate(self.aux_names):
                data[name] = self.aux_outputs[:, i]
        return pd.DataFrame(data)

    def plot_results(
        self,
        variables: str | list[str] | None = None,
        ax: Axes | None = None,
        legend: bool = True,
        xlabel: str = "Time",
        ylabel: str = "Value",
        **kwargs,
    ) -> Axes:
        """
        Plot the ODE solution results for selected variables (states or calculated dynamics).

        Args:
            variables: str or list of str, variable names to plot (state or calc_dyn). If None, plot all states.
            ax: Optional matplotlib axis to plot on. If None, a new figure/axis is created.
            legend: Whether to display the legend (default: True).
            xlabel: Label for the x-axis (default: 'Time').
            ylabel: Label for the y-axis (default: 'Value').
            **kwargs: Additional keyword arguments passed to plt.plot.

        Returns:
            The matplotlib axis object containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        if variables is None:
            variables = self.var_names
        if isinstance(variables, str):
            variables = [variables]
        if not all(
            var in self.var_names or (self.aux_names is not None and var in self.aux_names)
            for var in variables
        ):
            raise KeyError(
                f"One or more variables '{variables}' not found in states or calculated dynamics."
            )
        for var in variables:
            if var in self.var_names:
                idx = self.var_names.index(var)
                ax.plot(self.times, self.states[:, idx], label=var, **kwargs)
            elif self.aux_names is not None and var in self.aux_names:
                idx = self.aux_names.index(var)
                ax.plot(self.times, self.aux_outputs[:, idx], label=var, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend()
        return ax

    def __getitem__(self, key: int | str) -> np.ndarray:
        """
        Access state variable arrays by name or index.

        Args:
            key: int (column index) or str (variable name).

        Returns:
            Array of values for the selected variable across all time points.

        Raises:
            KeyError: If the key is not a valid index or variable name.
        """
        if isinstance(key, int):
            return self.states[:, key]
        elif isinstance(key, str):
            idx = self.var_names.index(key)
            return self.states[:, idx]
        else:
            raise KeyError(f"Invalid key: {key}")

    def get_calc(self, key: int | str) -> np.ndarray:
        """
        Access calculated dynamic variable arrays by name or index.

        Args:
            key: int (column index) or str (variable name).

        Returns:
            Array of values for the selected calculated variable across all time points.

        Raises:
            KeyError: If the key is not a valid index or variable name.
        """
        if self.aux_outputs is None or self.aux_names is None:
            raise AttributeError("No calculated dynamics stored in this ComputedModel.")
        if isinstance(key, int):
            return self.aux_outputs[:, key]
        elif isinstance(key, str):
            idx = self.aux_names.index(key)
            return self.aux_outputs[:, idx]
        else:
            raise KeyError(f"Invalid key: {key}")


class OdeModel(ABC):
    def __init__(self, model: str | Path):
        """
        Load and parse a model from a file path or string, initializing parameters and initial conditions.

        Args:
            model: Path to model file or model string.
        """
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
        """
        Assign the parameters and initial conditions (Y0) from the model tree to the model instance.
        """
        self.parameters = self.model_tree.parameters
        self.Y0 = self.model_tree.Y0  # dict(state_var_name: value)

    def update_constants(self, **parameters: float | int) -> None:
        """
        Update any constants in the model tree in place.

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
        """
        Update any initial conditions in the model tree in place.

        Args:
            **Y0: Keyword arguments where keys are state variable names and values are the new initial values.

        Raises:
            KeyError: If a state variable name does not exist in the model tree.
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
        """
        Abstract ODE right-hand side function for subclass implementation.
        Should compute the time derivatives for the system of ODEs.

        Args:
            t: Current time.
            y: Current state vector.
            args: Additional arguments (e.g., parameters).

        Returns:
            Array of time derivatives for each state variable.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def evaluate_expression(self, expr, y) -> object:
        """
        Abstract expression evaluator for subclass implementation.
        Should recursively evaluate a parsed expression tree using the current variable context.

        Args:
            expr: Expression node to evaluate.
            y: Variable context (array or dict, depending on backend).

        Returns:
            Evaluated value of the expression.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def run_model(self, times: Sequence) -> ComputedModel:
        """
        Abstract ODE solver runner for subclass implementation.
        Should solve the ODE system over the given time points and return a ComputedModel.

        Args:
            times: Sequence of time points at which to solve the ODE system.

        Returns:
            ComputedModel instance containing the solution.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")


class JaxModel(OdeModel):
    def __init__(self, model: str | Path):
        """
        Initialize a JaxModel from a model string or file, setting up variable indices for JAX evaluation.

        Args:
            model: Path to model file or model string.
        """
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
        Handles all supported expression types for JAX-based ODE models.

        Args:
            expr: Expression node to evaluate.
            all_vars: JAX array of all variables (state, parameters, calcs).

        Returns:
            Evaluated value as a JAX array or scalar.
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
        Computes the time derivatives for the system of ODEs using the current state and parameters.

        Args:
            t: Current time (ignored for autonomous systems, but required by diffrax signature).
            y: Current state vector (JAX array).
            args: Tuple containing parameter values as a JAX array.

        Returns:
            JAX array of time derivatives for each state variable.
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

    def run_model(self, times: Sequence[int, float]) -> ComputedModel:
        """
        Solve the ODE system using diffrax (JAX backend) and return a ComputedModel, including calculated dynamics.
        All calculations remain JAX-compatible until the ComputedModel, where arrays are converted to numpy.
        """
        ode_term = diffrax.ODETerm(self.model)
        t0 = times[0]
        t_end = times[-1]
        y_init = jnp.array([self.Y0[state] for state in self.dep_var_names])
        param_vals = jnp.array([self.parameters[name] for name in self.param_names])
        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t_end, len(times)))
        sol = diffrax.diffeqsolve(
            ode_term, solver, t0=t0, t1=t_end, dt0=0.1, y0=y_init, saveat=saveat, args=(param_vals,)
        )
        self.sol = sol  # Store the raw solution with JaxModel

        # Calculate the additional dynamic variables (JAX-compatible)
        calc_names = list(self.model_tree.dynamic_calcs.keys())
        n_times = sol.ts.shape[0]
        n_calcs = len(calc_names)
        calc_dyn = jnp.zeros((n_times, n_calcs))
        for i in range(n_times):
            state_vals = sol.ys[i]
            param_vals_jax = jnp.array([self.parameters[name] for name in self.param_names])
            all_vars = jnp.concatenate([state_vals, param_vals_jax, jnp.zeros(n_calcs)])
            for k, cname in enumerate(calc_names):
                expr = self.model_tree.dynamic_calcs[cname]
                val = self.evaluate_expression(expr, all_vars)
                calc_dyn = calc_dyn.at[i, k].set(val)
                all_vars = all_vars.at[len(state_vals) + len(param_vals_jax) + k].set(val)

        # Convert to numpy arrays only in ComputedModel
        return ComputedModel(
            times=np.asarray(sol.ts),
            states=np.asarray(sol.ys),  # shape (n_times, n_states)
            var_names=self.dep_var_names,
            aux_outputs=np.asarray(calc_dyn),
            aux_names=calc_names,
        )


class ScipyModel(OdeModel):
    def __init__(self, model: str | Path):
        """
        Initialize a ScipyModel from a model string or file.

        Args:
            model: Path to model file or model string.
        """
        super().__init__(model=model)

    def model(self, t: float, y: np.ndarray, args: None = None) -> np.ndarray:
        """
        ODE right-hand side function for use with scipy.integrate.solve_ivp.
        Computes the time derivatives for the system of ODEs using the current state and parameters.

        Args:
            t: Current time (ignored for autonomous systems, but required by solve_ivp signature).
            y: Current state vector (NumPy array).
            args: Optional extra arguments (not used, included for compatibility with JAX interface).

        Returns:
            NumPy array of time derivatives for each state variable.
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
        """
        Recursively evaluate an expression using the provided context dictionary.
        Handles all supported expression types for SciPy-based ODE models.

        Args:
            expr: Expression node to evaluate.
            context: Dictionary mapping variable names to their current values.

        Returns:
            Evaluated value as a float or int.
        """

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

    def run_model(self, times: Sequence[int, float]) -> ComputedModel:
        """
        Solve the ODE system using scipy.integrate.solve_ivp and return a ComputedModel, including calculated dynamics.
        """
        times = np.array(times)
        y_init = np.array([self.Y0[state] for state in self.dep_var_names])
        t_span = np.array([times[0], times[-1]])
        sol = sci.solve_ivp(
            fun=self.model,
            t_span=t_span,
            y0=y_init,
            t_eval=times,
        )
        self.sol = sol  # Store the raw solution with ScipyModel

        # Calculate the addtional dynamic variables (TODO: only variables in MCSim Outputs section)
        calc_names = list(self.model_tree.dynamic_calcs.keys())
        calc_dyn = np.zeros((sol.t.shape[0], len(calc_names)))
        for i, t in enumerate(sol.t):
            context = {name: sol.y[:, i][j] for j, name in enumerate(self.dep_var_names)}
            context.update(self.parameters)
            for k, cname in enumerate(calc_names):
                expr = self.model_tree.dynamic_calcs[cname]
                calc_dyn[i, k] = self.evaluate_expression(expr, context)
                context[cname] = calc_dyn[i, k]

        return ComputedModel(
            times=sol.t,
            states=sol.y.T,  # shape (n_times, n_states)
            var_names=self.dep_var_names,
            aux_outputs=calc_dyn,
            aux_names=calc_names,
        )
