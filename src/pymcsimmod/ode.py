from abc import ABC, abstractmethod
from collections.abc import Sequence

import diffrax
import jax.numpy as jnp
import numpy as np
import scipy.integrate as sci
from pydantic import BaseModel

from .model import (
    Identifier,
    MathematicalExpression,
    Number,
    ParenthesizedExpression,
    SignedExpression,
)
from .parser import ModelParser


class Computed_Model(BaseModel):
    pass


class ODE_Model(ABC):
    def __init__(self):
        self.calc_outputs = []  # calculated outputs from CalcOutputs Section

    def from_model(self, path: str | None = None, model_str: str | None = None) -> None:
        """Load a model from a file path. This is a placeholder for the actual implementation."""
        parser = ModelParser()
        if path is None and model_str is None:
            raise ValueError("Either path to .model or model_str must be provided.")

        if path is not None:
            with open(path) as file:
                parsed_model = parser.parse(file.read())
        elif model_str is not None:
            parsed_model = parser.parse(model_str)

        self.model_tree = parsed_model.model_copy()

        # Once model is loaded, initialize the model parameters and intitial conditions
        self._init_parameters()

        self.dep_var_names = list(self.Y0.keys())
        self.dep_var_indices = {name: i for i, name in enumerate(self.dep_var_names)}

    def _init_parameters(self) -> None:
        """Assign the parameters and intitial conditions (Y0) from the model tree to the model instance."""
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
        for key, value in parameters.items():
            if key in self.parameters:
                self.parameters[key] = value
            else:
                raise KeyError(f"Parameter '{key}' does not exist in the model tree.")

    def update_Y0(self, **Y0: float | int) -> None:
        """Update any initial conditions in the model tree in place. If a key passed in the
        Y0 dictionary does not exist in the model tree, it will raise an exception.

        Args:
            **parameters: Keyword arguments where keys are parameter names and values are the new values.

        Raises:
            KeyError: If a parameter name does not exist in the model tree.
        """
        for key, value in Y0.items():
            if key in self.Y0:
                self.Y0[key] = value
            else:
                raise KeyError(
                    f"Initial condiditon for state '{key}' does not exist in the model tree."
                )

    @abstractmethod
    def model(self, t, y):
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def evaluate_expression(self, expr, y):
        raise NotImplementedError("This method should be implemented in a subclass.")

    @abstractmethod
    def run_model(self, times: Sequence) -> Computed_Model:
        raise NotImplementedError("This method should be implemented in a subclass.")


class Jax_Model(ODE_Model):
    def __init__(self):
        super().__init__()
        self.use_jax = True
        # Will be set in from_model
        self.all_var_names = []
        self.all_var_indices = {}

    def from_model(self, path: str | None = None, model_str: str | None = None) -> None:
        super().from_model(path=path, model_str=model_str)
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
            if expr.operator == "+":
                return lhs + rhs
            elif expr.operator == "-":
                return lhs - rhs
            elif expr.operator == "*":
                return lhs * rhs
            elif expr.operator == "/":
                return lhs / rhs
            else:
                raise ValueError(f"Unknown operator '{expr.operator}' in expression.")
        elif hasattr(expr, "expression"):
            return self.evaluate_expression(expr.expression, all_vars)
        elif hasattr(expr, "condition") and hasattr(expr, "if_true") and hasattr(expr, "if_false"):
            cond = expr.condition
            lhs = self.evaluate_expression(cond.lhs, all_vars)
            rhs = self.evaluate_expression(cond.rhs, all_vars)
            op = cond.operator
            if op == "==":
                result = lhs == rhs
            elif op == "!=":
                result = lhs != rhs
            elif op == "<":
                result = lhs < rhs
            elif op == ">":
                result = lhs > rhs
            elif op == "<=":
                result = lhs <= rhs
            elif op == ">=":
                result = lhs >= rhs
            else:
                raise ValueError(f"Unknown condition operator: {op}")
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

    def model(self, t, y, args):
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

    def run_model(self, times: Sequence) -> Computed_Model:
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
        return solution


class Scipy_Model(ODE_Model):
    def __init__(self):
        self.use_jax = False

    def model(self, t, y):
        """Build a tuple of dydt from the model tree, using dynamic_calcs for intermediate variables and generic expression evaluation."""
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
        Recursively evaluate an expression using the provided context dict.
        Context should include state variables, parameters, and any calculated variables.
        This is for evaluating expressions for a Python-specific ODE model.
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
            if expr.operator == "+":
                return lhs + rhs
            elif expr.operator == "-":
                return lhs - rhs
            elif expr.operator == "*":
                return lhs * rhs
            elif expr.operator == "/":
                return lhs / rhs
            else:
                raise ValueError(f"Unknown operator '{expr.operator}' in expression.")
        # ParenthesizedExpression
        elif isinstance(expr, ParenthesizedExpression):
            return self.evaluate_expression(expr.expression, context)
        # TernaryExpression
        elif hasattr(expr, "condition") and hasattr(expr, "if_true") and hasattr(expr, "if_false"):
            cond = expr.condition
            lhs = self.evaluate_expression(cond.lhs, context)
            rhs = self.evaluate_expression(cond.rhs, context)
            op = cond.operator
            if op == "==":
                result = lhs == rhs
            elif op == "!=":
                result = lhs != rhs
            elif op == "<":
                result = lhs < rhs
            elif op == ">":
                result = lhs > rhs
            elif op == "<=":
                result = lhs <= rhs
            elif op == ">=":
                result = lhs >= rhs
            else:
                raise ValueError(f"Unknown condition operator: {op}")
            return (
                self.evaluate_expression(expr.if_true, context)
                if result
                else self.evaluate_expression(expr.if_false, context)
            )
        # MathematicalFunction (e.g., pow)
        elif hasattr(expr, "func") and hasattr(expr, "args"):
            if expr.func == "pow":
                args = [self.evaluate_expression(arg, context) for arg in expr.args]
                return pow(*args)
            else:
                raise ValueError(f"Unknown function: {expr.func}")
        # SpecialFunction (e.g., BetaRandom)
        elif hasattr(expr, "func") and hasattr(expr, "args"):
            # For now, just return 0 or raise (implement as needed)
            raise NotImplementedError(f"Special function {expr.func} not implemented.")
        else:
            raise TypeError(f"Unsupported expression type: {type(expr)}")

    def run_model(self, times: Sequence) -> Computed_Model:
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
        return sol
