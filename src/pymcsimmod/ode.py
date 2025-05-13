from .model import Identifier, Number, MathematicalExpression
from .parser import ModelParser
from typing import Sequence
from pydantic import BaseModel
import collections

import jax.numpy as jnp
import numpy as np
from scipy.integrate import odeint
import diffrax
import scipy.integrate as sci

from abc import ABC, abstractmethod

class Computed_Model(BaseModel):
    pass

class ODEVars:
    def __init__(self, fnames: list[str]):
        self._fnames = fnames
        self._vals = []
    def _clear(self):
        self._vals = []
    def _tonumpy(self) -> np.array:
        for f in self._fnames:
            self._vals.append(self.__getattribute__(f))
        z = np.array(self._vals)
        return z
    def _tojax(self) -> jnp.ndarray:
        for f in self._fnames:
            self._vals.append(self.__getattribute__(f))
        z = jnp.array(self._vals)
        return z

class ODE_Model(ABC):
    def __init__(self):
        #self.calc_outputs = [] # calculated outputs from CalcOutputs Section
        pass
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

        #if self.use_jax:
        self.dep_var_names = list(self.Y0.keys())
        self.dep_var_indices = {name: i for i, name in enumerate(self.dep_var_names)}
        #else:
        #    self.dep_var_names = jnp.array(self.Y0.keys())
        #self._dep_vars = collections.namedtuple('dep_vars', self.dep_var_names)

        

    def _init_parameters(self) -> None:
        """Assign the parameters from the model tree to the model instance. This includes Y0"""
        self.parameters = self.model_tree.parameters
        self.Y0 = self.model_tree.Y0 # dict(state_var_name: value)

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

    def update_Y0(self, Y0: float | int) -> None:
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
                raise KeyError(f"Initial condiditon for state '{key}' does not exist in the model tree.")
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

    def evaluate_expression(self, expr, y, extra_vars=None):
        if extra_vars is None:
            extra_vars = {}
        if isinstance(expr, Identifier):
            name = expr.name
            if name in extra_vars:
                return extra_vars[name]
            if name in self.parameters:
                return self.parameters[name]
            elif name in self.Y0:
                idx = self.dep_var_indices[name]
                return y[idx]
            else:
                raise KeyError(f"Unknown identifier '{name}' in expression.")
        elif isinstance(expr, Number):
            return expr.value
        elif isinstance(expr, MathematicalExpression):
            lhs = self.evaluate_expression(expr.lhs, y, extra_vars=extra_vars)
            rhs = self.evaluate_expression(expr.rhs, y, extra_vars=extra_vars)
            if expr.operator == '+':
                return lhs + rhs
            elif expr.operator == '-':
                return lhs - rhs
            elif expr.operator == '*':
                return lhs * rhs
            elif expr.operator == '/':
                return lhs / rhs
            else:
                raise ValueError(f"Unknown operator '{expr.operator}' in expression.")
        else:
            raise TypeError(f"Unsupported expression type: {type(expr)}")

    def model(self, t, y):
        """Build a tuple of dydt from the model tree, using dynamic_calcs for intermediate variables."""
        # Step 1: Compute dynamic_calcs (e.g., C) and store them
        calc_vars = {}
        for var, expr in self.model_tree.dynamic_calcs.items():
            calc_vars[var] = self.evaluate_expression(expr, y, extra_vars=calc_vars)

        # Step 2: Compute dydt, using calc_vars if needed
        dydt = []
        for state in self.dep_var_names:
            expr = self.model_tree.dynamics[state]
            val = self.evaluate_expression(expr, y, extra_vars=calc_vars)
            dydt.append(val)
        return jnp.array(dydt)

    def run_model(self, times: Sequence) -> Computed_Model:
        """Use the tuple of dydt to build the module-specific model call
        """
        #times = jnp.array(times)
        ode_term = diffrax.ODETerm(self.model.__func__)

        t0 = times[0]
        t_end = times[-1]

        y_init = jnp.array([self.Y0[state] for state in self.dep_var_names])
        print(y_init)
        #saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t_end, 500))
        
        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t_end, 500))

        solution = diffrax.diffeqsolve(
            ode_term,
            solver,
            t0=t0,
            t1=t_end,
            dt0=0.1,
            y0=y_init,
            saveat=saveat,
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
            context[var] = self.model_tree.evaluate_expression(expr, context)
        # Compute dydt, using context (which now includes calc vars)
        dydt = []
        for state in self.dep_var_names:
            expr = self.model_tree.dynamics[state]
            val = self.model_tree.evaluate_expression(expr, context)
            dydt.append(val)
        return np.array(dydt)

    def evaluate_expression(self, expr, y, extra_vars=None):
        # Deprecated: use model_tree.evaluate_expression instead
        raise NotImplementedError("Use model_tree.evaluate_expression(expr, context) instead.")

    def run_model(self, times: Sequence) -> Computed_Model:
        """Use the tuple of dydt to build the module-specific model call
        """
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
