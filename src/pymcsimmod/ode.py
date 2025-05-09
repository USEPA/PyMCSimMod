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

        if self.use_jax:
            self.dep_var_names = list(self.Y0.keys())
        else:
            self.dep_var_names = jnp.array(self.Y0.keys())
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

    def model(self, t, y):
        """Build a generic tuple of dydt from the model tree. 
        
        """
        dydt = []
        # y is a vector (jnp.ndarray or np.ndarray)
        for state in self.dep_var_names:
            expr = self.model_tree.dynamics[state]
            val = self.evaluate_expression(expr, y)
            dydt.append(val)
        if self.use_jax:
            return jnp.array(dydt)
        else:
            return np.array(dydt)
    
    def evaluate_expression(self, expr, y=None) -> float:
        """Evaluate a mathematical expression using the current parameters and state vector y."""
        if isinstance(expr, Identifier):
            name = expr.name
            if name in self.parameters:
                return self.parameters[name]
            elif name in self.Y0:
                idx = self.dep_var_names.index(name)
                return y[idx]
            else:
                raise KeyError(f"Unknown identifier '{name}' in expression.")
        elif isinstance(expr, Number):
            return expr.value
        elif isinstance(expr, MathematicalExpression):
            lhs = self.evaluate_expression(expr.lhs, y)
            rhs = self.evaluate_expression(expr.rhs, y)
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

    @abstractmethod
    def run_model(self, times: Sequence) -> Computed_Model:
        raise NotImplementedError("This method should be implemented in a subclass.")

class JAX_Model(ODE_Model):
    def __init__(self):
        super().__init__()
        self.use_jax = True

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

class ODEint_Model(ODE_Model):
    def __init__(self):
        self.use_jax = False

    def run_model(self, times: Sequence) -> Computed_Model:
        """Use the tuple of dydt to build the module-specific model call
        """
        times = np.array(times)

        y_init = np.array([self.Y0[state] for state in self.dep_var_names])
        solution = sci.odeint(
            self.model,
            y_init,  # Initial conditions
            times,  # Time points
            tfirst=True,
        )
        return solution

