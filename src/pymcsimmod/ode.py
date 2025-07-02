from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as sci
from matplotlib.axes import Axes
from pydantic import BaseModel

from pymcsimmod.extra_typing import NumericArray

from .model import (
    Approach,
    Expression,
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
    input_functions: dict | None = None  # Maps input name to callable

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

    def plot_inputs(
        self,
        variables: str | list[str] | None = None,
        ax: Axes | None = None,
        legend: bool = True,
        xlabel: str = "Time",
        ylabel: str = "Input Value",
        **kwargs,
    ) -> Axes:
        """
        Plot the input (forcing) functions over the solution time grid.

        Args:
            variables: str or list of str, input names to plot. If None, plot all.
            ax: Optional matplotlib axis to plot on. If None, a new figure/axis is created.
            legend: Whether to display the legend (default: True).
            xlabel: Label for the x-axis (default: 'Time').
            ylabel: Label for the y-axis (default: 'Input Value').
            **kwargs: Additional keyword arguments passed to plt.plot.

        Returns:
            The matplotlib axis object containing the plot.
        """
        if self.input_functions is None:
            raise AttributeError("No input functions stored in this ComputedModel.")
        if ax is None:
            fig, ax = plt.subplots()
        if variables is None:
            variables = list(self.input_functions.keys())
        if isinstance(variables, str):
            variables = [variables]
        for var in variables:
            if var not in self.input_functions:
                raise KeyError(f"Input '{var}' not found in input_functions.")
            # Evaluate input function at all time points
            values = [self.input_functions[var](t) for t in self.times]
            ax.plot(self.times, values, label=var, **kwargs)
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

        self.inputs = parsed_model.inputs
        self.outputs = parsed_model.outputs
        self.state_names = list(self.Y0.keys())
        self.dep_var_indices = {name: i for i, name in enumerate(self.state_names)}
        # Assign default forcing functions: all inputs get ZeroFunc with correct dict structure
        self.forcing_functions = {
            input_name: {'function': 'ZeroFunc', 'args': (), 'kwargs': {}}
            for input_name in self.inputs
        }

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

    def assign_forcing_function(self, input_name, forcing_function_name, *args, **kwargs):
        """
        Assign a forcing function to an input variable, storing only the function name and parameters (not the factory or callable).
        Args:
            input_name: Name of the input variable to assign the forcing function to.
            forcing_function_name: Name of the forcing function ('PerDose', 'NDoses', etc.).
            *args, **kwargs: Parameters for the forcing function.
        Raises:
            ValueError: If input_name is not in self.inputs.
        """
        if not hasattr(self, 'forcing_functions'):
            self.forcing_functions = {}
        if input_name not in self.inputs:
            raise ValueError(f"'{input_name}' is not a valid input variable. Valid inputs: {self.inputs}")
        # Only store the function name and parameters; do not check or call the factory here
        self.forcing_functions[input_name] = {
            'function': forcing_function_name,
            'args': args,
            'kwargs': kwargs
        }

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

class JaxModelEqx(eqx.Module):
    parameters: dict = eqx.field()
    Y0: dict = eqx.field()
    state_names: list = eqx.field()
    param_names: list = eqx.field()
    calc_names: list = eqx.field()
    inputs: list = eqx.field()
    model_tree: object = eqx.static_field()
    forcing_functions: dict = eqx.field()

    @staticmethod
    def from_model(model: str | Path):
        model_str = model.read_text() if isinstance(model, Path) else model
        parser = ModelParser()
        parsed_model = parser.parse(model_str)
        model_tree = parsed_model.model_copy()
        parameters = dict(model_tree.parameters)
        Y0 = dict(model_tree.Y0)
        state_names = list(Y0.keys())
        param_names = list(parameters.keys())
        calc_names = list(model_tree.dynamic_calcs.keys())
        inputs = list(parsed_model.inputs)
        return JaxModelEqx(
            parameters=parameters,
            Y0=Y0,
            state_names=state_names,
            param_names=param_names,
            calc_names=calc_names,
            inputs=inputs,
            model_tree=model_tree,
            forcing_functions={},
        )

    def update_constants(self, **parameters: float | int):
        missing = [key for key in parameters if key not in self.parameters]
        if missing:
            raise KeyError(f"Parameter(s) '{', '.join(missing)}' do not exist in the model tree.")
        new_params = self.parameters.copy()
        new_params.update(parameters)
        return eqx.tree_at(lambda m: m.parameters, self, replace=new_params)

    def update_Y0(self, **Y0: float | int):
        missing = [key for key in Y0 if key not in self.Y0]
        if missing:
            raise KeyError(f"Initial condition(s) '{', '.join(missing)}' do not exist in the model tree.")
        new_Y0 = self.Y0.copy()
        new_Y0.update(Y0)
        return eqx.tree_at(lambda m: m.Y0, self, replace=new_Y0)

    def assign_forcing_function(self, input_name, forcing_function_name, *args, **kwargs):
        """
        Assign a forcing function to a single input variable.
        This replaces the entire forcing_functions dict with a new one containing the updated function for the given input.
        """
        if input_name not in self.inputs:
            raise ValueError(f"'{input_name}' is not a valid input variable. Valid inputs: {self.inputs}")
        func_factory = getattr(self, forcing_function_name, None)
        if func_factory is None or not callable(func_factory):
            raise AttributeError(f"Forcing function '{forcing_function_name}' not found in JaxModelEqx.")
        # Only update the single input, replacing the entire dict
        new_ff = {**self.forcing_functions, input_name: func_factory(*args, **kwargs)}
        return eqx.tree_at(lambda m: m.forcing_functions, self, replace=new_ff)

    @staticmethod
    def OnOff(t, t0, t1, s=10.0):
        t = jnp.asarray(t)
        t0 = jnp.asarray(t0)
        t1 = jnp.asarray(t1)
        return (jnp.tanh(s * (t - t0)) - jnp.tanh(s * (t - t1))) / 2

    @staticmethod
    def PerDose(t0, duration, period, s=10.0):
        t0 = float(t0)
        duration = float(duration)
        period = float(period)
        def func(t):
            t = jnp.asarray(t)
            n = jnp.floor((t - t0) / period)
            start = t0 + n * period
            stop = start + duration
            return JaxModelEqx.OnOff(t, start, stop, s)
        return func

    @staticmethod
    def NDoses(t0_list, duration, s=10.0):
        t0_arr = jnp.array(t0_list)
        duration = float(duration)
        def func(t):
            t = jnp.asarray(t)
            return jnp.sum(JaxModelEqx.OnOff(t, t0_arr, t0_arr + duration, s), axis=-1)
        return func

    def _build_context_and_dydt(self, all_vars, t):
        context = {}
        idx = 0
        for name in self.state_names:
            context[name] = all_vars[idx]
            idx += 1
        for name in self.param_names:
            context[name] = all_vars[idx]
            idx += 1
        for name in self.calc_names:
            expr_calc = self.model_tree.dynamic_calcs[name]
            context[name] = expr_calc.evaluate(**context)
            idx += 1
        for input_name in self.inputs:
            if (self.forcing_functions is not None) and (input_name in self.forcing_functions):
                context[input_name] = self.forcing_functions[input_name](t)
            elif input_name in self.param_names:
                context[input_name] = context[input_name]
            else:
                context[input_name] = 0.0
        dydt = [self.model_tree.dynamics[state].evaluate(**context) for state in self.state_names]
        return context, dydt

    def model(self, t, y, param_vals):
        all_vars = jnp.concatenate([y, param_vals])
        _, dydt = self._build_context_and_dydt(all_vars, t)
        return jnp.stack(dydt)

    def run_model(self, times: Sequence):
        t0 = float(times[0])
        t_end = float(times[-1])
        y_init = jnp.asarray([self.Y0[state] for state in self.state_names], dtype=jnp.float32)
        param_vals = jnp.asarray([self.parameters[name] for name in self.param_names], dtype=jnp.float32)
        @jax.jit
        def ode_rhs(t, y, args):
            return self.model(t, y, args[0])
        ode_term = diffrax.ODETerm(ode_rhs)
        solver = diffrax.Dopri8()
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t_end, len(times)))
        sol = diffrax.diffeqsolve(
            ode_term, solver, t0=t0, t1=t_end, dt0=0.01, y0=y_init, saveat=saveat, args=(param_vals,)
        )
        #self.sol = sol
        @jax.jit
        def calc_dyn_single(state_vals, t):
            all_vars = jnp.concatenate([state_vals, param_vals])
            context, _ = self._build_context_and_dydt(all_vars, t)
            return jnp.array([context[cname] for cname in self.calc_names], dtype=jnp.float32)
        calc_dyn_jax = jax.vmap(calc_dyn_single, in_axes=(0, 0))(sol.ys, sol.ts)
        calc_dyn = np.asarray(calc_dyn_jax)
        return ComputedModel(
            times=np.asarray(sol.ts),
            states=np.asarray(sol.ys),
            var_names=self.state_names,
            aux_outputs=calc_dyn,
            aux_names=self.calc_names,
        )

class EqxModel(eqx.Module):
    
    parameters: dict = eqx.field() # Constants
    forcing_functions: dict = eqx.field() # Forcing functions for inputs
    Y0: dict = eqx.field() # State variable initial conditions
    model_tree: object = eqx.static_field()
    state_names: tuple = eqx.field()
    output_names: tuple = eqx.field()

    @staticmethod
    @jax.jit
    def OnOff(t, t0, t1, s=10.0):
        t = jnp.asarray(t)
        t0 = jnp.asarray(t0)
        t1 = jnp.asarray(t1)
        return (jnp.tanh(s * (t - t0)) - jnp.tanh(s * (t - t1))) / 2

    @staticmethod
    def PerDose(t0, duration, period, s=10.0):
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
        t0_arr = jnp.array(t0_list)
        duration = float(duration)
        @jax.jit
        def func(t):
            t = jnp.asarray(t)
            return jnp.sum(EqxModel.OnOff(t, t0_arr, t0_arr + duration, s), axis=-1)
        return func
    
    @staticmethod
    def ZeroFunc():
        """
        Default static method for forcing functions: always returns zero for any t input.
        """
        @jax.jit
        def func(t):
            return 0.0
        return func
    
    def compile_forcing_functions(self):
        """
        Convert all dict-based forcing functions to JIT-compiled callables in-place.
        Should be called before ODE solve if forcing_functions contains dicts.
        """
        for input_name, ff in list(self.forcing_functions.items()):
            if isinstance(ff, dict) and 'function' in ff:
                func_name = ff['function']
                args = ff.get('args', ())
                kwargs = ff.get('kwargs', {})
                func_factory = getattr(self, func_name, None)
                if func_factory is None or not callable(func_factory):
                    raise AttributeError(f"Forcing function '{func_name}' not found in EqxModel.")
                self.forcing_functions[input_name] = func_factory(*args, **kwargs)
            # else: already a callable, leave as is

    def build_context(self, state_vals, t):
        """
        Build the context dictionary for a given state vector and time.
        Includes state variables, parameters, forcing functions, and dynamic calcs.
        JAX-compatible (no side effects).
        """
        context = {name: state_vals[i] for i, name in enumerate(self.state_names)}
        context.update(self.parameters)
        for input_name, ff in self.forcing_functions.items():
            context[input_name] = ff(t)
        # Evaluate all dynamic calcs (needed for outputs or ODEs)
        for var, expr in self.model_tree.dynamic_calcs.items():
            context[var] = expr.evaluate(context, Approach.JAX)
        if hasattr(self.model_tree, 'calc_outputs'):
            for var, expr in self.model_tree.calc_outputs.items():
                context[var] = expr.evaluate(context, Approach.JAX)
        return context
    
    @eqx.filter_jit
    def model(self, t, y):
        context = self.build_context(y, t)
        dydt = [self.model_tree.dynamics[state].evaluate(context, Approach.JAX) for state in self.state_names]
        return jnp.stack(dydt)

    def run_model(self, times):
        # Compile forcing functions before running ODE solve
        self.compile_forcing_functions()
        t0 = float(times[0])
        t_end = float(times[-1])
        y_init = jnp.asarray([self.Y0[state] for state in self.state_names], dtype=jnp.float32)

        @eqx.filter_jit
        def ode_rhs(t, y, args):
            return self.model(t, y)

        ode_term = diffrax.ODETerm(ode_rhs)
        solver = diffrax.Dopri8()
        saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t_end, len(times)))
        sol = diffrax.diffeqsolve(
            ode_term, solver, t0=t0, t1=t_end, dt0=0.01, y0=y_init, saveat=saveat, args=()
        )

        @eqx.filter_jit
        def calc_outputs_single(state_vals, t):
            context = self.build_context(state_vals, t)
            return jnp.array([context[name] for name in self.output_names], dtype=jnp.float32)

        calc_outputs = jax.vmap(calc_outputs_single, in_axes=(0, 0))(sol.ys, sol.ts)
        # Return the compiled forcing functions for plotting
        return sol, calc_outputs, dict(self.forcing_functions)
        

class JaxModel(OdeModel):
    def __init__(self, model: str | Path):
        """
        Initialize a JaxModel from a model string or file, setting up variable indices for JAX evaluation.

        Args:
            model: Path to model file or model string.
        """
        super().__init__(model=model)

    
    def model(self, t: float, y, args) -> object:
        raise NotImplementedError("This method should be implemented in equinox module class.")

    def run_model(self, times: Sequence[int, float]) -> ComputedModel:
        """
        Solve the ODE system using diffrax (JAX backend) and return a ComputedModel, including calculated dynamics.

        Args:
            times: Sequence of time points at which to solve the ODE system.

        Returns:
            ComputedModel instance containing the solution.
        """
        eqx_model = self._to_eqx()
        sol, calc_outputs, input_functions = eqx_model.run_model(times)
        return ComputedModel(
            times=np.asarray(sol.ts),
            states=np.asarray(sol.ys),
            var_names=self.state_names,
            aux_outputs=np.asarray(calc_outputs),
            aux_names=self.outputs,
            input_functions=input_functions,
        )
    def _to_eqx(self):
        """
        Return an EqxModel object initialized from this JaxModel instance.
        The context is a copy of self.parameters.

        All parameter updates happen prior to creating the EqxModel instance.
        All Jax-based computation happens in EqxModel.
        """
        # Use tuples for state_names/output_names for JAX compatibility
        return EqxModel(
            parameters=self.parameters.copy(),
            forcing_functions=self.forcing_functions.copy(),
            Y0=self.Y0.copy(),
            model_tree=self.model_tree,
            state_names=tuple(self.state_names),
            output_names=tuple(self.outputs)
        )

class ScipyModel(OdeModel):
    def __init__(self, model: str | Path):
        """
        Initialize a ScipyModel from a model string or file.

        Args:
            model: Path to model or model string.
        """
        super().__init__(model=model)
    
    @staticmethod
    def OnOff(t, t0, t1):
        """
        t: current time
        t0: time (wrt to t) when dose is applied
        t1: time (wrt to t) when dose is stopped
        """
        s = 10
        y = (np.tanh(s*(t-t0)) - np.tanh(s*(t-t1)))/2
        return y

    @staticmethod
    def PerDose(t0, duration, period):
        """
        Returns a function of t for periodic dosing using OnOff, with parameters fixed.
        Usage: PerDose(t0, duration, period)(t)
        """
        def func(t):
            if t < t0:
                return 0.0
            n = int((t - t0) // period)
            start = t0 + n * period
            stop = start + duration
            return ScipyModel.OnOff(t, start, stop)
        return func
    
    @staticmethod
    def ZeroFunc():
        """
        Default static method for forcing functions: always returns zero for any t input.
        """
        def func(t):
            return 0.0
        return func

    @staticmethod
    def NDoses(t0_list, duration):
        """
        Returns a function of t for multiple dosing using OnOff, with parameters fixed.
        Usage: NDoses(t0_list, duration)(t)
        """
        def func(t):
            return sum(ScipyModel.OnOff(t, t0, t0 + duration) for t0 in t0_list)
        return func

    def build_context(self, state_vals, t):
        """
        Build the context dictionary for a given state vector and time.
        Includes state variables, parameters, forcing functions, and dynamic calcs/outputs.
        SciPy/NumPy compatible (not JAX).
        """
        context = {name: state_vals[i] for i, name in enumerate(self.state_names)}
        context.update(self.parameters)
        for input_name, ff in self.forcing_functions.items():
            if isinstance(ff, dict) and 'function' in ff:
                func_name = ff['function']
                args = ff.get('args', ())
                kwargs = ff.get('kwargs', {})
                func_factory = getattr(self, func_name, None)
                if func_factory is None or not callable(func_factory):
                    raise AttributeError(f"Forcing function '{func_name}' not found in ScipyModel.")
                func = func_factory(*args, **kwargs)
                context[input_name] = func(t)
            else:
                # fallback for legacy or direct function (should not occur with new logic)
                context[input_name] = ff(t)
        # Evaluate all dynamic calcs (needed for outputs or ODEs)
        for var, expr in self.model_tree.dynamic_calcs.items():
            context[var] = expr.evaluate(context, Approach.SCIPY)
        # Evaluate any calculated outputs if needed
        if hasattr(self.model_tree, 'calc_outputs'):
            for var, expr in self.model_tree.calc_outputs.items():
                context[var] = expr.evaluate(context, Approach.SCIPY)
        return context

    def model(self, t: float, y: np.ndarray, args: None = None) -> np.ndarray:
        """
        ODE right-hand side function for use with scipy.integrate.solve_ivp.
        Computes the time derivatives for the system of ODEs using the current state and parameters.
        """
        context = self.build_context(y, t)
        dydt = []
        for state in self.state_names:
            expr = self.model_tree.dynamics[state]
            val = expr.evaluate(context, Approach.SCIPY)
            dydt.append(val)
        return np.array(dydt)

    def run_model(self, times: Sequence[int, float]) -> ComputedModel:
        """
        Solve the ODE system using scipy.integrate.solve_ivp and return a ComputedModel, including calculated outputs.
        """
        times = np.array(times)
        y_init = np.array([self.Y0[state] for state in self.state_names])
        t_span = np.array([times[0], times[-1]])
        sol = sci.solve_ivp(
            fun=self.model,
            t_span=t_span,
            y0=y_init,
            t_eval=times,
            method='BDF'
        )
        self.sol = sol  # Store the raw solution with ScipyModel

        # Vectorized calculation of outputs (from self.outputs) for each time point
        output_names = self.outputs

        def calc_outputs_single(state_vals, t):
            context = self.build_context(state_vals, t)
            return np.array([context[name] for name in output_names], dtype=np.float64)

        # Use numpy vectorization for speed (not jax.vmap, since this is numpy/scipy)
        calc_outputs = np.stack([
            calc_outputs_single(sol.y[:, i], sol.t[i])
            for i in range(sol.t.shape[0])
        ], axis=0)

        # Build input_functions dict: input name -> callable
        input_functions = {}
        for input_name, ff in self.forcing_functions.items():
            if isinstance(ff, dict) and 'function' in ff:
                func_name = ff['function']
                args = ff.get('args', ())
                kwargs = ff.get('kwargs', {})
                func_factory = getattr(self, func_name, None)
                if func_factory is None or not callable(func_factory):
                    raise AttributeError(f"Forcing function '{func_name}' not found in ScipyModel.")
                input_functions[input_name] = func_factory(*args, **kwargs)
            else:
                input_functions[input_name] = ff  # already a callable

        return ComputedModel(
            times=sol.t,
            states=sol.y.T,  # shape (n_times, n_states)
            var_names=self.state_names,
            aux_outputs=calc_outputs,
            aux_names=output_names,
            input_functions=input_functions,
        )
