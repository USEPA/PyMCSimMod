"""ComputedModel class for ODE solution results."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pydantic import BaseModel

from pymcsimmod.extra_typing import NumericArray


class ComputedModel(BaseModel):
    """
    Adapter class for ODE solution results from either JaxModel (diffrax) or ScipyModel (solve_ivp).
    Provides a unified interface for accessing time, state, and plotting results.

    Attributes:
        times: Array of time points at which the solution is evaluated.
        states: Array of state variable values at each time point (shape: [n_times, n_states]).
        var_names: List of state variable names (order matches columns of states).
        aux_outputs: Array of calculated dynamic variable values at each time point (shape: [n_times, n_calcs]).
        aux_names: List of calculated dynamic variable names (order matches columns of aux_outputs).
        input_functions: Maps input name to callable.
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
        labels: dict[str, str] | None = None,
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
            labels: Optional dict mapping variable names to custom legend labels.
                   If not provided, variable names are used as labels.
            **kwargs: Additional keyword arguments passed to plt.plot.

        Returns:
            The matplotlib axis object containing the plot.
        """
        if ax is None:
            _, ax = plt.subplots()
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
            # Use custom label if provided, otherwise use variable name
            label = labels.get(var, var) if labels is not None else var

            if var in self.var_names:
                idx = self.var_names.index(var)
                ax.plot(self.times, self.states[:, idx], label=label, **kwargs)
            elif self.aux_names is not None and var in self.aux_names:
                idx = self.aux_names.index(var)
                ax.plot(self.times, self.aux_outputs[:, idx], label=label, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend(loc='best')
        return ax

    def plot_inputs(
        self,
        variables: str | list[str] | None = None,
        ax: Axes | None = None,
        legend: bool = True,
        xlabel: str = "Time",
        ylabel: str = "Input Value",
        labels: dict[str, str] | None = None,
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
            labels: Optional dict mapping input names to custom legend labels.
                   If not provided, input names are used as labels.
            **kwargs: Additional keyword arguments passed to plt.plot.

        Returns:
            The matplotlib axis object containing the plot.
        """
        if self.input_functions is None:
            raise AttributeError("No input functions stored in this ComputedModel.")
        if ax is None:
            _, ax = plt.subplots()
        if variables is None:
            variables = list(self.input_functions.keys())
        if isinstance(variables, str):
            variables = [variables]
        for var in variables:
            if var not in self.input_functions:
                raise KeyError(f"Input '{var}' not found in input_functions.")
            # Use custom label if provided, otherwise use variable name
            label = labels.get(var, var) if labels is not None else var
            # Evaluate input function at all time points
            values = [self.input_functions[var](t) for t in self.times]
            ax.plot(self.times, values, label=label, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend(loc='best')
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


__all__ = ["ComputedModel"]
