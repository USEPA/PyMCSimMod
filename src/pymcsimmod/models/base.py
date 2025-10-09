"""Abstract base class for ODE models."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from ..parser import ModelParser
from .computed import ComputedModel
from .events import DiscreteEvent


class OdeModel(ABC):
    """Abstract base class for ODE models."""
    
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
        # Initialize events list
        self.events = []

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

    def add_event(self, time: float, state_var: str, value: float, method: Literal['replace', 'add', 'multiply'] = 'add') -> None:
        """
        Add a discrete event to occur at a specific time.
        
        Args:
            time: Time at which the event occurs.
            state_var: Name of the state variable to modify.
            value: Value to use in the operation.
            method: Type of operation ('replace', 'add', 'multiply').
            
        Raises:
            KeyError: If state_var is not a valid state variable.
        """
        if state_var not in self.state_names:
            raise KeyError(f"State variable '{state_var}' not found. Valid state variables: {self.state_names}")
        
        event = DiscreteEvent(time=time, state_var=state_var, value=value, method=method)
        self.events.append(event)
        # Keep events sorted by time for efficient processing
        self.events.sort(key=lambda e: e.time)

    def clear_events(self) -> None:
        """Clear all discrete events."""
        self.events = []

    def get_event_times(self, t_start: float, t_end: float) -> list[float]:
        """
        Get all event times within the specified time range.
        
        Args:
            t_start: Start time.
            t_end: End time.
            
        Returns:
            List of event times within [t_start, t_end].
        """
        return [event.time for event in self.events if t_start <= event.time <= t_end]

    def apply_events_at_time(self, t: float, state_dict: dict[str, float]) -> dict[str, float]:
        """
        Apply all events that occur at time t.
        
        Args:
            t: Current time.
            state_dict: Current state as a dictionary.
            
        Returns:
            Updated state dictionary after applying events.
        """
        # Apply events in order (already sorted by time)
        for event in self.events:
            if abs(event.time - t) < 1e-12:  # Use small tolerance for floating point comparison
                state_dict = event.apply(state_dict, self.state_names)
        return state_dict

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

    def extract_switch_times(self, forcing_functions, t_start, t_end):
        """
        Extract switching times from forcing functions and discrete events.
        
        Args:
            forcing_functions: Dictionary of forcing function specifications.
            t_start: Start time for extraction.
            t_end: End time for extraction.
            
        Returns:
            Sorted list of switch times within [t_start, t_end].
        """
        switch_times = set()
        
        # Add forcing function switch times
        for ff in forcing_functions.values():
            if isinstance(ff, dict) and 'function' in ff:
                func = ff['function']
                kwargs = ff.get('kwargs', {})
                if func == 'PerDose':
                    t0 = kwargs['t0']
                    duration = kwargs['duration']
                    period = kwargs['period']
                    n = 0
                    while True:
                        on = t0 + n * period
                        off = on + duration
                        if on > t_end:
                            break
                        if on >= t_start:
                            switch_times.add(on)
                        if off >= t_start and off <= t_end:
                            switch_times.add(off)
                        n += 1
                elif func == 'NDoses':
                    t0_list = kwargs['t0_list']
                    duration = kwargs['duration']
                    for t0 in t0_list:
                        on = t0
                        off = t0 + duration
                        if on >= t_start and on <= t_end:
                            switch_times.add(on)
                        if off >= t_start and off <= t_end:
                            switch_times.add(off)
                elif func == 'OnOff':
                    t0 = kwargs['t0']
                    t1 = kwargs['t1']
                    if t0 >= t_start and t0 <= t_end:
                        switch_times.add(t0)
                    if t1 >= t_start and t1 <= t_end:
                        switch_times.add(t1)
        
        # Add event times
        event_times = self.get_event_times(t_start, t_end)
        switch_times.update(event_times)
        
        return sorted(switch_times)

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
            times: Sequence of time points to solve at.

        Returns:
            ComputedModel containing the solution.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")


__all__ = ["OdeModel"]