"""Abstract base class for ODE models."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

try:
    import pandas as pd
except ImportError:
    pd = None

from ..config import BackendType
from ..model import Approach, InitializeSection
from ..parser import ModelParser
from .computed import ComputedModel
from .events import DiscreteEvent


class OdeModel(ABC):
    """Abstract base class for ODE models."""

    # Backend type - to be set by subclasses
    backend: BackendType

    def __init__(self, model: str | Path):
        """
        Load and parse a model from a file path or string, initializing parameters and initial conditions.

        Args:
            model: Path to model file (as string or Path object) or model content as string.
                  If a string that represents a valid file path, it will be treated as a file path.
                  Otherwise, it will be treated as model content.
        """
        model_str = self._resolve_model_input(model)

        parser = ModelParser()
        parsed_model = parser.parse(model_str)
        if parsed_model is None:
            raise ValueError("Failed to parse model. Please check model syntax.")
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
            input_name: {"function": "ZeroFunc", "args": (), "kwargs": {}}
            for input_name in self.inputs
        }
        # Initialize events list
        self.events = []
        self._manual_events = []
        self._event_schedulers = {}

    def _resolve_model_input(self, model: str | Path) -> str:
        """
        Resolve model input to string content, handling both file paths and direct content.

        Args:
            model: Path to model file (as string or Path object) or model content as string.

        Returns:
            Model content as string.

        Raises:
            TypeError: If model is not a string or Path object.
        """
        if isinstance(model, Path):
            return model.read_text()

        if isinstance(model, str):
            # Quick check: if string has newlines or is very long, treat as model content
            # Can not run Path(model).exists() on very long stings
            if "\n" in model or len(model) > 500:
                return model

            # Otherwise, check if it's a file path
            potential_path = Path(model)
            if potential_path.exists() and potential_path.is_file():
                return potential_path.read_text()
            return model  # Treat as model content

        raise TypeError(f"model must be a string or Path object, got {type(model)}")

    def _init_parameters(self) -> None:
        """
        Assign the parameters and initial conditions (Y0) from the model tree to the model instance.
        """
        self.parameters = self.model_tree.parameters.copy()

        # Add calculated parameters from Initialize section
        self._extract_calculated_parameters()

        self.Y0 = self._evaluate_Y0()  # dict(state_var_name: value)

    def _extract_calculated_parameters(self) -> None:
        """
        Extract calculated parameters from the Initialize section.
        These are non-state variables that are calculated and should be added to parameters.
        """
        # Get the list of state variable names
        state_names = self.model_tree.states

        # Extract calculated parameters from Initialize section
        context = self.parameters.copy()  # Use current parameters as evaluation context
        approach = self._get_approach()  # Get the appropriate approach from subclass

        for section in self.model_tree.sections:
            if isinstance(section, InitializeSection):
                for statement in section.statements:
                    var_name = (
                        statement.lhs.eval()
                        if hasattr(statement.lhs, "eval")
                        else statement.lhs.name
                    )
                    # If this variable is not a state variable, it's a calculated parameter
                    if var_name not in state_names:
                        calculated_value = statement.rhs.evaluate(context, approach)
                        self.parameters[var_name] = calculated_value
                        # Update context for subsequent calculations that might depend on this
                        context[var_name] = calculated_value

    @abstractmethod
    def _get_approach(self) -> Approach:
        """
        Get the evaluation approach for this model implementation.

        Returns:
            Approach enum indicating whether to use JAX or SCIPY evaluation.
        """
        raise NotImplementedError("Subclasses must implement _get_approach()")

    def _evaluate_Y0(self) -> dict[str, float | int]:
        """
        Evaluate initial conditions from the model tree using current parameter values as context.
        Only extracts state variables, not calculated parameters.

        Returns:
            Dictionary where keys are state variable names and values are their evaluated initial values.
        """
        context = self.parameters.copy()  # Use current parameters as evaluation context
        approach = self._get_approach()  # Get the appropriate approach from subclass

        # Get the list of state variable names
        state_names = self.model_tree.states

        Y0 = {}
        for section in self.model_tree.sections:
            if isinstance(section, InitializeSection):
                for statement in section.statements:
                    var_name = (
                        statement.lhs.eval()
                        if hasattr(statement.lhs, "eval")
                        else statement.lhs.name
                    )
                    # Only include state variables in Y0
                    if var_name in state_names:
                        Y0[var_name] = statement.rhs.evaluate(context, approach)
        return Y0

    def update_constants(self, reset_to_defaults: bool = False, **parameters: float | int) -> None:
        """
        Update any constants in the model tree in place. Re-evaluate calculated parameters and Y0 after updating.

        Args:
            reset_to_defaults: If True, reset all parameters to their original model tree defaults
                             before applying updates. If False, update current parameters in place.
            **parameters: Keyword arguments where keys are parameter names and values are the new values.

        Raises:
            KeyError: If a parameter name does not exist in the model tree.
        """
        missing = [key for key in parameters if key not in self.model_tree.parameters]
        if missing:
            raise KeyError(f"Parameter(s) '{', '.join(missing)}' do not exist in the model tree.")

        if reset_to_defaults:
            # Reset to original model tree defaults first
            self.parameters = self.model_tree.parameters.copy()
            # Re-extract calculated parameters with default values
            self._extract_calculated_parameters()
            self.Y0 = self._evaluate_Y0()  # dict(state_var_name: value)

        # Apply the provided parameter updates
        for key, value in parameters.items():
            self.parameters[key] = value

        # Re-extract calculated parameters in case any depend on updated parameters
        self._extract_calculated_parameters()
        # Re-evaluate Y0 in case any initial conditions depend on updated parameters
        self.Y0 = self._evaluate_Y0()

    def update_Y0(self, reset_to_defaults: bool = False, **Y0: float | int) -> None:
        """
        Update any initial conditions in the model tree in place.

        This method updates Y0 values directly and then re-evaluates all parameter-dependent
        initial conditions to ensure consistency.

        Args:
            reset_to_defaults: If True, reset all Y0 values to their original model tree defaults
                             before applying updates. If False, update current Y0 values in place.
            **Y0: Keyword arguments where keys are state variable names and values are the new initial values.

        Raises:
            KeyError: If a state variable name does not exist in the model tree.
        """
        # Check against state variables, not current Y0 keys
        state_names = self.model_tree.states
        missing = [key for key in Y0 if key not in state_names]
        if missing:
            raise KeyError(
                f"Initial condition(s) '{', '.join(missing)}' do not exist in the model tree."
            )

        if reset_to_defaults:
            # Reset to original model tree defaults first
            self.Y0 = self._evaluate_Y0()

        # Update the specified Y0 values
        for key, value in Y0.items():
            self.Y0[key] = value

        # Re-evaluate all Y0 values to handle parameter dependencies
        # This ensures that if any Y0 depends on parameters, they get updated correctly
        evaluated_Y0 = self._evaluate_Y0()

        # Keep user-specified values but update parameter-dependent ones
        for key, value in evaluated_Y0.items():
            if key not in Y0:  # Only update values not explicitly set by user
                self.Y0[key] = value

    def add_event(
        self,
        time: float,
        state_var: str,
        value: float,
        method: Literal["replace", "add", "multiply"] = "add",
    ) -> None:
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
            raise KeyError(
                f"State variable '{state_var}' not found. Valid state variables: {self.state_names}"
            )

        if not hasattr(self, "_manual_events"):
            self._manual_events = []

        event = DiscreteEvent(time=time, state_var=state_var, value=value, method=method)
        self._manual_events.append(event)
        self.events = list(self._manual_events)
        self.events.sort(key=lambda e: e.time)

    def clear_events(self) -> None:
        """Clear all discrete events and event schedulers."""
        self.events = []
        self._manual_events = []
        if hasattr(self, "_event_schedulers"):
            self._event_schedulers.clear()

    def assign_event(self, state_var: str, event_type=None, *args, **kwargs) -> None:
        """
        Assign an event scheduler to a state variable.

        Args:
            state_var: Name of the state variable to modify.
            event_type: Type of event scheduler ('OnOff', 'PerDoses', 'NDoses', 'DataFrame', etc.) or an instance of BaseEventScheduler.
            *args: Positional arguments for the event scheduler.
            **kwargs: Keyword arguments for the event scheduler.
        """
        if state_var not in self.state_names:
            raise KeyError(
                f"State variable '{state_var}' not found. Valid state variables: {self.state_names}"
            )

        if not hasattr(self, "_event_schedulers"):
            self._event_schedulers = {}

        from ..events.base import BaseEventScheduler
        if isinstance(event_type, BaseEventScheduler):
            scheduler = event_type
        else:
            from ..events.base import create_event_scheduler
            scheduler = create_event_scheduler(event_type, *args, **kwargs)

        self._event_schedulers[state_var] = [scheduler]

    def add_event_scheduler(self, state_var: str, scheduler) -> None:
        """
        Add an event scheduler to a state variable without removing existing ones.

        Args:
            state_var: Name of the state variable to modify.
            scheduler: An instance of BaseEventScheduler.
        """
        if state_var not in self.state_names:
            raise KeyError(
                f"State variable '{state_var}' not found. Valid state variables: {self.state_names}"
            )

        if not hasattr(self, "_event_schedulers"):
            self._event_schedulers = {}

        self._event_schedulers.setdefault(state_var, []).append(scheduler)

    def _evaluate_event_schedulers(self, t_start: float, t_end: float) -> None:
        """Evaluate all event schedulers and populate self.events."""
        if not hasattr(self, "_manual_events"):
            self._manual_events = []

        sched_events = []
        if hasattr(self, "_event_schedulers"):
            for state_var, schedulers in self._event_schedulers.items():
                for scheduler in schedulers:
                    sched_events.extend(scheduler.get_events(state_var, t_start, t_end))

        self.events = list(self._manual_events) + sched_events
        self.events.sort(key=lambda e: e.time)

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

    def assign_forcing_function(self, input_name, forcing_function_name=None, *args, **kwargs):
        """
        Assign a forcing function to an input variable with consistent API.

        Standard API:
            model.assign_forcing_function(var, forcing_type, **kwargs)

        Examples:
            # Traditional forcing functions
            model.assign_forcing_function('input', 'PerDose', t0=0, duration=1, period=24)
            model.assign_forcing_function('input', 'NDoses', t0_list=[0,24,48], duration=1)
            model.assign_forcing_function('input', 'OnOff', t0=0, t1=10)

            # Interpolated forcing
            model.assign_forcing_function('input', 'Interpolate', times=[0,1,2], values=[10,20,30])
            model.assign_forcing_function('input', 'Interpolate', dataframe=df)
            model.assign_forcing_function('input', 'Interpolate', data_dict={'time': [0,1,2], 'value': [10,20,30]})

        Legacy support (deprecated):
            - DataFrame/dict as first argument for multi-variable interpolation
            - Times array as second argument for backward compatibility

        Args:
            input_name: Name of input variable OR dictionary/DataFrame (legacy)
            forcing_function_name: Name of forcing function ('PerDose', 'NDoses', 'OnOff', 'Interpolate', etc.)
            *args: Positional arguments (legacy support)
            **kwargs: Parameters for the forcing function

        Raises:
            ValueError: If input validation fails or invalid parameters provided
        """
        if not hasattr(self, "forcing_functions"):
            self.forcing_functions = {}

        # Handle unsupported DataFrame multi-variable interpolation
        if pd is not None and isinstance(input_name, pd.DataFrame):
            raise ValueError(
                "Passing DataFrame as first argument is not supported. Use: "
                "assign_forcing_function(var, 'Interpolate', dataframe=df) for each variable"
            )

        # Handle unsupported dictionary multi-variable interpolation
        elif isinstance(input_name, dict):
            raise ValueError(
                "Passing dictionary as first argument is not supported. Use: "
                "assign_forcing_function(var, 'Interpolate', data_dict=data) for each variable"
            )

        # Validate input variable name
        if input_name not in self.inputs:
            raise ValueError(
                f"'{input_name}' is not a valid input variable. Valid inputs: {self.inputs}"
            )

        # Handle unsupported times/values patterns
        if (
            forcing_function_name is not None
            and isinstance(forcing_function_name, list | tuple)
            and "values" in kwargs
        ):
            raise ValueError(
                "Passing times as second argument is not supported. Use: "
                "assign_forcing_function(var, 'Interpolate', times=times, values=values)"
            )

        # Handle unsupported times in kwargs with variable name
        elif "times" in kwargs and input_name in kwargs:
            raise ValueError(
                "Using variable name in kwargs is not supported. Use: "
                "assign_forcing_function(var, 'Interpolate', times=times, values=values)"
            )

        # Require forcing function name
        if forcing_function_name is None:
            raise ValueError(
                "Forcing function type must be specified. Available types: "
                "'OnOff', 'PerDose', 'NDoses', 'Interpolate', 'ZeroFunc', 'ConstFunc'"
            )

        # Store the forcing function specification
        self.forcing_functions[input_name] = {
            "function": forcing_function_name,
            "args": args,
            "kwargs": kwargs,
        }

    def _create_forcing_functions_for_backend(self, backend: BackendType = None):
        """
        Create actual forcing functions for the specified backend from stored specifications.

        This method is compatible with both ScipyModel and JAXModel implementations:
        - ScipyModel: Can use this for batch compilation or compile on-demand
        - JAXModel: Can use this in EqxModel.compile_forcing_functions() for batch compilation

        Args:
            backend: Backend type to use for creating forcing functions.
                    If None, uses the model's default backend.

        Returns:
            Dictionary mapping input names to compiled forcing functions.
            For JAX backend, all functions are JIT-compiled and ready for use.
        """
        from ..forcing.unified import UnifiedForcingFactory

        if backend is None:
            backend = self.backend if hasattr(self, "backend") else BackendType.SCIPY

        compiled_functions = {}

        for input_name, ff_spec in self.forcing_functions.items():
            # Use duck typing approach compatible with JAX equinox modules
            # Check for dict-like interface (same as JAXModel does)
            if (
                hasattr(ff_spec, "get")
                and hasattr(ff_spec, "__getitem__")
                and "function" in ff_spec
            ):
                func_name = ff_spec["function"]
                args = ff_spec.get("args", ())
                kwargs = ff_spec.get("kwargs", {})

                # Create backend-specific forcing function
                # For JAX: will be JIT-compiled and ready for use
                # For SciPy: will be a regular callable
                compiled_functions[input_name] = UnifiedForcingFactory.create_forcing_function(
                    func_name, backend, args=args, **kwargs
                )
            else:
                # Already a compiled function or other callable
                # This handles both legacy format and pre-compiled functions
                compiled_functions[input_name] = ff_spec

        return compiled_functions

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
            if isinstance(ff, dict) and "function" in ff:
                func = ff["function"]
                kwargs = ff.get("kwargs", {})
                if func == "PerDose":
                    t0 = kwargs["t0"]
                    duration = kwargs["duration"]
                    period = kwargs["period"]
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
                elif func == "NDoses":
                    t0_list = kwargs["t0_list"]
                    duration = kwargs["duration"]
                    for t0 in t0_list:
                        on = t0
                        off = t0 + duration
                        if on >= t_start and on <= t_end:
                            switch_times.add(on)
                        if off >= t_start and off <= t_end:
                            switch_times.add(off)
                elif func == "OnOff":
                    t0 = kwargs["t0"]
                    t1 = kwargs["t1"]
                    if t0 >= t_start and t0 <= t_end:
                        switch_times.add(t0)
                    if t1 >= t_start and t1 <= t_end:
                        switch_times.add(t1)
                elif func in ["InterpolatedForcing", "Interpolate"]:
                    # For interpolated forcing, extract times from different data formats
                    times_data = []

                    # Check args for times array (legacy format)
                    if len(ff.get("args", [])) > 0:
                        times_data = ff["args"][0]
                    # Check kwargs for various time data formats
                    elif "data_dict" in kwargs and "time" in kwargs["data_dict"]:
                        times_data = kwargs["data_dict"]["time"]
                    elif "times" in kwargs:
                        times_data = kwargs["times"]
                    elif "dataframe" in kwargs:
                        df = kwargs["dataframe"]
                        time_col = kwargs.get("time_col", "time")
                        if hasattr(df, "columns") and time_col in df.columns:
                            times_data = df[time_col].tolist()

                    for t in times_data:
                        if t >= t_start and t <= t_end:
                            switch_times.add(t)

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
