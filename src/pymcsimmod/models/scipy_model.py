"""ScipyModel implementation for ODE solving using scipy.integrate."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy.integrate as sci

from ..config import BackendType
from ..forcing.unified import UnifiedForcingFactory
from ..model import Approach
from ..utils.context import build_evaluation_context
from .base import OdeModel
from .computed import ComputedModel


class ScipyModel(OdeModel):
    """ODE model implementation using scipy.integrate.solve_ivp."""

    backend = BackendType.SCIPY

    def __init__(self, model: str | Path):
        """
        Initialize a ScipyModel from a model string or file.

        Args:
            model: Path to model or model string.
        """
        super().__init__(model=model)
        # Cache for compiled forcing functions (populated in run_model)
        self._compiled_forcing: dict[str, Callable[[float], float]] = {}

    def _get_approach(self) -> Approach:
        """Get the evaluation approach for ScipyModel."""
        return Approach.SCIPY

    def _compile_forcing_functions(self) -> dict[str, Callable[[float], float]]:
        """
        Compile all forcing functions once per simulation run.

        Returns:
            Dictionary mapping input name to compiled callable.
        """
        compiled: dict[str, Callable[[float], float]] = {}
        for input_name, ff in self.forcing_functions.items():
            if isinstance(ff, dict) and "function" in ff:
                func_name = ff["function"]
                args = ff.get("args", ())
                kwargs = ff.get("kwargs", {})
                compiled[input_name] = UnifiedForcingFactory.create_forcing_function(
                    func_name, backend=self.backend, args=args, **kwargs
                )
            else:  # pragma: no cover
                # Direct callable (fallback for compatibility)
                compiled[input_name] = ff
        return compiled

    def build_context(self, state_vals: np.ndarray, t: float) -> dict[str, Any]:
        """
        Build the context dictionary for a given state vector and time.

        Uses pre-compiled forcing functions for performance (populated by run_model).
        Falls back to compiling on-the-fly if called outside of run_model.

        Args:
            state_vals: Array of state variable values.
            t: Current time.

        Returns:
            Dictionary containing all variables for expression evaluation.
        """
        # Use cached forcing functions (compiled once in run_model for performance).
        # Fall back to compiling on-the-fly if called outside run_model.
        forcing_funcs = self._compiled_forcing or self._compile_forcing_functions()
        forcing_values = {name: ff(t) for name, ff in forcing_funcs.items()}

        context = build_evaluation_context(
            state_vals=state_vals,
            state_names=self.state_names,
            parameters=self.parameters,
            forcing_values=forcing_values,
            dynamic_calcs=self.model_tree.dynamic_calcs,
            approach=Approach.SCIPY,
        )

        # Add calculated outputs if needed
        if hasattr(self.model_tree, "calc_outputs"):
            for var, expr in self.model_tree.calc_outputs.items():
                context[var] = expr.evaluate(context, Approach.SCIPY)

        return context

    def model(self, t: float, y: np.ndarray, args: Any = None) -> np.ndarray:
        """
        ODE right-hand side function for use with scipy.integrate.solve_ivp.

        Args:
            t: Current time.
            y: Current state vector.
            args: Additional arguments (unused).

        Returns:
            Array of time derivatives for each state variable.
        """
        context = self.build_context(y, t)
        dydt = []
        for state in self.state_names:
            expr = self.model_tree.dynamics[state]
            val = expr.evaluate(context, Approach.SCIPY)
            dydt.append(val)
        return np.array(dydt)

    def make_event_switch(self, time: float) -> Callable[[float, np.ndarray], float]:
        """
        Create an event function for scipy.integrate that triggers at a specific time.

        Args:
            time: Time at which the event should trigger.

        Returns:
            Event function for scipy.integrate.
        """

        def event(t: float, y: np.ndarray) -> float:
            return t - time

        event.terminal = False
        event.direction = 0
        return event

    def run_model(self, times: Sequence[int | float], method: str = "BDF") -> ComputedModel:
        """
        Solve the ODE system using scipy.integrate.solve_ivp and return a ComputedModel.

        Forcing functions are compiled once at the start of each run for performance.
        Handles discrete events by integrating between event times and applying events.

        Args:
            times: Sequence of time points to solve at.
            method: Integration method for solve_ivp. Default is 'BDF'.
                   See scipy.integrate.solve_ivp documentation for available methods:
                   'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'.

        Returns:
            ComputedModel containing the solution.
        """
        times = np.array(times)
        self._evaluate_event_schedulers(times[0], times[-1])
        y_init = np.array([self.Y0[state] for state in self.state_names])

        # Compile forcing functions once for the duration of this run
        self._compiled_forcing = self._compile_forcing_functions()

        # Disable vectorized mode completely to avoid scipy numerical issues
        use_vectorized = False

        # Get all switch times (forcing functions + events)
        switch_times = self.extract_switch_times(self.forcing_functions, times[0], times[-1])

        # If no events, use the original method
        if not self.events:
            t_span = np.array([times[0], times[-1]])
            all_times = np.unique(np.concatenate([np.asarray(times), np.asarray(switch_times)]))
            events = [self.make_event_switch(t) for t in switch_times]

            # Handle edge case where t_span has identical start and end times
            if t_span[0] == t_span[1] and times[0] == 0.0:
                class MockSolutionMinimal:
                    def __init__(self, t, y_init):
                        self.t = np.array([t])
                        self.y = y_init.reshape(-1, 1)

                sol = MockSolutionMinimal(times[0], y_init)
            elif t_span[0] == t_span[1]:
                t_span_corrected = [0.0, times[0]] if times[0] > 0.0 else [times[0], 0.0]
                sol = sci.solve_ivp(
                    fun=self.model,
                    t_span=t_span_corrected,
                    y0=y_init,
                    t_eval=times,
                    vectorized=use_vectorized,
                    method=method,
                )
                sol.t = np.asarray(sol.t)
                sol.y = np.asarray(sol.y)
            else:
                sol = sci.solve_ivp(
                    fun=self.model,
                    t_span=t_span,
                    y0=y_init,
                    t_eval=all_times,
                    vectorized=use_vectorized,
                    events=events,
                    method=method,
                )
                sol.t = np.asarray(sol.t)
                sol.y = np.asarray(sol.y)
        else:
            # Handle discrete events using deSolve-inspired approach
            from ..events.utils import apply_events_at_time, check_events

            # Validate events and potentially modify output times
            validated_events, modified_times = check_events(self.events, times, self.state_names)

            if not np.array_equal(times, modified_times):
                times = modified_times

            # Get event times within simulation range
            event_times = [e.time for e in validated_events if times[0] <= e.time <= times[-1]]

            # Apply events at start time if any
            current_y = y_init.copy()
            if event_times and abs(event_times[0] - times[0]) < 1e-12:
                current_y = np.array(
                    [
                        apply_events_at_time(
                            times[0],
                            {name: current_y[i] for i, name in enumerate(self.state_names)},
                            validated_events,
                        )[name]
                        for name in self.state_names
                    ]
                )

            # Create segments between event times
            segments = []
            t_start = times[0]

            for event_time in sorted(event_times):
                if event_time > t_start:
                    segments.append((t_start, event_time))
                    t_start = event_time

            # Add final segment
            if t_start < times[-1]:
                segments.append((t_start, times[-1]))

            # Integrate each segment
            all_sol_times = []
            all_sol_states = []

            for i, (seg_start, seg_end) in enumerate(segments):
                seg_times = times[(times >= seg_start) & (times <= seg_end)]
                seg_switch_times = [t for t in switch_times if seg_start < t < seg_end]
                seg_all_times = np.unique(np.concatenate([seg_times, seg_switch_times]))

                if len(seg_all_times) > 0 and seg_all_times[0] != seg_start:
                    seg_all_times = np.concatenate([[seg_start], seg_all_times])
                if len(seg_all_times) > 0 and seg_all_times[-1] != seg_end:
                    seg_all_times = np.concatenate([seg_all_times, [seg_end]])

                if len(seg_all_times) > 1:
                    seg_events = [self.make_event_switch(t) for t in seg_switch_times]

                    seg_sol = sci.solve_ivp(
                        fun=self.model,
                        t_span=[seg_start, seg_end],
                        y0=current_y,
                        t_eval=seg_all_times,
                        vectorized=use_vectorized,
                        events=seg_events,
                        method=method,
                    )

                    all_sol_times.append(seg_sol.t)
                    all_sol_states.append(seg_sol.y)
                    current_y = seg_sol.y[:, -1]

                # Apply events at segment end time if any
                if seg_end in event_times:
                    state_dict = {name: current_y[i] for i, name in enumerate(self.state_names)}
                    state_dict = apply_events_at_time(seg_end, state_dict, validated_events)
                    current_y = np.array([state_dict[name] for name in self.state_names])

            # Combine all solutions
            if all_sol_times:
                combined_times = np.concatenate(all_sol_times)
                combined_states = np.concatenate(all_sol_states, axis=1)

                # Remove duplicate time points from segment boundaries
                unique_indices = np.unique(combined_times, return_index=True)[1]
                combined_times = combined_times[unique_indices]
                combined_states = combined_states[:, unique_indices]

                class MockSolution:
                    def __init__(self, t, y):
                        self.t = np.asarray(t)
                        self.y = np.asarray(y)

                sol = MockSolution(combined_times, combined_states)
            else:
                raise RuntimeError(
                    "No segments could be integrated. This indicates an issue with event "
                    "timing or segment creation logic. Please check your discrete events "
                    "and evaluation time points."
                )

        self.sol = sol

        # Vectorized calculation of outputs for each time point
        output_names = self.outputs

        def calc_outputs_single(state_vals: np.ndarray, t: float) -> np.ndarray:
            context = self.build_context(state_vals, t)
            return np.array([context[name] for name in output_names], dtype=np.float64)

        calc_outputs = np.stack(
            [calc_outputs_single(sol.y[:, i], sol.t[i]) for i in range(sol.t.shape[0])], axis=0
        )

        # The input_functions in ComputedModel use the already-compiled forcing functions
        return ComputedModel(
            times=sol.t,
            states=sol.y.T,  # shape (n_times, n_states)
            var_names=self.state_names,
            aux_outputs=calc_outputs,
            aux_names=output_names,
            input_functions=self._compiled_forcing,
        )


__all__ = ["ScipyModel"]
