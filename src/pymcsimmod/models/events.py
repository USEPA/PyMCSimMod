"""Discrete event handling for ODE models."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class DiscreteEvent(BaseModel):
    """
    Represents a discrete event similar to deSolve's event handling.

    Attributes:
        time: Time at which the event occurs.
        state_var: Name of the state variable to modify.
        value: Value to use in the operation.
        method: Type of operation ('replace', 'add', 'multiply').
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    time: float
    state_var: str
    value: Any
    method: Literal["replace", "add", "multiply"] = "add"

    def __hash__(self):
        try:
            return hash((self.time, self.state_var, self.value, self.method))
        except TypeError:
            return id(self)

    def __eq__(self, other):
        if not isinstance(other, DiscreteEvent):
            return False
        # If value is a tracer, comparison with == might return a tracer, so check identity or array equality safely
        # Standard fallback:
        try:
            val_eq = bool(self.value == other.value)
        except Exception:
            val_eq = self.value is other.value
        return (
            self.time == other.time
            and self.state_var == other.state_var
            and val_eq
            and self.method == other.method
        )

    def apply(self, state_dict: dict[str, float], state_names: list[str]) -> dict[str, float]:
        """
        Apply the event to a state dictionary.

        Args:
            state_dict: Dictionary mapping state variable names to values.
            state_names: List of state variable names for validation.

        Returns:
            Updated state dictionary.

        Raises:
            KeyError: If state_var is not in state_names.
        """
        if self.state_var not in state_names:
            raise KeyError(
                f"State variable '{self.state_var}' not found in state names: {state_names}"
            )

        new_state = state_dict.copy()
        current_value = new_state[self.state_var]

        if self.method == "replace":
            new_state[self.state_var] = self.value
        elif self.method == "add":
            new_state[self.state_var] = current_value + self.value
        elif self.method == "multiply":
            new_state[self.state_var] = current_value * self.value

        return new_state


__all__ = ["DiscreteEvent"]
