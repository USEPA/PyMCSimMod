"""Discrete event handling for ODE models."""

from typing import Literal

from pydantic import BaseModel


class DiscreteEvent(BaseModel):
    """
    Represents a discrete event similar to deSolve's event handling.

    Attributes:
        time: Time at which the event occurs.
        state_var: Name of the state variable to modify.
        value: Value to use in the operation.
        method: Type of operation ('replace', 'add', 'multiply').
    """

    time: float
    state_var: str
    value: float
    method: Literal["replace", "add", "multiply"] = "add"

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
