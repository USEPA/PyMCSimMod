"""Context management utilities for ODE model evaluation.

Provides a shared ``build_evaluation_context`` function used by both
SciPy and JAX backends to assemble the variable context for expression
evaluation. The function is pure (no side effects), making it safe for
use inside JAX-JIT compiled functions.
"""

from typing import Any


def build_evaluation_context(
    state_vals: list[float] | dict[str, float],
    state_names: list[str] | tuple[str, ...],
    parameters: dict[str, float],
    forcing_values: dict[str, float] | None = None,
    dynamic_calcs: dict[str, Any] | None = None,
    approach=None,
) -> dict[str, Any]:
    """
    Build a complete context dictionary for expression evaluation.

    This function is pure and side-effect-free, making it safe for use inside
    JAX-JIT compiled functions when ``approach=Approach.JAX``.

    Args:
        state_vals: State variable values (list/array or dict).
        state_names: Names of state variables (used when state_vals is a list/array).
        parameters: Model parameters dict.
        forcing_values: Current forcing function values (optional).
        dynamic_calcs: Dynamic calculations dict of ``{var: expr}`` (optional).
        approach: Evaluation approach (``Approach.SCIPY`` or ``Approach.JAX``).

    Returns:
        Complete context dictionary for expression evaluation.
    """
    # Build state variable entries
    if isinstance(state_vals, dict):
        context = dict(state_vals)
    else:
        if len(state_vals) != len(state_names):
            raise ValueError("state_vals and state_names must have the same length")
        context = {name: state_vals[i] for i, name in enumerate(state_names)}

    # Add parameters
    if parameters is not None:
        context.update(parameters)

    # Add forcing function values
    if forcing_values:
        context.update(forcing_values)

    # Evaluate dynamic calculations in order
    if dynamic_calcs and approach is not None:
        for var, expr in dynamic_calcs.items():
            context[var] = expr.evaluate(context, approach)

    return context


def validate_context(context: dict[str, float], required_vars: list[str]) -> None:
    """
    Validate that all required variables are present in the context.

    Args:
        context: Context dictionary to validate.
        required_vars: List of required variable names.

    Raises:
        KeyError: If any required variable is missing.
    """
    if required_vars is None:
        return

    missing = [var for var in required_vars if var not in context]
    if missing:
        raise KeyError(f"Missing required variables in context: {missing}")


def merge_contexts(*contexts: dict[str, float]) -> dict[str, float]:
    """
    Merge multiple context dictionaries, with later ones taking precedence.

    Args:
        *contexts: Context dictionaries to merge.

    Returns:
        Merged context dictionary.
    """
    merged: dict[str, float] = {}
    for context in contexts:
        merged.update(context)
    return merged


__all__ = ["build_evaluation_context", "merge_contexts", "validate_context"]
