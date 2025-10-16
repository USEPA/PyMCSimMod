"""Context management utilities for ODE model evaluation."""

from typing import Any


def build_evaluation_context(
    state_vals: list[float] | dict[str, float],
    state_names: list[str],
    parameters: dict[str, float],
    forcing_values: dict[str, float] | None = None,
    dynamic_calcs: dict[str, Any] | None = None,
    approach=None,
) -> dict[str, float]:
    """
    Build a complete context dictionary for expression evaluation.
    
    Args:
        state_vals: State variable values (list or dict)
        state_names: Names of state variables (if state_vals is a list)
        parameters: Model parameters
        forcing_values: Current forcing function values (optional)
        dynamic_calcs: Dynamic calculations dictionary (optional)
        approach: Evaluation approach for dynamic calculations (optional)
        
    Returns:
        Complete context dictionary for expression evaluation
    """
    # Handle state variables
    if isinstance(state_vals, dict):
        context = state_vals.copy()
    else:
        # Validate that state_vals and state_names have the same length
        if len(state_vals) != len(state_names):
            raise ValueError("state_vals and state_names must have the same length")
        context = {name: state_vals[i] for i, name in enumerate(state_names)}
    
    # Add parameters (handle None case)
    if parameters is not None:
        context.update(parameters)
    
    # Add forcing function values
    if forcing_values:
        context.update(forcing_values)
    
    # Add dynamic calculations
    if dynamic_calcs and approach:
        for var, expr in dynamic_calcs.items():
            context[var] = expr.evaluate(context, approach)
    
    return context


def validate_context(context: dict[str, float], required_vars: list[str]) -> None:
    """
    Validate that all required variables are present in the context.
    
    Args:
        context: Context dictionary to validate
        required_vars: List of required variable names
        
    Raises:
        KeyError: If any required variable is missing
    """
    # Handle None case
    if required_vars is None:
        return
        
    missing = [var for var in required_vars if var not in context]
    if missing:
        raise KeyError(f"Missing required variables in context: {missing}")


def merge_contexts(*contexts: dict[str, float]) -> dict[str, float]:
    """
    Merge multiple context dictionaries with later ones taking precedence.
    
    Args:
        *contexts: Context dictionaries to merge
        
    Returns:
        Merged context dictionary
    """
    merged = {}
    for context in contexts:
        merged.update(context)
    return merged


__all__ = ["build_evaluation_context", "merge_contexts", "validate_context"]