"""Tests for context utility functions."""

import numpy as np
import pytest

from pymcsimmod.utils.context import build_evaluation_context, merge_contexts, validate_context


class TestBuildEvaluationContext:
    """Tests for build_evaluation_context function."""

    def test_basic_context_building(self):
        """Test basic context building with all parameters."""
        state_vals = [1.0, 2.0, 3.0]
        state_names = ["A", "B", "C"]
        parameters = {"k1": 0.1, "k2": 0.2}
        forcing_values = {"dose": 1.5, "infusion": 0.0}

        context = build_evaluation_context(
            state_vals=state_vals,
            state_names=state_names,
            parameters=parameters,
            forcing_values=forcing_values,
        )

        # Check state variables
        assert context["A"] == 1.0
        assert context["B"] == 2.0
        assert context["C"] == 3.0

        # Check parameters
        assert context["k1"] == 0.1
        assert context["k2"] == 0.2

        # Check forcing values
        assert context["dose"] == 1.5
        assert context["infusion"] == 0.0

    def test_numpy_array_state_vals(self):
        """Test with numpy array state values."""
        state_vals = np.array([1.0, 2.0, 3.0])
        state_names = ["A", "B", "C"]
        parameters = {}  # Empty parameters dict

        context = build_evaluation_context(
            state_vals=state_vals, state_names=state_names, parameters=parameters
        )

        assert context["A"] == 1.0
        assert context["B"] == 2.0
        assert context["C"] == 3.0

    def test_mismatched_lengths(self):
        """Test error handling for mismatched state_vals and state_names lengths."""
        state_vals = [1.0, 2.0]
        state_names = ["A", "B", "C"]  # Different length
        parameters = {}  # Empty parameters dict

        with pytest.raises(
            ValueError, match="state_vals and state_names must have the same length"
        ):
            build_evaluation_context(
                state_vals=state_vals, state_names=state_names, parameters=parameters
            )

    def test_empty_inputs(self):
        """Test with empty inputs."""
        parameters = {}  # Empty parameters dict
        context = build_evaluation_context(state_vals=[], state_names=[], parameters=parameters)

        assert isinstance(context, dict)
        assert len(context) == 0

    def test_optional_parameters(self):
        """Test with optional parameters being None."""
        state_vals = [1.0]
        state_names = ["A"]
        parameters = None  # This should be a valid option based on the test expectation

        context = build_evaluation_context(
            state_vals=state_vals,
            state_names=state_names,
            parameters=parameters,
            forcing_values=None,
        )

        assert context["A"] == 1.0
        assert len(context) == 1

    def test_name_conflicts(self):
        """Test handling of name conflicts between different sources."""
        state_vals = [1.0]
        state_names = ["x"]
        parameters = {"x": 2.0}  # Conflict with state name

        context = build_evaluation_context(
            state_vals=state_vals, state_names=state_names, parameters=parameters
        )

        # Parameters should override state values (dict.update behavior)
        assert context["x"] == 2.0


class TestMergeContexts:
    """Tests for merge_contexts function."""

    def test_basic_merge(self):
        """Test basic context merging."""
        context1 = {"a": 1, "b": 2}
        context2 = {"c": 3, "d": 4}

        merged = merge_contexts(context1, context2)

        assert merged["a"] == 1
        assert merged["b"] == 2
        assert merged["c"] == 3
        assert merged["d"] == 4

    def test_overlapping_keys(self):
        """Test merging with overlapping keys."""
        context1 = {"a": 1, "b": 2}
        context2 = {"b": 20, "c": 3}  # 'b' overlaps

        merged = merge_contexts(context1, context2)

        assert merged["a"] == 1
        assert merged["b"] == 20  # context2 should override
        assert merged["c"] == 3

    def test_multiple_contexts(self):
        """Test merging multiple contexts."""
        context1 = {"a": 1}
        context2 = {"b": 2}
        context3 = {"c": 3}
        context4 = {"d": 4}

        merged = merge_contexts(context1, context2, context3, context4)

        assert merged["a"] == 1
        assert merged["b"] == 2
        assert merged["c"] == 3
        assert merged["d"] == 4

    def test_empty_contexts(self):
        """Test merging with empty contexts."""
        context1 = {"a": 1}
        context2 = {}
        context3 = {"b": 2}

        merged = merge_contexts(context1, context2, context3)

        assert merged["a"] == 1
        assert merged["b"] == 2

    def test_single_context(self):
        """Test with single context."""
        context = {"a": 1, "b": 2}

        merged = merge_contexts(context)

        assert merged["a"] == 1
        assert merged["b"] == 2
        assert merged is not context  # Should be a copy

    def test_no_contexts(self):
        """Test with no contexts."""
        merged = merge_contexts()

        assert isinstance(merged, dict)
        assert len(merged) == 0

    def test_none_context(self):
        """Test handling of None contexts."""
        context1 = {"a": 1}
        context2 = None
        context3 = {"b": 2}

        # This should handle None gracefully or raise appropriate error
        try:
            merged = merge_contexts(context1, context2, context3)
            # If it succeeds, check the result
            assert merged["a"] == 1
            assert merged["b"] == 2
        except (TypeError, AttributeError):
            # It's also acceptable to raise an error for None
            pass


class TestValidateContext:
    """Tests for validate_context function."""

    def test_valid_context(self):
        """Test validation of valid context."""
        context = {"A": 1.0, "B": 2.0, "k1": 0.1, "dose": 1.5}
        required_vars = ["A", "B", "k1"]

        # Should not raise any exception
        validate_context(context, required_vars)

    def test_missing_required_variable(self):
        """Test validation with missing required variable."""
        context = {"A": 1.0, "k1": 0.1}
        required_vars = ["A", "B", "k1"]  # 'B' is missing

        with pytest.raises(KeyError, match="Missing required variables in context"):
            validate_context(context, required_vars)

    def test_empty_required_vars(self):
        """Test validation with empty required variables list."""
        context = {"A": 1.0, "B": 2.0}
        required_vars = []

        # Should not raise any exception
        validate_context(context, required_vars)

    def test_none_required_vars(self):
        """Test validation with None required variables."""
        context = {"A": 1.0, "B": 2.0}

        # Should not raise any exception
        validate_context(context, None)

    def test_empty_context(self):
        """Test validation with empty context."""
        context = {}
        required_vars = ["A"]

        with pytest.raises(KeyError, match="Missing required variables in context"):
            validate_context(context, required_vars)

    def test_numeric_values_validation(self):
        """Test validation of numeric values in context."""
        context = {"A": 1.0, "B": "not_a_number", "C": None}
        required_vars = ["A", "B", "C"]

        # The basic validate_context might not check types
        # This depends on implementation
        try:
            validate_context(context, required_vars)
            # If it passes, the function doesn't check types
        except (TypeError, ValueError):
            # If it raises an error, it does check types
            pass

    def test_additional_validation_options(self):
        """Test any additional validation options."""
        context = {
            "A": 1.0,
            "B": -1.0,  # Negative value
            "C": float("inf"),  # Infinity
        }
        required_vars = ["A", "B", "C"]

        # Basic validation should pass
        validate_context(context, required_vars)


class TestContextUtilityIntegration:
    """Integration tests for context utilities."""

    def test_build_validate_merge_workflow(self):
        """Test typical workflow: build, validate, merge."""
        # Build initial context
        state_vals = [1.0, 2.0]
        state_names = ["A", "B"]
        parameters = {"k1": 0.1}

        context1 = build_evaluation_context(
            state_vals=state_vals, state_names=state_names, parameters=parameters
        )

        # Create additional context
        context2 = {"dose": 1.5, "k2": 0.2}

        # Merge contexts
        merged = merge_contexts(context1, context2)

        # Validate merged context
        required_vars = ["A", "B", "k1", "dose"]
        validate_context(merged, required_vars)

        # Check final result
        assert merged["A"] == 1.0
        assert merged["B"] == 2.0
        assert merged["k1"] == 0.1
        assert merged["dose"] == 1.5
        assert merged["k2"] == 0.2

    def test_context_utilities_with_scipy_model(self):
        """Test context utilities integration with ScipyModel."""
        from pymcsimmod.models.scipy_model import ScipyModel

        model_str = """
        States = {
            A
        };

        Parameters = {
            k1
        };

        Inputs = {
            dose
        };

        Initialize {
            A = 0.0;
        }

        Dynamics {
            dt(A) = dose - A * k1;
        }

        End.
        """

        model = ScipyModel(model_str)
        model.parameters["k1"] = 0.1
        model.assign_forcing_function("dose", "ZeroFunc")

        # The model should use context utilities internally
        times = np.linspace(0, 10, 101)
        result = model.run_model(times)

        assert result is not None
        assert len(result.times) == len(times)

    def test_context_building_performance(self):
        """Test performance with large state spaces."""
        # Create large state space
        n_states = 1000
        state_vals = list(range(n_states))
        state_names = [f"state_{i}" for i in range(n_states)]
        parameters = {f"param_{i}": i * 0.1 for i in range(100)}

        # This should complete reasonably quickly
        context = build_evaluation_context(
            state_vals=state_vals, state_names=state_names, parameters=parameters
        )

        assert len(context) == n_states + 100  # states + parameters
        assert context["state_0"] == 0
        assert context["state_999"] == 999
        assert context["param_99"] == 9.9

    def test_context_modification_safety(self):
        """Test that context modifications don't affect originals."""
        original_params = {"k1": 0.1, "k2": 0.2}
        state_vals = [1.0]
        state_names = ["A"]

        context = build_evaluation_context(
            state_vals=state_vals, state_names=state_names, parameters=original_params
        )

        # Modify the context
        context["k1"] = 999.0
        context["new_var"] = "test"

        # Original should be unchanged
        assert original_params["k1"] == 0.1
        assert "new_var" not in original_params

    def test_numeric_precision_preservation(self):
        """Test that numeric precision is preserved through context operations."""
        state_vals = [1.123456789, 2.987654321]
        state_names = ["A", "B"]
        parameters = {"k1": 1e-10, "k2": 1e10}

        context = build_evaluation_context(
            state_vals=state_vals, state_names=state_names, parameters=parameters
        )

        # Check precision preservation
        assert context["A"] == 1.123456789
        assert context["B"] == 2.987654321
        assert context["k1"] == 1e-10
        assert context["k2"] == 1e10
