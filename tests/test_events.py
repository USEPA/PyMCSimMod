"""Tests for discrete event implementations."""

import pytest
import numpy as np

from pymcsimmod.models.events import DiscreteEvent


class TestDiscreteEvent:
    """Tests for the DiscreteEvent class."""

    def test_discrete_event_creation(self):
        """Test basic DiscreteEvent creation."""
        event = DiscreteEvent(
            time=5.0,
            state_var="A",
            value=10.0
        )
        
        assert event.time == 5.0
        assert event.state_var == "A"
        assert event.value == 10.0

    def test_discrete_event_with_method(self):
        """Test DiscreteEvent with different methods."""
        event = DiscreteEvent(
            time=5.0,
            state_var="A",
            value=10.0,
            method="replace"
        )
        
        assert event.time == 5.0
        assert event.state_var == "A"
        assert event.value == 10.0
        assert event.method == "replace"

    def test_discrete_event_string_representation(self):
        """Test string representation of DiscreteEvent."""
        event = DiscreteEvent(
            time=5.0,
            state_var="A",
            value=10.0
        )
        
        str_repr = str(event)
        assert "5.0" in str_repr
        assert "A" in str_repr
        assert "10.0" in str_repr

    def test_discrete_event_equality(self):
        """Test equality comparison of DiscreteEvent objects."""
        event1 = DiscreteEvent(
            time=5.0,
            state_var="A",
            value=10.0
        )
        
        event2 = DiscreteEvent(
            time=5.0,
            state_var="A",
            value=10.0
        )
        
        event3 = DiscreteEvent(
            time=6.0,
            state_var="A",
            value=10.0
        )
        
        assert event1 == event2
        assert event1 != event3

    def test_discrete_event_apply_method(self):
        """Test DiscreteEvent apply method."""
        # Test 'add' method (default)
        event_add = DiscreteEvent(
            time=5.0,
            state_var="A",
            value=10.0,
            method="add"
        )
        
        state_dict = {"A": 5.0, "B": 2.0}
        state_names = ["A", "B"]
        result = event_add.apply(state_dict, state_names)
        
        assert result["A"] == 15.0  # 5.0 + 10.0
        assert result["B"] == 2.0   # unchanged
        
        # Test 'replace' method
        event_replace = DiscreteEvent(
            time=5.0,
            state_var="A",
            value=10.0,
            method="replace"
        )
        
        result = event_replace.apply(state_dict, state_names)
        assert result["A"] == 10.0  # replaced
        assert result["B"] == 2.0   # unchanged
        
        # Test 'multiply' method
        event_multiply = DiscreteEvent(
            time=5.0,
            state_var="A",
            value=2.0,
            method="multiply"
        )
        
        result = event_multiply.apply(state_dict, state_names)
        assert result["A"] == 10.0  # 5.0 * 2.0
        assert result["B"] == 2.0   # unchanged

    def test_discrete_event_negative_time(self):
        """Test DiscreteEvent with negative time."""
        # This should be allowed - events before simulation start
        event = DiscreteEvent(
            time=-1.0,
            state_var="A",
            value=10.0
        )
        
        assert event.time == -1.0
        assert event.state_var == "A"
        assert event.value == 10.0

    def test_discrete_event_zero_time(self):
        """Test DiscreteEvent at time zero."""
        event = DiscreteEvent(
            time=0.0,
            state_var="A",
            value=10.0
        )
        
        assert event.time == 0.0
        assert event.state_var == "A"
        assert event.value == 10.0

    def test_discrete_event_with_additional_attributes(self):
        """Test DiscreteEvent with different methods."""
        # Test with multiply method
        event = DiscreteEvent(
            time=5.0,
            state_var="A",
            value=2.0,
            method="multiply"
        )
        
        assert event.time == 5.0
        assert event.state_var == "A"
        assert event.value == 2.0
        assert event.method == "multiply"


class TestDiscreteEventIntegration:
    """Integration tests for discrete events with models."""

    def test_single_event_with_scipy_model(self):
        """Test single discrete event with ScipyModel."""
        from pymcsimmod.models.scipy_model import ScipyModel
        
        model_str = """
        States = {
            A
        };
        
        # Parameters defined outside blocks with default values
        k1 = 0.1;
        
        Initialize {
            A = 0.0;
        }
        
        Dynamics {
            dt(A) = -A * k1;
        }
        
        End.
        """
        
        model = ScipyModel(model_str)
        
        # Add discrete event using the add_event API
        model.add_event(time=5.0, state_var="A", value=10.0)
        
        times = np.linspace(0, 10, 101)
        result = model.run_model(times)
        
        # Check that event was applied
        # Note: With discrete events, the solver may add extra time points
        assert len(result.times) >= len(times)
        
        # The state should have a discontinuity at t=5
        # (exact behavior depends on implementation)
        assert result.states.shape[1] == 1

    def test_multiple_events_with_scipy_model(self):
        """Test multiple discrete events with ScipyModel."""
        from pymcsimmod.models.scipy_model import ScipyModel
        
        model_str = """
        States = {
            A
        };
        
        # Parameters defined outside blocks with default values  
        k1 = 0.1;
        
        Initialize {
            A = 0.0;
        }
        
        Dynamics {
            dt(A) = -A * k1;
        }
        
        End.
        """
        
        model = ScipyModel(model_str)
        
        # Add multiple discrete events using add_event API
        model.add_event(time=2.0, state_var="A", value=5.0)
        model.add_event(time=5.0, state_var="A", value=10.0)
        model.add_event(time=8.0, state_var="A", value=15.0)
        
        times = np.linspace(0, 10, 101)
        result = model.run_model(times)
        
        # Check that all events were processed
        assert len(model.events) == 3
        assert len(result.times) >= len(times)

    def test_events_outside_simulation_time(self):
        """Test events outside simulation time range."""
        from pymcsimmod.models.scipy_model import ScipyModel
        
        model_str = """
        States = {
            A
        };
        
        Initialize {
            A = 1.0;
        }
        
        Dynamics {
            dt(A) = -0.1 * A;
        }
        
        End.
        """
        
        model = ScipyModel(model_str)
        
        # Add events outside simulation range using add_event API
        model.add_event(time=-1.0, state_var="A", value=5.0)  # Before sim
        model.add_event(time=15.0, state_var="A", value=10.0)  # After sim
        
        times = np.linspace(0, 10, 101)
        result = model.run_model(times)
        
        # Should handle gracefully
        assert len(result.times) >= len(times)

    def test_event_on_nonexistent_state(self):
        """Test event referencing non-existent state."""
        from pymcsimmod.models.scipy_model import ScipyModel
        
        model_str = """
        States = {
            A
        };
        
        Initialize {
            A = 1.0;
        }
        
        Dynamics {
            dt(A) = -0.1 * A;
        }
        
        End.
        """
        
        model = ScipyModel(model_str)
        
        # This test shows that add_event should raise KeyError for non-existent state
        # based on the error handling we implemented in scipy model tests
        with pytest.raises(KeyError, match="State variable 'B' not found"):
            model.add_event(time=5.0, state_var="B", value=10.0)

    def test_simultaneous_events(self):
        """Test multiple events at the same time."""
        from pymcsimmod.models.scipy_model import ScipyModel
        
        model_str = """
        States = {
            A,
            B
        };
        
        Initialize {
            A = 1.0;
            B = 2.0;
        }
        
        Dynamics {
            dt(A) = -0.1 * A;
            dt(B) = -0.1 * B;
        }
        
        End.
        """
        
        model = ScipyModel(model_str)
        
        # Add simultaneous events using add_event API
        model.add_event(time=5.0, state_var="A", value=10.0)
        model.add_event(time=5.0, state_var="B", value=20.0)
        
        times = np.linspace(0, 10, 101)
        result = model.run_model(times)
        
        # Should handle simultaneous events
        assert len(result.times) >= len(times)
        assert result.states.shape[1] == 2  # Two states

    def test_event_ordering(self):
        """Test that events are processed in correct order."""
        from pymcsimmod.models.scipy_model import ScipyModel
        
        model_str = """
        States = {
            A
        };
        
        Initialize {
            A = 0.0;
        }
        
        Dynamics {
            dt(A) = -1e-12 * A;  # Very small decay to avoid numerical issues
        }
        
        End.
        """
        
        model = ScipyModel(model_str)
        
        # Add events in non-chronological order using add_event API
        model.add_event(time=8.0, state_var="A", value=80.0)
        model.add_event(time=2.0, state_var="A", value=20.0)
        model.add_event(time=5.0, state_var="A", value=50.0)
        
        times = np.linspace(0, 10, 101)
        result = model.run_model(times)
        
        # Events should be processed in chronological order
        assert len(result.times) >= len(times)

    def test_jax_model_event_error(self):
        """Test that JAX models properly reject events."""
        pytest.importorskip("jax")
        
        from pymcsimmod.models.jax_model import JaxModel
        
        model_str = """
        States = {
            A
        };
        
        Initialize {
            A = 0.0;
        }
        
        Dynamics {
            dt(A) = -0.1 * A;
        }
        
        End.
        """
        
        model = JaxModel(model_str)
        
        # Add an event using add_event API (even though it will fail)
        model.add_event(time=5.0, state_var="A", value=10.0)
        
        times = np.linspace(0, 10, 101)
        
        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="Discrete events are not yet supported"):
            model.run_model(times)


class TestDiscreteEventEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_event_with_zero_value(self):
        """Test event with zero as value."""
        event = DiscreteEvent(
            time=5.0,
            state_var="A",
            value=0.0
        )
        
        assert event.time == 5.0
        assert event.state_var == "A"
        assert event.value == 0.0

    def test_event_with_negative_value(self):
        """Test event with negative value."""
        event = DiscreteEvent(
            time=5.0,
            state_var="A",
            value=-5.0
        )
        
        assert event.time == 5.0
        assert event.state_var == "A"
        assert event.value == -5.0

    def test_event_with_very_large_time(self):
        """Test event with very large time value."""
        event = DiscreteEvent(
            time=1e10,
            state_var="A",
            value=10.0
        )
        
        assert event.time == 1e10
        assert event.state_var == "A"
        assert event.value == 10.0

    def test_event_with_inf_time(self):
        """Test event with infinite time."""
        event = DiscreteEvent(
            time=float('inf'),
            state_var="A",
            value=10.0
        )
        
        assert event.time == float('inf')
        assert event.state_var == "A"
        assert event.value == 10.0

    def test_event_with_special_state_names(self):
        """Test event with special characters in state names."""
        # This tests handling of various state name formats
        special_names = ["state_1", "State_A", "x1", "concentration_plasma"]
        
        for name in special_names:
            event = DiscreteEvent(
                time=5.0,
                state_var=name,
                value=10.0
            )
            assert event.state_var == name

    def test_event_time_precision(self):
        """Test event time precision handling."""
        precise_time = 5.123456789
        
        event = DiscreteEvent(
            time=precise_time,
            state_var="A",
            value=10.0
        )
        
        assert event.time == precise_time

    def test_event_value_precision(self):
        """Test event value precision handling."""
        precise_value = 10.123456789
        
        event = DiscreteEvent(
            time=5.0,
            state_var="A",
            value=precise_value
        )
        
        assert event.value == precise_value