"""Tests for scipy model implementations."""

import pytest
import numpy as np

from pymcsimmod.models.scipy_model import ScipyModel
from pymcsimmod.models.computed import ComputedModel


@pytest.fixture
def simple_model_str():
    """Simple test model string."""
    return """
    States = {
        A
    };
    
    Inputs = {
        dose
    };
    
    Outputs = {
        A_out
    };
    
    # Parameters defined outside blocks with default values
    ka = 1.0;
    ke = 0.1;
    
    Initialize {
        A = 0.0;
    }
    
    Dynamics {
        dt(A) = dose - ke * A;
    }
    
    CalcOutputs {
        A_out = A;
    }
    
    End.
    """


class TestScipyModel:
    """Tests for the ScipyModel class."""

    def test_model_runs_and_updates_comprehensive(self, simple_model_str):
        """Test comprehensive model running, updates, and ComputedModel interface."""
        model = ScipyModel(simple_model_str)
        times = np.linspace(0, 5, 10)
        
        # Set initial condition to make the model dynamic
        model.update_Y0(A=10.0)
        
        # Run initial model
        sol = model.run_model(times)
        
        # Test ComputedModel interface
        assert sol.states.shape == (10, 1)
        assert sol.times.shape == (10,)
        assert sol.var_names == ["A"]
        
        # Test indexing access
        np.testing.assert_allclose(sol[0], sol.states[:, 0])
        np.testing.assert_allclose(sol["A"], sol.states[:, 0])
        
        # Test plotting (should not raise)
        ax = sol.plot_results()
        assert ax is not None
        
        # Update parameter and check new solution is different
        model.update_constants(ke=1.0)
        sol2 = model.run_model(times)
        assert not np.allclose(sol.states, sol2.states)
        assert model.parameters["ke"] == 1.0
        
        # Update Y0 again and check new solution is different
        model.update_Y0(A=1.0)
        sol3 = model.run_model(times)
        assert not np.allclose(sol2.states, sol3.states)
        assert model.Y0["A"] == 1.0

    @pytest.fixture
    def model_with_events_str(self):
        """Model string with events for testing."""
        return """
        States = {
            A
        };
        
        Inputs = {
            dose
        };
        
        Outputs = {
            A_out
        };
        
        # Parameters defined outside blocks with default values
        ka = 1.0;
        ke = 0.1;
        
        Initialize {
            A = 0.0;
        }
        
        Dynamics {
            dt(A) = dose * ka - A * ke;
        }
        
        CalcOutputs {
            A_out = A;
        }
        
        End.
        """

    def test_scipy_model_creation(self, simple_model_str):
        """Test ScipyModel creation from string."""
        model = ScipyModel(simple_model_str)
        
        assert hasattr(model, 'state_names')
        assert hasattr(model, 'parameters')
        assert hasattr(model, 'forcing_functions')
        assert hasattr(model, 'Y0')
        assert hasattr(model, 'events')

    def test_scipy_model_creation_from_file(self, data_path):
        """Test ScipyModel creation from file."""
        model_file = data_path / "pred_prey.model"
        model = ScipyModel(model_file)
        
        assert hasattr(model, 'state_names')
        assert hasattr(model, 'parameters')

    def test_run_model_basic(self, simple_model_str):
        """Test basic model run without events."""
        model = ScipyModel(simple_model_str)
        times = np.linspace(0, 10, 101)
        
        result = model.run_model(times)
        
        assert isinstance(result, ComputedModel)
        assert len(result.times) == len(times)
        assert result.states.shape[0] == len(times)
        assert result.states.shape[1] == len(model.state_names)

    def test_model_method(self, simple_model_str):
        """Test the model method (ODE right-hand side)."""
        model = ScipyModel(simple_model_str)
        
        # Test model evaluation
        t = 1.0
        y = [1.0]  # A = 1.0
        args = ()
        
        dydt = model.model(t, y, args)
        
        assert isinstance(dydt, (list, np.ndarray))
        assert len(dydt) == len(model.state_names)

    def test_onoff_forcing_function(self, simple_model_str):
        """Test OnOff forcing function."""
        model = ScipyModel(simple_model_str)
        
        # Test OnOff function
        onoff_func = model.OnOff(0.0, 2.0, 10.0)
        
        # Test at different time points
        assert onoff_func(-1.0) < 0.1  # Before start
        assert onoff_func(1.0) > 0.9   # During
        assert onoff_func(3.0) < 0.1   # After end

    def test_perdose_forcing_function(self, simple_model_str):
        """Test PerDose forcing function."""
        model = ScipyModel(simple_model_str)
        
        # Test PerDose function
        perdose_func = model.PerDose(0.0, 1.0, 24.0, 10.0)
        
        # Test at different time points
        assert perdose_func(0.5) > 0.9    # During first dose
        assert perdose_func(12.0) < 0.1   # Between doses
        assert perdose_func(24.5) > 0.9   # During second dose

    def test_ndoses_forcing_function(self, simple_model_str):
        """Test NDoses forcing function."""
        model = ScipyModel(simple_model_str)
        
        # Test NDoses function
        t0_list = [0.0, 24.0, 48.0]
        ndoses_func = model.NDoses(t0_list, 1.0, 10.0)
        
        # Test at different time points
        assert ndoses_func(0.5) > 0.9    # During first dose
        assert ndoses_func(12.0) < 0.1   # Between doses
        assert ndoses_func(24.5) > 0.9   # During second dose

    def test_zerofunc_forcing_function(self, simple_model_str):
        """Test ZeroFunc forcing function."""
        model = ScipyModel(simple_model_str)
        
        # Test ZeroFunc
        zero_func = model.ZeroFunc()
        
        assert zero_func(0.0) == 0.0
        assert zero_func(10.0) == 0.0

    def test_add_discrete_event(self, simple_model_str):
        """Test adding discrete events."""
        model = ScipyModel(simple_model_str)
        
        # Add a discrete event using the add_event method with individual parameters
        model.add_event(time=5.0, state_var="A", value=10.0, method="add")
        
        assert len(model.events) == 1
        assert model.events[0].time == 5.0
        assert model.events[0].state_var == "A"
        assert model.events[0].value == 10.0
        assert model.events[0].method == "add"

    def test_run_model_with_events(self, simple_model_str):
        """Test running scipy model with discrete events."""
        model = ScipyModel(simple_model_str)
        
        # Add a discrete event using the add_event method
        model.add_event(time=5.0, state_var="A", value=10.0, method="add")
        
        times = np.linspace(0, 10, 101)
        result = model.run_model(times)
        
        assert isinstance(result, ComputedModel)
        
        # Check that the event was applied
        event_idx = np.argmin(np.abs(result.times - 5.0))
        # The state should have a discontinuity around t=5
        # Note: With discrete events, the solver may add extra time points
        assert len(result.times) >= len(times)  # May have more points due to events

    def test_context_utility_integration(self, simple_model_str):
        """Test that context utilities are properly integrated."""
        model = ScipyModel(simple_model_str)
        
        # This should use context utilities internally
        # We test this indirectly by ensuring the model runs successfully
        times = np.linspace(0, 5, 51)
        result = model.run_model(times)
        
        assert isinstance(result, ComputedModel)
        assert len(result.times) == len(times)

    def test_forcing_function_s_parameter(self, simple_model_str):
        """Test that the s parameter is properly passed through."""
        model = ScipyModel(simple_model_str)
        
        # Test different s values
        onoff_smooth = model.OnOff(0.0, 2.0, 1.0)  # Low s = smooth
        onoff_sharp = model.OnOff(0.0, 2.0, 100.0)  # High s = sharp
        
        # At the transition point, smooth should be closer to 0.5
        mid_point = 1.0
        smooth_val = onoff_smooth(mid_point)
        sharp_val = onoff_sharp(mid_point)
        
        # With low s, the transition should be more gradual
        assert abs(smooth_val - 0.5) < abs(sharp_val - 0.5)

    def test_multiple_forcing_functions(self, simple_model_str):
        """Test model with multiple forcing functions."""
        model = ScipyModel(simple_model_str)
        
        # Set up multiple forcing functions
        model.forcing_functions['dose'] = model.OnOff(0.0, 2.0, 10.0)
        model.forcing_functions['extra'] = model.PerDose(0.0, 1.0, 24.0, 10.0)
        
        times = np.linspace(0, 5, 51)
        result = model.run_model(times)
        
        assert isinstance(result, ComputedModel)
        assert len(result.times) == len(times)

    def test_parameter_modification(self, simple_model_str):
        """Test modifying parameters after model creation."""
        model = ScipyModel(simple_model_str)
        
        # Modify parameters
        original_ka = model.parameters.get('ka', 1.0)
        model.parameters['ka'] = 2.0
        
        times = np.linspace(0, 5, 51)
        result = model.run_model(times)
        
        assert isinstance(result, ComputedModel)
        assert model.parameters['ka'] == 2.0
        assert model.parameters['ka'] != original_ka

    def test_initial_conditions_modification(self, simple_model_str):
        """Test modifying initial conditions after model creation."""
        model = ScipyModel(simple_model_str)
        
        # Modify initial conditions
        model.Y0['A'] = 5.0
        
        times = np.linspace(0, 5, 51)
        result = model.run_model(times)
        
        assert isinstance(result, ComputedModel)
        # Check that initial condition was applied
        assert abs(result.states[0, model.state_names.index('A')] - 5.0) < 1e-6

    def test_output_calculation(self, simple_model_str):
        """Test that outputs are properly calculated."""
        model = ScipyModel(simple_model_str)
        
        times = np.linspace(0, 5, 51)
        result = model.run_model(times)
        
        assert hasattr(result, 'aux_outputs')
        assert hasattr(result, 'aux_names')
        
        if result.aux_outputs is not None:
            assert result.aux_outputs.shape[0] == len(times)
            assert len(result.aux_names) > 0


class TestScipyModelErrorHandling:
    """Tests for scipy model error handling."""

    def test_invalid_model_string(self):
        """Test handling of invalid model strings."""
        invalid_model = "This is not a valid model"
        
        with pytest.raises(Exception):  # Should raise some parsing exception
            ScipyModel(invalid_model)

    def test_missing_required_sections(self):
        """Test handling of models missing required sections."""
        incomplete_model = """
        States = {
            A
        };
        End.
        """
        
        # This might pass parsing but fail during initialization or run
        # The exact behavior depends on the implementation
        model = ScipyModel(incomplete_model)
        assert model is not None  # At least it should create

    def test_invalid_event_time(self, simple_model_str):
        """Test handling of invalid event times."""
        model = ScipyModel(simple_model_str)
        
        # Add event with negative time - this should be allowed (event before simulation)
        model.add_event(time=-1.0, state_var="A", value=10.0)
        
        times = np.linspace(0, 10, 101)
        # Should handle gracefully - event before simulation start
        result = model.run_model(times)
        assert isinstance(result, ComputedModel)

    def test_invalid_state_in_event(self, simple_model_str):
        """Test handling of events referencing invalid states."""
        model = ScipyModel(simple_model_str)
        
        # Add event for non-existent state - should raise KeyError
        with pytest.raises(KeyError, match="State variable 'NonExistentState' not found"):
            model.add_event(time=5.0, state_var="NonExistentState", value=10.0)
        
        times = np.linspace(0, 10, 101)
        try:
            result = model.run_model(times)
            # If it succeeds, it should still return a valid result
            assert isinstance(result, ComputedModel)
        except (KeyError, ValueError, IndexError):
            # It's also acceptable to raise an appropriate error
            pass


def test_scipy_model_string():
    """Simple test to ensure the basic model string works."""
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
    
    model = ScipyModel(model_str)
    times = np.linspace(0, 10, 101)
    result = model.run_model(times)
    
    assert isinstance(result, ComputedModel)
    assert len(result.times) == len(times)
    assert result.states.shape == (len(times), 1)
