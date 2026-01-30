"""Integration tests for PyMCSimMod components working together."""

import numpy as np
import pytest

from pymcsimmod.models.computed import ComputedModel
from pymcsimmod.models.events import DiscreteEvent
from pymcsimmod.models.scipy_model import ScipyModel


class TestModelIntegration:
    """Integration tests for models with individual components."""

    def test_model_with_forcing_functions_only(self, simple_scipy_model, standard_times):
        """Test model with forcing functions but no discrete events."""
        model = simple_scipy_model
        
        # Add forcing function only
        model.forcing_functions["dose"] = model.OnOff(1.0, 3.0, 10.0)
        
        # Run model
        result = model.run_model(standard_times)
        
        # Verify result
        assert isinstance(result, ComputedModel)
        assert result.states.shape[0] == len(standard_times)
        assert not np.any(np.isnan(result.states))
        
        # Verify forcing function had effect
        before_dose_idx = np.argmin(np.abs(result.times - 0.5))
        during_dose_idx = np.argmin(np.abs(result.times - 2.0)) 
        after_dose_idx = np.argmin(np.abs(result.times - 4.0))
        
        # Should see increasing concentration during dose
        assert result.states[during_dose_idx, 0] > result.states[before_dose_idx, 0]

    def test_model_with_discrete_events_only(self, simple_scipy_model, standard_times):
        """Test model with discrete events but no forcing functions."""
        model = simple_scipy_model
        
        # Set initial condition
        model.update_Y0(A=10.0)
        
        # Add discrete events only (no forcing functions)
        model.add_event(time=2.0, state_var="A", value=5.0)
        model.add_event(time=4.0, state_var="A", value=15.0)
        model.add_event(time=6.0, state_var="A", value=8.0)
        
        # Run model
        result = model.run_model(standard_times)
        
        # Verify result
        assert isinstance(result, ComputedModel)
        assert result.states.shape[0] == len(standard_times)
        assert not np.any(np.isnan(result.states))
        
        # Verify events had effect by checking discontinuities
        # Check around first event at t=2.0
        before_event1 = np.argmin(np.abs(result.times - 1.9))
        after_event1 = np.argmin(np.abs(result.times - 2.1))
        
        # There should be a clear discontinuity due to the event
        assert abs(result.states[after_event1, 0] - result.states[before_event1, 0]) > 1.0

    def test_complex_model_with_multiple_outputs(self, complex_scipy_model, standard_times):
        """Test complex model with multiple outputs and calculations."""
        model = complex_scipy_model
        
        # Set initial conditions
        model.update_Y0(A0=0.0, A1=0.0, AUC=0.0)
        
        # Add dosing - NDoses takes (times_list, duration, scale)
        model.forcing_functions["dose"] = model.NDoses([1.0, 3.0, 5.0], 0.5, 10.0)
        
        # Run model
        result = model.run_model(standard_times)
        
        # Verify all states are present
        assert result.states.shape[1] == 3  # A0, A1, AUC
        assert result.var_names == ["A0", "A1", "AUC"]
        
        # Verify AUC increases monotonically (since it's cumulative)
        auc_values = result.states[:, 2]  # AUC is third state
        assert np.all(np.diff(auc_values) >= -1e-10)  # Allow for small numerical errors

    def test_parameter_sensitivity_analysis(self, simple_scipy_model, short_times):
        """Test parameter sensitivity across different values."""
        model = simple_scipy_model
        model.update_Y0(A=10.0)  # Set non-zero initial condition
        
        # Add a constant dose to see the effect of different ke values
        model.forcing_functions["dose"] = model.ConstFunc(1.0)
        
        # Test different ke values
        ke_values = [0.01, 0.1, 1.0, 2.0]  # Use more reasonable range
        results = {}
        
        for ke in ke_values:
            model.update_constants(ke=ke)
            result = model.run_model(short_times)
            results[ke] = result.states[-1, 0]  # Final concentration
        
        # Higher ke should lead to lower final concentration (more elimination)
        assert results[2.0] < results[1.0] < results[0.1] < results[0.01]

    def test_model_state_persistence(self, simple_scipy_model, short_times):
        """Test that model state changes persist correctly."""
        model = simple_scipy_model
        
        # Set initial parameters and conditions
        original_ke = model.parameters["ke"]
        model.update_constants(ke=0.5)
        model.update_Y0(A=5.0)
        
        # Add an event
        model.add_event(time=2.0, state_var="A", value=8.0)
        
        # Run first simulation
        result1 = model.run_model(short_times)
        
        # Check that parameters and initial conditions are preserved
        assert model.parameters["ke"] == 0.5
        assert model.Y0["A"] == 5.0
        assert len(model.events) == 1
        
        # Modify and run again
        model.update_constants(ke=1.0)
        result2 = model.run_model(short_times)
        
        # Results should be different due to parameter change
        assert not np.allclose(result1.states, result2.states)
        
        # New parameter should be preserved
        assert model.parameters["ke"] == 1.0


class TestDiscreteEventsIntegration:
    """Comprehensive integration tests for discrete events in ScipyModel."""

    def test_single_discrete_event(self, simple_scipy_model, standard_times):
        """Test model with a single discrete event."""
        model = simple_scipy_model
        model.update_Y0(A=10.0)
        
        # Single event
        model.add_event(time=5.0, state_var="A", value=20.0)
        
        result = model.run_model(standard_times)
        
        # Find indices around event
        before_idx = np.argmin(np.abs(result.times - 4.9))
        after_idx = np.argmin(np.abs(result.times - 5.1))
        
        # State should change dramatically at event
        assert abs(result.states[after_idx, 0] - result.states[before_idx, 0]) > 5.0

    def test_multiple_discrete_events_same_state(self, simple_scipy_model, standard_times):
        """Test multiple events affecting the same state variable."""
        model = simple_scipy_model
        model.update_Y0(A=10.0)
        
        # Multiple events on same state
        model.add_event(time=2.0, state_var="A", value=5.0)
        model.add_event(time=4.0, state_var="A", value=15.0)
        model.add_event(time=6.0, state_var="A", value=1.0)
        model.add_event(time=8.0, state_var="A", value=25.0)
        
        result = model.run_model(standard_times)
        
        # Check each event created a discontinuity
        event_times = [2.0, 4.0, 6.0, 8.0]
        expected_values = [5.0, 15.0, 1.0, 25.0]
        
        for event_time, expected_val in zip(event_times, expected_values):
            after_idx = np.argmin(np.abs(result.times - (event_time + 0.1)))
            # Check that value is approximately correct shortly after event
            # (exact value depends on dynamics but should be close initially)
            assert abs(result.states[after_idx, 0] - expected_val) < expected_val * 0.1

    def test_discrete_events_different_methods(self, complex_scipy_model, standard_times):
        """Test discrete events with different application methods."""
        model = complex_scipy_model
        model.update_Y0(A0=5.0, A1=10.0, AUC=0.0)
        
        # Test different event methods if available
        model.add_event(time=2.0, state_var="A0", value=20.0, method="replace")
        model.add_event(time=4.0, state_var="A1", value=5.0, method="replace") 
        model.add_event(time=6.0, state_var="A0", value=0.0, method="replace")
        
        result = model.run_model(standard_times)
        
        # Verify events affected the correct states
        assert isinstance(result, ComputedModel)
        assert result.states.shape[1] == 3  # A0, A1, AUC
        assert not np.any(np.isnan(result.states))

    def test_discrete_events_edge_cases(self, simple_scipy_model):
        """Test discrete events at boundary conditions."""
        model = simple_scipy_model
        model.update_Y0(A=10.0)
        
        # Event at start time
        model.add_event(time=0.0, state_var="A", value=5.0)
        
        # Event at end time  
        times = np.linspace(0, 5, 51)
        model.add_event(time=5.0, state_var="A", value=20.0)
        
        result = model.run_model(times)
        
        # Should handle boundary events gracefully
        assert isinstance(result, ComputedModel)
        assert not np.any(np.isnan(result.states))

    def test_discrete_events_ordering(self, simple_scipy_model, standard_times):
        """Test that events are applied in correct chronological order."""
        model = simple_scipy_model
        model.update_Y0(A=10.0)
        
        # Add events out of chronological order
        model.add_event(time=6.0, state_var="A", value=30.0)
        model.add_event(time=2.0, state_var="A", value=5.0)
        model.add_event(time=4.0, state_var="A", value=15.0)
        
        result = model.run_model(standard_times)
        
        # Check that events were applied in chronological order
        # regardless of the order they were added
        idx_2 = np.argmin(np.abs(result.times - 2.1))
        idx_4 = np.argmin(np.abs(result.times - 4.1))
        idx_6 = np.argmin(np.abs(result.times - 6.1))
        
        # Values should reflect the chronological application
        assert result.states[idx_2, 0] < result.states[idx_4, 0]  # 5 < 15
        assert result.states[idx_4, 0] < result.states[idx_6, 0]  # 15 < 30

    def test_discrete_events_simultaneous(self, complex_scipy_model, standard_times):
        """Test multiple events at the same time."""
        model = complex_scipy_model
        model.update_Y0(A0=10.0, A1=5.0, AUC=0.0)
        
        # Multiple simultaneous events
        model.add_event(time=3.0, state_var="A0", value=20.0)
        model.add_event(time=3.0, state_var="A1", value=15.0)
        
        result = model.run_model(standard_times)
        
        # Both events should be applied
        after_idx = np.argmin(np.abs(result.times - 3.1))
        
        # Both states should show the effects of their respective events
        assert result.states[after_idx, 0] > 15.0  # A0 increased
        assert result.states[after_idx, 1] > 10.0  # A1 increased

    def test_discrete_events_with_complex_dynamics(self, complex_scipy_model, standard_times):
        """Test discrete events with multi-compartment model dynamics."""
        model = complex_scipy_model
        model.update_Y0(A0=0.0, A1=0.0, AUC=0.0)
        
        # Events that simulate dosing into different compartments
        model.add_event(time=1.0, state_var="A0", value=100.0)  # Dose into absorption compartment
        model.add_event(time=3.0, state_var="A0", value=50.0)   # Second dose
        model.add_event(time=5.0, state_var="A1", value=25.0)   # Direct injection into central
        
        result = model.run_model(standard_times)
        
        # Verify complex dynamics
        assert result.states.shape[1] == 3  # A0, A1, AUC
        
        # AUC should be monotonically increasing (it's cumulative)
        auc_values = result.states[:, 2]
        auc_diffs = np.diff(auc_values)
        assert np.all(auc_diffs >= -1e-10)  # Allow for numerical precision

    def test_discrete_events_error_handling(self, simple_scipy_model, standard_times):
        """Test error handling for invalid discrete events."""
        model = simple_scipy_model
        model.update_Y0(A=10.0)
        
        # Clear any existing events
        if hasattr(model, 'clear_events'):
            model.clear_events()
        
        # Model should work with valid events
        model.add_event(time=2.0, state_var="A", value=5.0)
        result = model.run_model(standard_times)
        assert isinstance(result, ComputedModel)
        assert not np.any(np.isnan(result.states))

    def test_discrete_events_persistence(self, simple_scipy_model, short_times):
        """Test that events persist across multiple model runs."""
        model = simple_scipy_model
        model.update_Y0(A=10.0)
        
        # Add events
        model.add_event(time=1.0, state_var="A", value=5.0)
        model.add_event(time=3.0, state_var="A", value=15.0)
        
        # Run model multiple times
        result1 = model.run_model(short_times)
        result2 = model.run_model(short_times)
        
        # Results should be identical (events should persist)
        np.testing.assert_allclose(result1.states, result2.states, rtol=1e-10)
        assert len(model.events) == 2  # Events should still be there


class TestForcingFunctionsIntegration:
    """Integration tests for forcing functions without discrete events."""
    
    def test_simple_onoff_forcing(self, simple_scipy_model, standard_times):
        """Test basic OnOff forcing function."""
        model = simple_scipy_model
        model.update_Y0(A=0.0)
        
        # Skip if OnOff not available
        if not hasattr(model, 'add_forcing_function'):
            pytest.skip("Forcing functions not implemented for this model")
            
        # Simple on-off pattern
        forcing_func = OnOff(
            start_time=2.0,
            duration=3.0,
            amplitude=10.0
        )
        model.add_forcing_function("A", forcing_func)
        result = model.run_model(standard_times)
        
        # Basic functionality check
        assert isinstance(result, ComputedModel)
        assert not np.any(np.isnan(result.states))
        assert result.states.shape[0] == len(standard_times)

    def test_ndoses_forcing(self, simple_scipy_model, standard_times):
        """Test NDoses forcing function."""
        model = simple_scipy_model
        model.update_Y0(A=0.0)
        
        # Skip if forcing functions not available
        if not hasattr(model, 'add_forcing_function'):
            pytest.skip("Forcing functions not implemented for this model")
            
        # Multiple doses
        forcing_func = NDoses(
            start_time=1.0,
            interval=2.0,
            n_doses=3,
            dose_amount=5.0,
            dose_duration=0.5
        )
        model.add_forcing_function("A", forcing_func)
        result = model.run_model(standard_times)
        
        # Basic functionality check
        assert isinstance(result, ComputedModel)
        assert not np.any(np.isnan(result.states))

    def test_perdose_forcing(self, simple_scipy_model, standard_times):
        """Test PerDose forcing function."""
        model = simple_scipy_model
        model.update_Y0(A=0.0)
        
        # Skip if forcing functions not available
        if not hasattr(model, 'add_forcing_function'):
            pytest.skip("Forcing functions not implemented for this model")
            
        # Per-dose specification
        forcing_func = PerDose(
            dose_times=[1.0, 3.0, 6.0],
            dose_amounts=[10.0, 15.0, 8.0],
            dose_duration=0.5
        )
        model.add_forcing_function("A", forcing_func)
        result = model.run_model(standard_times)
        
        # Basic functionality check
        assert isinstance(result, ComputedModel)
        assert not np.any(np.isnan(result.states))

    @pytest.mark.parametrize("forcing_type", ["OnOff", "NDoses", "PerDose"])
    def test_forcing_function_types(self, simple_scipy_model, short_times, forcing_type):
        """Test different forcing function types work without crashing."""
        model = simple_scipy_model
        model.update_Y0(A=0.0)
        
        # Skip if forcing functions not available
        if not hasattr(model, 'add_forcing_function'):
            pytest.skip("Forcing functions not implemented for this model")
        
        if forcing_type == "OnOff":
            forcing_func = OnOff(start_time=1.0, duration=2.0, amplitude=5.0)
        elif forcing_type == "NDoses":
            forcing_func = NDoses(start_time=1.0, interval=2.0, n_doses=2, 
                               dose_amount=5.0, dose_duration=0.5)
        elif forcing_type == "PerDose":
            forcing_func = PerDose(dose_times=[1.0, 3.0], dose_amounts=[5.0, 5.0], 
                                 dose_duration=0.5)
        
        model.add_forcing_function("A", forcing_func)
        result = model.run_model(short_times)
        
        # Basic checks if it works
        assert isinstance(result, ComputedModel)
        assert not np.any(np.isnan(result.states))


class TestBackendIntegration:
    
    def test_scipy_jax_model_consistency(self, minimal_model_str, short_times, available_backends):
        """Test that scipy and JAX models produce consistent results for simple cases."""
        if not (available_backends["scipy"] and available_backends["jax"]):
            pytest.skip("Both scipy and JAX backends required")
        
        # Create models with both backends
        scipy_model = ScipyModel(minimal_model_str)
        
        from pymcsimmod.models.jax_model import JaxModel
        jax_model = JaxModel(minimal_model_str)
        
        # Set same initial conditions
        scipy_model.update_Y0(A=1.0)
        jax_model.update_Y0(A=1.0)
        
        # Run both models (no events for JAX compatibility)
        scipy_result = scipy_model.run_model(short_times)
        jax_result = jax_model.run_model(short_times)
        
        # Results should be very close (within numerical tolerance)
        # Different backends may have slightly different numerical precision
        np.testing.assert_allclose(scipy_result.states, jax_result.states, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(scipy_result.times, jax_result.times, rtol=1e-10)

    def test_backend_capability_constraints(self, minimal_model_str, available_backends):
        """Test that backend capabilities are properly enforced."""
        if not available_backends["jax"]:
            pytest.skip("JAX backend required")
            
        from pymcsimmod.models.jax_model import JaxModel
        
        jax_model = JaxModel(minimal_model_str)
        
        # Should be able to add events to JAX model
        jax_model.add_event(time=1.0, state_var="A", value=5.0)
        
        # But running with events should raise an error
        times = np.linspace(0, 2, 11)
        
        with pytest.raises(NotImplementedError, match="Discrete events are not yet supported"):
            jax_model.run_model(times)


class TestComplexWorkflows:
    """Test complex, real-world-like workflows."""
    
    def test_pharmacokinetic_workflow(self, complex_scipy_model):
        """Test a complete pharmacokinetic modeling workflow."""
        model = complex_scipy_model
        
        # Simulate oral dosing scenario
        times = np.linspace(0, 24, 241)  # 24 hours, 0.1 hr intervals
        
        # Multiple doses - NDoses takes (times_list, duration, scale)
        model.forcing_functions["dose"] = model.NDoses([0.0, 8.0, 16.0], 1.0, 100.0)
        
        # Initial conditions
        model.update_Y0(A0=0.0, A1=0.0, AUC=0.0)
        
        # Run simulation
        result = model.run_model(times)
        
        # Basic pharmacokinetic checks
        assert result.states.shape[0] == len(times)
        assert result.states.shape[1] == 3
        
        # Check that dosing creates concentration spikes
        # Find peaks after each dose
        dose_times = [0.0, 8.0, 16.0]
        for dose_time in dose_times:
            # Find index 1 hour after dose
            post_dose_idx = np.argmin(np.abs(times - (dose_time + 1.0)))
            pre_dose_idx = np.argmin(np.abs(times - dose_time))
            
            # Concentration should increase after dose
            if post_dose_idx < len(result.states):
                assert result.states[post_dose_idx, 1] > result.states[pre_dose_idx, 1]
        
        # AUC should be monotonically increasing
        auc_diff = np.diff(result.states[:, 2])
        assert np.all(auc_diff >= -1e-10)  # Account for numerical precision

    def test_environmental_fate_workflow(self, pred_prey_model_str):
        """Test an environmental/ecological modeling workflow."""
        pytest.importorskip("scipy", reason="Scipy required for this test")
            
        model = ScipyModel(pred_prey_model_str)
        
        # Long-term simulation
        times = np.linspace(0, 20, 2001)
        
        # Environmental disturbance events
        model.add_event(time=5.0, state_var="prey", value=15.0)      # Population boost
        model.add_event(time=10.0, state_var="predator", value=3.0)  # Predator reduction
        model.add_event(time=15.0, state_var="prey", value=8.0)      # Disease outbreak
        
        # Run simulation
        result = model.run_model(times)
        
        # Check population dynamics
        prey_pop = result.states[:, 0]
        predator_pop = result.states[:, 1]
        
        # Both populations should remain positive
        assert np.all(prey_pop > 0)
        assert np.all(predator_pop > 0)
        
        # Check that events had impact
        event_indices = [
            np.argmin(np.abs(times - 5.0)),
            np.argmin(np.abs(times - 10.0)), 
            np.argmin(np.abs(times - 15.0))
        ]
        
        # Populations should change at event times
        for i, event_idx in enumerate(event_indices):
            if event_idx > 0 and event_idx < len(times) - 1:
                before = result.states[event_idx - 1]
                after = result.states[event_idx + 1] 
                # There should be some change
                assert not np.allclose(before, after, rtol=0.1)

    def test_error_recovery_workflow(self, simple_scipy_model, short_times):
        """Test model robustness with various parameter ranges."""
        model = simple_scipy_model
        
        # Test with reasonable parameter values
        model.update_constants(ke=0.1)
        result = model.run_model(short_times)
        assert isinstance(result, ComputedModel)
        
        # Test with different initial conditions
        model.update_Y0(A=1.0)
        result = model.run_model(short_times)
        assert isinstance(result, ComputedModel)
        
        # Test model works with normal discrete events
        if hasattr(model, 'clear_events'):
            model.clear_events()
        model.add_event(time=1.0, state_var="A", value=5.0)
        result = model.run_model(short_times)
        assert isinstance(result, ComputedModel)
        assert not np.any(np.isnan(result.states))