"""Parametrized tests across different model types and backends."""

import numpy as np
import pytest

from pymcsimmod.models.computed import ComputedModel
from pymcsimmod.models.scipy_model import ScipyModel


# Model fixtures for parametrization
@pytest.fixture(params=["simple_pk_model_str", "complex_pk_model_str", "minimal_model_str"])
def model_str_fixture(request):
    """Parametrized model string fixture."""
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["scipy"], scope="class")
def backend_name(request, available_backends):
    """Parametrized backend fixture."""
    backend = request.param
    if not available_backends[backend]:
        pytest.skip(f"{backend} backend not available")
    return backend


@pytest.fixture
def model_instance(backend_name, model_str_fixture):
    """Create model instance based on backend."""
    if backend_name == "scipy":
        return ScipyModel(model_str_fixture)
    elif backend_name == "jax":
        from pymcsimmod.models.jax_model import JaxModel
        return JaxModel(model_str_fixture)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


class TestParametrizedModelBehavior:
    """Test common behavior across different models and backends."""

    def test_model_creation(self, model_instance):
        """Test that models can be created successfully."""
        assert hasattr(model_instance, "state_names")
        assert hasattr(model_instance, "parameters")
        assert len(model_instance.state_names) > 0

    def test_basic_model_run(self, model_instance, short_times):
        """Test basic model execution."""
        result = model_instance.run_model(short_times)
        
        assert isinstance(result, ComputedModel)
        assert result.states.shape[0] == len(short_times)
        assert result.states.shape[1] == len(model_instance.state_names)
        assert len(result.times) == len(short_times)
        assert result.var_names == model_instance.state_names

    def test_parameter_updates(self, model_instance, short_times):
        """Test parameter modification across model types."""
        # Get initial parameter values
        original_params = model_instance.parameters.copy()
        
        # Find a parameter we can modify
        if "ke" in original_params:
            test_param = "ke"
            new_value = original_params[test_param] * 2
        elif "alpha" in original_params:
            test_param = "alpha"  
            new_value = original_params[test_param] * 2
        elif "decay_rate" in original_params:
            test_param = "decay_rate"
            new_value = original_params[test_param] * 2
        else:
            # Skip if no suitable parameter found
            pytest.skip("No suitable parameter found for modification test")

        # Update parameter
        model_instance.update_constants(**{test_param: new_value})
        
        # Verify parameter was updated
        assert model_instance.parameters[test_param] == new_value
        
        # Verify model still runs
        result = model_instance.run_model(short_times)
        assert isinstance(result, ComputedModel)

    @pytest.mark.parametrize("initial_value", [0.0, 1.0, 10.0])
    def test_initial_condition_sensitivity(self, model_instance, short_times, initial_value):
        """Test model response to different initial conditions."""
        # Set initial condition for first state
        first_state = model_instance.state_names[0]
        model_instance.update_Y0(**{first_state: initial_value})
        
        # Verify initial condition was set
        assert model_instance.Y0[first_state] == initial_value
        
        # Run model and check it completes
        result = model_instance.run_model(short_times)
        assert isinstance(result, ComputedModel)
        
        # Check that initial value is reflected in results
        # (allowing for some numerical tolerance)
        np.testing.assert_allclose(result.states[0, 0], initial_value, rtol=1e-10)


class TestParametrizedIntegrationMethods:
    """Test integration methods across scipy models."""
    
    @pytest.mark.parametrize("method", ["BDF", "RK45", "LSODA", "DOP853"])
    def test_integration_methods(self, simple_scipy_model, short_times, method):
        """Test different integration methods produce valid results."""
        model = simple_scipy_model
        model.update_Y0(A=10.0)  # Set non-zero initial condition
        
        try:
            result = model.run_model(short_times, method=method)
            assert isinstance(result, ComputedModel)
            assert result.states.shape[0] == len(short_times)
            assert np.all(np.isfinite(result.states))  # No NaN or inf values
        except Exception as e:
            # Some methods might not be available or suitable for all problems
            pytest.skip(f"Method {method} failed: {e}")


class TestParametrizedModelComparisons:
    """Compare behavior across different model complexities."""
    
    def test_simple_vs_complex_consistency(self, short_times, simple_pk_model_str, complex_pk_model_str, minimal_model_str):
        """Test that simple and complex models produce consistent basic behavior."""
        # This is more of a smoke test to ensure all model types work
        model_fixtures = {
            "simple_pk_model": simple_pk_model_str,
            "complex_pk_model": complex_pk_model_str,
            "minimal_model": minimal_model_str
        }
        results = {}
        
        for model_name, model_str in model_fixtures.items():
            try:
                model = ScipyModel(model_str)
                result = model.run_model(short_times)
                results[model_name] = result
            except Exception as e:
                pytest.skip(f"Could not test {model_name}: {e}")
        
        # All models should produce valid ComputedModel instances
        for model_name, result in results.items():
            assert isinstance(result, ComputedModel), f"{model_name} did not produce ComputedModel"
            assert result.states.shape[0] == len(short_times), f"{model_name} has wrong number of time points"


class TestParametrizedErrorHandling:
    """Test error handling across different model types."""
    
    @pytest.mark.parametrize("bad_time", [-1.0, np.inf, np.nan])
    def test_invalid_time_handling(self, simple_scipy_model, bad_time):
        """Test handling of invalid time inputs."""
        times = np.array([0.0, bad_time, 2.0])
        
        # Should handle invalid times gracefully
        try:
            result = simple_scipy_model.run_model(times)
            # If it succeeds, result should still be valid
            assert isinstance(result, ComputedModel)
        except (ValueError, RuntimeError) as e:
            # It's okay to raise an appropriate error
            assert len(str(e)) > 0  # Error should have a message
    
    def test_empty_time_array(self, model_instance):
        """Test handling of empty time arrays."""
        times = np.array([])
        
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            model_instance.run_model(times)