"""Test backend standardization with Pydantic validation."""

import pytest
from pydantic import ValidationError
import numpy as np

from src.pymcsimmod import create_model
from src.pymcsimmod.config import BackendType


class TestBackendStandardization:
    """Test that backend standardization works correctly throughout the package."""

    def test_create_model_with_string_backends(self):
        """Test that create_model accepts string backends and validates them."""
        # These will fail due to invalid model strings, but should validate the backend
        with pytest.raises(Exception) as exc_info:
            create_model("invalid_model", "scipy")
        # Should not be a ValidationError - backend validation should pass
        assert not isinstance(exc_info.value, ValidationError)
        
        with pytest.raises(Exception) as exc_info:
            create_model("invalid_model", "jax") 
        assert not isinstance(exc_info.value, ValidationError)

    def test_create_model_with_enum_backends(self):
        """Test that create_model accepts BackendType enums."""
        with pytest.raises(Exception) as exc_info:
            create_model("invalid_model", BackendType.SCIPY)
        assert not isinstance(exc_info.value, ValidationError)
        
        with pytest.raises(Exception) as exc_info:
            create_model("invalid_model", BackendType.JAX)
        assert not isinstance(exc_info.value, ValidationError)

    def test_create_model_rejects_invalid_backends(self):
        """Test that create_model properly rejects invalid backends.""" 
        with pytest.raises(ValidationError) as exc_info:
            create_model("dummy", "invalid_backend")
        
        error_message = str(exc_info.value)
        assert "Input should be" in error_message
        assert "scipy" in error_message or "jax" in error_message

    def test_backend_type_enum_functionality(self):
        """Test BackendType enum basic functionality."""
        # Test enum creation from strings
        assert BackendType("scipy") == BackendType.SCIPY
        assert BackendType("jax") == BackendType.JAX
        
        # Test enum values
        assert BackendType.SCIPY.value == "scipy"
        assert BackendType.JAX.value == "jax"
        
        # Test string representation (BackendType inherits from str, so .value gives the string)
        assert BackendType.SCIPY.value == "scipy"
        assert BackendType.JAX.value == "jax"
        
        # Test that it works as a string in string contexts
        assert f"{BackendType.SCIPY}" == "scipy"  # StrEnum representation shows value
        assert BackendType.SCIPY == "scipy"  # But equals the string value

    def test_backend_type_validation_errors(self):
        """Test that BackendType raises appropriate errors for invalid inputs."""
        with pytest.raises(ValueError) as exc_info:
            BackendType("invalid")
        
        assert "is not a valid BackendType" in str(exc_info.value)

    def test_forcing_functions_use_backend_enum(self):
        """Test that forcing functions work with BackendType enum."""
        from src.pymcsimmod.forcing.unified import UnifiedForcingFactory
        
        # Test with enum
        func = UnifiedForcingFactory.create_onoff(0, 5, backend=BackendType.SCIPY)
        assert callable(func)
        assert func(2.5) == 1.0  # Should be fully "on"
        
        # Test with string (should also work due to automatic conversion)  
        func2 = UnifiedForcingFactory.create_onoff(0, 5, backend=BackendType("scipy"))
        assert callable(func2)
        assert func2(2.5) == 1.0

class TestBackendIntegration:
    """Test that different backends integrate correctly with models."""
    def test_scipy_jax_model_consistency(self, minimal_model_str, short_times, available_backends):
        """Test that scipy and JAX models produce consistent results for simple cases."""
        if not (available_backends["scipy"] and available_backends["jax"]):
            pytest.skip("Both scipy and JAX backends required")
        
        # Create models with both backends
        scipy_model = create_model(minimal_model_str, BackendType.SCIPY)
        jax_model = create_model(minimal_model_str, BackendType.JAX)
        
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
        
        
        jax_model = create_model(minimal_model_str, BackendType.JAX)
        
        # Should be able to add events to JAX model
        jax_model.add_event(time=1.0, state_var="A", value=5.0)
        
        # But running with events should raise an error
        times = np.linspace(0, 2, 11)
        
        with pytest.raises(NotImplementedError, match="Discrete events are not yet supported"):
            jax_model.run_model(times)