"""Consolidated tests for backend functionality and standardization."""

import numpy as np
import pytest
from pydantic import ValidationError

from pymcsimmod.utils.backends import (
    detect_available_backends,
    get_backend_capabilities,
    recommend_backend,
    validate_backend,
)
from src.pymcsimmod import create_model
from src.pymcsimmod.config import BackendType


class TestDetectAvailableBackends:
    """Tests for backend detection functionality."""

    def test_detect_available_backends_returns_dict(self, available_backends):
        """Test that detect_available_backends returns a dictionary."""
        assert isinstance(available_backends, dict)
        assert "scipy" in available_backends
        assert "jax" in available_backends
        assert isinstance(available_backends["scipy"], bool)
        assert isinstance(available_backends["jax"], bool)

    def test_scipy_backend_detection(self, has_scipy):
        """Test scipy backend detection."""
        import importlib.util

        # Scipy should generally be available in test environment
        scipy_available = importlib.util.find_spec("scipy.integrate") is not None
        assert has_scipy == scipy_available

    def test_jax_backend_detection(self, has_jax):
        """Test JAX backend detection."""
        import importlib.util

        # JAX availability depends on installation
        jax_modules = ["jax", "equinox", "diffrax"]
        jax_available = all(importlib.util.find_spec(module) is not None for module in jax_modules)
        assert has_jax == jax_available


class TestValidateBackend:
    """Tests for backend validation functionality."""

    def test_validate_scipy_backend(self, has_scipy):
        """Test validation of scipy backend."""
        # Should not raise if scipy is available
        if has_scipy:
            validate_backend("scipy")  # Should not raise
        else:
            with pytest.raises(ImportError, match="Scipy backend requires"):
                validate_backend("scipy")

    def test_validate_jax_backend(self, has_jax):
        """Test validation of JAX backend."""
        # Should not raise if JAX is available
        if has_jax:
            validate_backend("jax")  # Should not raise
        else:
            with pytest.raises(ImportError, match="JAX backend requires"):
                validate_backend("jax")

    def test_validate_invalid_backend(self):
        """Test validation of invalid backend."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            validate_backend("invalid_backend")

    def test_validate_empty_backend_name(self):
        """Test validation with empty backend name."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            validate_backend("")

    def test_validate_none_backend(self):
        """Test validation with None backend."""
        with pytest.raises(TypeError):
            validate_backend(None)


class TestGetBackendCapabilities:
    """Tests for backend capabilities functionality."""

    def test_scipy_backend_capabilities(self):
        """Test scipy backend capabilities."""
        capabilities = get_backend_capabilities("scipy")

        assert isinstance(capabilities, dict)
        assert capabilities["discrete_events"] is True
        assert capabilities["forcing_functions"] is True
        assert capabilities["adaptive_stepping"] is True
        assert capabilities["event_detection"] is True
        assert capabilities["jit_compilation"] is False
        assert capabilities["automatic_differentiation"] is False

    def test_jax_backend_capabilities(self):
        """Test JAX backend capabilities."""
        capabilities = get_backend_capabilities("jax")

        assert isinstance(capabilities, dict)
        assert capabilities["discrete_events"] is False
        assert capabilities["forcing_functions"] is True
        assert capabilities["adaptive_stepping"] is True
        assert capabilities["event_detection"] is False
        assert capabilities["jit_compilation"] is True
        assert capabilities["automatic_differentiation"] is True

    def test_invalid_backend_capabilities(self):
        """Test capabilities query for invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend_capabilities("invalid_backend")

    def test_backend_capabilities_structure(self):
        """Test that backend capabilities have expected structure."""
        for backend in ["scipy", "jax"]:
            capabilities = get_backend_capabilities(backend)

            expected_keys = {
                "discrete_events",
                "forcing_functions",
                "adaptive_stepping",
                "event_detection",
                "jit_compilation",
                "automatic_differentiation",
            }

            assert set(capabilities.keys()) == expected_keys

            # All values should be boolean
            for key, value in capabilities.items():
                assert isinstance(value, bool)

    def test_backend_feature_matrix(self):
        """Test that backend feature matrix is consistent."""
        scipy_caps = get_backend_capabilities("scipy")
        jax_caps = get_backend_capabilities("jax")

        # Scipy should have events, JAX should not
        assert scipy_caps["discrete_events"] is True
        assert jax_caps["discrete_events"] is False

        # JAX should have JIT and autodiff, scipy should not
        assert jax_caps["jit_compilation"] is True
        assert jax_caps["automatic_differentiation"] is True
        assert scipy_caps["jit_compilation"] is False
        assert scipy_caps["automatic_differentiation"] is False

        # Both should have forcing functions and adaptive stepping
        assert scipy_caps["forcing_functions"] is True
        assert jax_caps["forcing_functions"] is True
        assert scipy_caps["adaptive_stepping"] is True
        assert jax_caps["adaptive_stepping"] is True


class TestRecommendBackend:
    """Tests for backend recommendation functionality."""

    def test_recommend_backend_no_requirements(self, available_backends):
        """Test backend recommendation with no special requirements."""
        recommended = recommend_backend()

        # Should recommend an available backend
        assert recommended in ["scipy", "jax"]
        assert available_backends[recommended] is True

    def test_recommend_backend_needs_events(self, has_scipy):
        """Test backend recommendation when events are required."""
        if has_scipy:
            recommended = recommend_backend(needs_events=True)
            assert recommended == "scipy"
        else:
            with pytest.raises(RuntimeError, match="Discrete events require scipy"):
                recommend_backend(needs_events=True)

    def test_recommend_backend_needs_jit(self, available_backends):
        """Test backend recommendation when JIT is preferred."""
        if available_backends["jax"]:
            recommended = recommend_backend(needs_jit=True)
            assert recommended == "jax"
        elif available_backends["scipy"]:
            recommended = recommend_backend(needs_jit=True)
            assert recommended == "scipy"  # Falls back to available backend
        else:
            with pytest.raises(RuntimeError, match="No supported backends"):
                recommend_backend(needs_jit=True)

    def test_recommend_backend_needs_autodiff(self, available_backends):
        """Test backend recommendation when autodiff is needed."""
        if available_backends["jax"]:
            recommended = recommend_backend(needs_autodiff=True)
            assert recommended == "jax"
        elif available_backends["scipy"]:
            recommended = recommend_backend(needs_autodiff=True)
            assert recommended == "scipy"  # Falls back to available backend
        else:
            with pytest.raises(RuntimeError, match="No supported backends"):
                recommend_backend(needs_autodiff=True)

    def test_recommend_backend_conflicting_requirements(self, has_scipy):
        """Test backend recommendation with conflicting requirements."""
        if has_scipy:
            # Events are only supported by scipy, so should recommend scipy
            # even if JIT is requested
            recommended = recommend_backend(needs_events=True, needs_jit=True)
            assert recommended == "scipy"
        else:
            with pytest.raises(RuntimeError, match="Discrete events require scipy"):
                recommend_backend(needs_events=True, needs_jit=True)

    def test_recommend_backend_all_requirements(self, has_scipy):
        """Test backend recommendation with all requirements."""
        if has_scipy:
            # Events requirement should force scipy selection
            recommended = recommend_backend(needs_events=True, needs_jit=True, needs_autodiff=True)
            assert recommended == "scipy"
        else:
            with pytest.raises(RuntimeError, match="Discrete events require scipy"):
                recommend_backend(needs_events=True, needs_jit=True, needs_autodiff=True)

    def test_recommend_backend_no_backends_available(self, monkeypatch):
        """Test backend recommendation when no backends are available."""

        # Mock detect_available_backends to return no available backends
        def mock_detect():
            return {"scipy": False, "jax": False}

        monkeypatch.setattr("pymcsimmod.utils.backends.detect_available_backends", mock_detect)

        with pytest.raises(RuntimeError, match="No supported backends are available"):
            recommend_backend()


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


class TestBackendWorkflows:
    """Integration tests for backend workflow functionality."""

    def test_backend_workflow(self, available_backends):
        """Test typical backend selection workflow."""
        # 1. Get recommendations based on requirements
        if available_backends["scipy"]:
            # Test scipy path
            recommended = recommend_backend(needs_events=True)
            assert recommended == "scipy"

            # Validate the backend
            validate_backend(recommended)

            # Check capabilities
            capabilities = get_backend_capabilities(recommended)
            assert capabilities["discrete_events"] is True

        if available_backends["jax"]:
            # Test JAX path
            recommended = recommend_backend(needs_jit=True)
            if recommended == "jax":  # Might fall back to scipy
                # Validate the backend
                validate_backend(recommended)

                # Check capabilities
                capabilities = get_backend_capabilities(recommended)
                assert capabilities["jit_compilation"] is True
