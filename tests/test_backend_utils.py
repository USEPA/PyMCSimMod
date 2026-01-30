"""Tests for backend utility functions."""

import pytest

from pymcsimmod.utils.backends import (
    detect_available_backends,
    get_backend_capabilities,
    recommend_backend,
    validate_backend,
)


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


class TestBackendUtilitiesIntegration:
    """Integration tests for backend utilities."""

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

    def test_backend_selection_with_model_creation(self, minimal_model_str, available_backends):
        """Test backend utilities with actual model creation."""
        # Test scipy backend if available
        if available_backends["scipy"]:
            validate_backend("scipy")
            from pymcsimmod.models.scipy_model import ScipyModel

            model = ScipyModel(minimal_model_str)
            assert model is not None

        # Test JAX backend if available
        if available_backends["jax"]:
            validate_backend("jax")
            from pymcsimmod.models.jax_model import JaxModel

            model = JaxModel(minimal_model_str)
            assert model is not None

    def test_capabilities_match_actual_functionality(self, minimal_model_str, available_backends):
        """Test that reported capabilities match actual functionality."""
        # Test scipy capabilities
        if available_backends["scipy"]:
            capabilities = get_backend_capabilities("scipy")

            # Scipy should support events
            assert capabilities["discrete_events"] is True

            # Should be able to create scipy model with events
            from pymcsimmod.models.scipy_model import ScipyModel

            model = ScipyModel(minimal_model_str)
            # Should be able to add events
            model.add_event(time=5.0, state_var="A", value=10.0)  # Should not raise

        # Test JAX capabilities
        if available_backends["jax"]:
            capabilities = get_backend_capabilities("jax")

            # JAX should not support discrete events
            assert capabilities["discrete_events"] is False

            # Should raise error when trying to use events with JAX
            from pymcsimmod.models.jax_model import JaxModel

            model = JaxModel(minimal_model_str)
            # JAX models can add events but running with events should fail
            model.add_event(time=5.0, state_var="A", value=10.0)

            import numpy as np

            times = np.linspace(0, 10, 101)

            with pytest.raises(NotImplementedError, match="Discrete events are not yet supported"):
                model.run_model(times)

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
