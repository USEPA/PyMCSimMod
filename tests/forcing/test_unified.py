"""Comprehensive tests for the unified forcing function backend system."""

import numpy as np
import pytest

from src.pymcsimmod.config import BackendType
from src.pymcsimmod.forcing.unified import (
    JAXBackend,
    ScipyBackend,
    UnifiedForcingFactory,
    create_constantfunc,
    create_ndoses,
    create_onoff,
    create_perdose,
    create_zerofunc,
)
from src.pymcsimmod.models.scipy_model import ScipyModel


class TestScipyForcingFunctions:
    """Test scipy backend forcing functions."""

    def test_onoff_creation(self):
        """Test OnOff forcing function creation and behavior."""
        # Test creation
        func = UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, s=10.0, backend=BackendType.SCIPY)
        assert callable(func)

        # Test before activation
        result_before = func(0.5)
        assert abs(result_before) < 0.01  # Should be near 0

        # Test during activation
        result_during = func(2.0)
        assert result_during > 0.95  # Should be near 1

        # Test after activation
        result_after = func(4.0)
        assert abs(result_after) < 0.01  # Should be near 0

    def test_onoff_via_unified_factory(self):
        """Test OnOff via create_forcing_function method."""
        func = UnifiedForcingFactory.create_forcing_function(
            "OnOff", backend=BackendType.SCIPY, t0=1.0, t1=3.0, s=10.0
        )
        assert callable(func)

        # Test behavior
        assert abs(func(0.5)) < 0.01  # Before
        assert func(2.0) > 0.95  # During
        assert abs(func(4.0)) < 0.01  # After

    def test_perdose_creation(self):
        """Test PerDose periodic behavior."""
        func = UnifiedForcingFactory.create_perdose(
            t0=0.0, duration=1.0, period=4.0, s=10.0, backend=BackendType.SCIPY
        )
        assert callable(func)

        # Test first dose period
        assert func(0.5) > 0.95  # During first dose
        assert abs(func(2.0)) < 0.01  # Between doses

        # Test second dose period
        assert func(4.5) > 0.95  # During second dose
        assert abs(func(6.0)) < 0.01  # Between doses

    def test_ndoses_creation(self):
        """Test NDoses discrete dosing behavior."""
        t0_list = [1.0, 5.0, 10.0]
        func = UnifiedForcingFactory.create_ndoses(
            t0_list=t0_list, duration=1.0, s=10.0, backend=BackendType.SCIPY
        )
        assert callable(func)

        # Test individual doses
        assert func(1.5) > 0.95  # During first dose
        assert func(5.5) > 0.95  # During second dose
        assert func(10.5) > 0.95  # During third dose

        # Test between doses
        assert abs(func(3.0)) < 0.01  # Between first and second
        assert abs(func(12.0)) < 0.01  # After all doses

    def test_zerofunc_creation(self):
        """Test ZeroFunc always returns zero."""
        func = UnifiedForcingFactory.create_zerofunc(backend=BackendType.SCIPY)
        assert callable(func)

        # Test various time points
        assert func(0.0) == 0.0
        assert func(5.0) == 0.0
        assert func(100.0) == 0.0

    def test_constantfunc_creation(self):
        """Test ConstFunc returns constant value."""
        val = 42.5
        func = UnifiedForcingFactory.create_constantfunc(val=val, backend=BackendType.SCIPY)
        assert callable(func)

        # Test various time points
        assert func(0.0) == val
        assert func(5.0) == val
        assert func(100.0) == val

    def test_ndoses_array_inputs(self):
        """Test NDoses with array inputs for scipy backend."""
        func = UnifiedForcingFactory.create_ndoses(
            t0_list=[1.0, 5.0], duration=1.0, backend=BackendType.SCIPY
        )

        # Test with numpy array input to trigger array branch
        t_array = np.array([1.5, 3.0, 5.5])
        results = func(t_array)

        assert len(results) == 3
        assert results[0] > 0.95  # During first dose
        assert abs(results[1]) < 0.01  # Between doses
        assert results[2] > 0.95  # During second dose

    def test_array_inputs(self):
        """Test with numpy array inputs."""
        func = UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend=BackendType.SCIPY)

        # Test with array input
        t_array = np.array([0.5, 2.0, 4.0])
        results = func(t_array)

        assert len(results) == 3
        assert abs(results[0]) < 0.01  # Before
        assert results[1] > 0.95  # During
        assert abs(results[2]) < 0.01  # After

    def test_parameter_validation(self):
        """Test missing required parameters raise ValueError."""
        # Test OnOff missing parameters
        with pytest.raises(ValueError, match="OnOff forcing function requires"):
            UnifiedForcingFactory.create_forcing_function("OnOff", backend=BackendType.SCIPY)

        with pytest.raises(ValueError, match="OnOff forcing function requires"):
            UnifiedForcingFactory.create_forcing_function(
                "OnOff", backend=BackendType.SCIPY, t0=1.0
            )

        # Test PerDose missing parameters
        with pytest.raises(ValueError, match="PerDose forcing function requires"):
            UnifiedForcingFactory.create_forcing_function(
                "PerDose", backend=BackendType.SCIPY, t0=1.0
            )

        # Test NDoses missing parameters
        with pytest.raises(ValueError, match="NDoses forcing function requires"):
            UnifiedForcingFactory.create_forcing_function("NDoses", backend=BackendType.SCIPY)

        # Test ConstFunc missing parameters
        with pytest.raises(ValueError, match="ConstFunc forcing function requires"):
            UnifiedForcingFactory.create_forcing_function("ConstFunc", backend=BackendType.SCIPY)

    def test_unknown_function_type(self):
        """Test error handling for unknown function types."""
        with pytest.raises(ValueError, match="Unknown forcing function type"):
            UnifiedForcingFactory.create_forcing_function("UnknownFunc", backend=BackendType.SCIPY)

    def test_convenience_functions(self):
        """Test convenience functions maintain compatibility."""
        # Test convenience functions exist and work
        func_onoff = create_onoff(t0=1.0, t1=3.0, backend=BackendType.SCIPY)
        func_perdose = create_perdose(t0=0.0, duration=1.0, period=4.0, backend=BackendType.SCIPY)
        func_ndoses = create_ndoses(t0_list=[1.0, 5.0], duration=1.0, backend=BackendType.SCIPY)
        func_zero = create_zerofunc(backend=BackendType.SCIPY)
        func_const = create_constantfunc(val=10.0, backend=BackendType.SCIPY)

        # All should be callable
        assert all(
            callable(f) for f in [func_onoff, func_perdose, func_ndoses, func_zero, func_const]
        )

        # Test basic functionality
        assert func_onoff(2.0) > 0.95
        assert func_perdose(0.5) > 0.95
        assert func_ndoses(1.5) > 0.95
        assert func_zero(5.0) == 0.0
        assert func_const(5.0) == 10.0

    @pytest.mark.parametrize(
        "func_type,params",
        [
            ("OnOff", {"t0": 1.0, "t1": 3.0}),
            ("PerDose", {"t0": 0.0, "duration": 1.0, "period": 4.0}),
            ("NDoses", {"t0_list": [1.0, 5.0], "duration": 1.0}),
            ("ZeroFunc", {}),
            ("ConstFunc", {"value": 5.0}),
        ],
    )
    def test_parametrized_functions(self, func_type, params):
        """Parametrized test for multiple function types."""
        func = UnifiedForcingFactory.create_forcing_function(
            func_type, backend=BackendType.SCIPY, **params
        )
        assert callable(func)

        # Test function returns numeric value
        result = func(2.0)
        assert isinstance(result, int | float | np.number)


class TestJAXForcingFunctions:
    """Test JAX backend forcing functions."""

    @pytest.fixture
    def check_jax_available(self):
        """Skip tests if JAX is not available."""
        pytest.importorskip("jax", reason="JAX not available")

    def test_jit_compilation(self, check_jax_available):
        """Test JAX functions are JIT compiled."""
        func = UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend=BackendType.JAX)

        # Check that function is JIT compiled (should have specific JAX type)
        func_type_name = type(func).__name__
        assert "Pjit" in func_type_name or "jit" in func_type_name.lower()

    def test_jax_array_compatibility(self, check_jax_available):
        """Test with various JAX array shapes and types."""
        import jax.numpy as jnp

        func = UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend=BackendType.JAX)

        # Test with standard JAX array
        t_jax = jnp.array([0.5, 2.0, 4.0])
        results = func(t_jax)
        assert hasattr(results, "__array__") or "jax" in str(type(results))
        assert len(results) == 3

        # Test with different JAX dtypes
        t_float32 = jnp.array([0.5, 2.0, 4.0], dtype=jnp.float32)
        results_f32 = func(t_float32)
        assert len(results_f32) == 3

        # Test with large arrays (ODE solver-like)
        t_large = jnp.linspace(0, 5, 1000)
        results_large = func(t_large)
        assert len(results_large) == 1000

        # Test with reshaped arrays (multi-dimensional)
        t_2d = jnp.reshape(t_jax, (3, 1))
        results_2d = func(t_2d)
        assert results_2d.shape == (3, 1)

        # Test with broadcasted operations
        t_broadcast = jnp.broadcast_to(jnp.array([2.0]), (5,))
        results_broadcast = func(t_broadcast)
        assert len(results_broadcast) == 5

    def test_jit_traceability(self, check_jax_available):
        """Test functions work with advanced JAX transformations."""
        import jax
        import jax.numpy as jnp

        func = UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend=BackendType.JAX)
        ndoses_func = UnifiedForcingFactory.create_ndoses(
            t0_list=[1.0, 5.0], duration=1.0, backend=BackendType.JAX
        )

        # Test nested JIT compilation
        @jax.jit
        def test_wrapper(t):
            return func(t) * 2.0

        result = test_wrapper(2.0)
        assert result > 1.8  # Should be close to 2.0

        # Test vmap (vectorization) compatibility
        vmapped_func = jax.vmap(func)
        vmap_input = jnp.array([0.5, 2.0, 4.0])
        vmap_result = vmapped_func(vmap_input)
        assert vmap_result.shape == (3,)

        # Test automatic differentiation compatibility
        grad_func = jax.grad(lambda t: jnp.sum(func(t)))
        grad_result = grad_func(2.0)
        assert isinstance(grad_result, int | float) or hasattr(grad_result, "__array__")

        # Test function composition
        @jax.jit
        def composed_func(t):
            return ndoses_func(t) * func(t) + 0.5

        composed_result = composed_func(jnp.array([1.5, 2.0]))
        assert composed_result.shape == (2,)

    def test_all_function_types_jit_compiled(self, check_jax_available):
        """Test all function types are JIT compiled."""
        functions = [
            UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend=BackendType.JAX),
            UnifiedForcingFactory.create_perdose(
                t0=0.0, duration=1.0, period=4.0, backend=BackendType.JAX
            ),
            UnifiedForcingFactory.create_ndoses(
                t0_list=[1.0, 5.0], duration=1.0, backend=BackendType.JAX
            ),
            UnifiedForcingFactory.create_zerofunc(backend=BackendType.JAX),
            UnifiedForcingFactory.create_constantfunc(val=5.0, backend=BackendType.JAX),
        ]

        for func in functions:
            func_type_name = type(func).__name__
            assert "Pjit" in func_type_name or "jit" in func_type_name.lower()

    def test_ndoses_jax_broadcasting(self, check_jax_available):
        """Test NDoses uses proper JAX broadcasting with ellipsis pattern."""
        import jax.numpy as jnp

        func = UnifiedForcingFactory.create_ndoses(
            t0_list=[1.0, 5.0, 10.0], duration=1.0, backend=BackendType.JAX
        )

        # Test with scalar inputs
        scalar_result = func(1.5)
        assert isinstance(scalar_result, int | float) or hasattr(scalar_result, "__array__")

        # Test with 1D array inputs
        array_1d = jnp.array([1.5, 5.5, 10.5])
        result_1d = func(array_1d)
        assert len(result_1d) == 3

        # Test with 0D array inputs (from scalar operations)
        array_0d = jnp.array(1.5)  # 0-dimensional array
        result_0d = func(array_0d)
        assert isinstance(result_0d, int | float) or hasattr(result_0d, "__array__")

        # Test with 2D array inputs (tests ellipsis broadcasting t[..., None])
        array_2d = jnp.reshape(array_1d, (3, 1))
        result_2d = func(array_2d)
        assert result_2d.shape == (3, 1), f"Expected shape (3, 1), got {result_2d.shape}"

        # Test consistency: 1D and 2D should give same values (different shapes)
        np.testing.assert_allclose(result_1d, result_2d.flatten(), rtol=1e-6)

    def test_performance_comparison(self, check_jax_available):
        """Test cross-backend consistency with comprehensive array testing."""
        import jax.numpy as jnp

        # Test all function types for consistency
        test_cases = [
            ("onoff", {"t0": 1.0, "t1": 3.0}),
            ("perdose", {"t0": 0.0, "duration": 1.0, "period": 4.0}),
            ("ndoses", {"t0_list": [1.0, 5.0], "duration": 1.0}),
            ("constantfunc", {"val": 2.5}),
            ("zerofunc", {}),
        ]

        for func_name, params in test_cases:
            # Create functions with both backends
            if func_name == "onoff":
                scipy_func = UnifiedForcingFactory.create_onoff(backend=BackendType.SCIPY, **params)
                jax_func = UnifiedForcingFactory.create_onoff(backend=BackendType.JAX, **params)
            elif func_name == "perdose":
                scipy_func = UnifiedForcingFactory.create_perdose(
                    backend=BackendType.SCIPY, **params
                )
                jax_func = UnifiedForcingFactory.create_perdose(backend=BackendType.JAX, **params)
            elif func_name == "ndoses":
                scipy_func = UnifiedForcingFactory.create_ndoses(
                    backend=BackendType.SCIPY, **params
                )
                jax_func = UnifiedForcingFactory.create_ndoses(backend=BackendType.JAX, **params)
            elif func_name == "constantfunc":
                scipy_func = UnifiedForcingFactory.create_constantfunc(
                    backend=BackendType.SCIPY, **params
                )
                jax_func = UnifiedForcingFactory.create_constantfunc(
                    backend=BackendType.JAX, **params
                )
            elif func_name == "zerofunc":
                scipy_func = UnifiedForcingFactory.create_zerofunc(
                    backend=BackendType.SCIPY, **params
                )
                jax_func = UnifiedForcingFactory.create_zerofunc(backend=BackendType.JAX, **params)

            # Warm up JAX function
            _ = jax_func(1.0)

            # Test with various array types and shapes
            test_arrays = [
                np.array([0.5, 2.0, 4.0]),  # Standard 1D array
                np.array([[1.5], [2.5]]),  # 2D array
                np.linspace(0, 5, 100),  # Large array
            ]

            for test_array in test_arrays:
                scipy_result = scipy_func(test_array)
                jax_result = jax_func(jnp.array(test_array))

                # Check results are close
                np.testing.assert_allclose(
                    scipy_result,
                    np.array(jax_result),
                    rtol=1e-3,
                    atol=1e-6,
                    err_msg=f"Backend inconsistency for {func_name} with shape {test_array.shape}",
                )

    def test_jax_backend_methods(self, check_jax_available):
        """Test JAXBackend class methods directly."""
        backend = JAXBackend()

        # Test backend methods
        assert hasattr(backend, "tanh")
        assert hasattr(backend, "floor")
        assert hasattr(backend, "sum")
        assert hasattr(backend, "asarray")
        assert hasattr(backend, "compile_function")

        # Test basic operations
        x = backend.asarray([1.0, 2.0, 3.0])
        tanh_result = backend.tanh(x)
        sum_result = backend.sum(x)

        assert len(tanh_result) == 3
        assert abs(sum_result - 6.0) < 1e-6

    def test_jax_perdose_execution(self, check_jax_available):
        """Test JAX perdose function execution to cover floor method."""
        func = UnifiedForcingFactory.create_perdose(
            t0=0.0, duration=1.0, period=4.0, backend=BackendType.JAX
        )

        # Actually call the function to trigger floor method
        result1 = func(0.5)  # During first dose
        result2 = func(2.0)  # Between doses
        result3 = func(4.5)  # During second dose

        assert float(result1) > 0.95  # During first dose
        assert abs(float(result2)) < 0.01  # Between doses
        assert float(result3) > 0.95  # During second dose

    def test_jax_ode_solver_integration(self, check_jax_available, bodyweight_pk_model_str):
        """Test forcing functions work correctly in actual JAX model ODE solver contexts."""
        import jax.numpy as jnp

        from pymcsimmod.models.jax_model import JaxModel

        # Test Case 1: Basic NDoses integration - simple, short duration
        model = JaxModel(bodyweight_pk_model_str)
        model.assign_forcing_function("dose_in", "NDoses", t0_list=[0.0], duration=0.1)
        model.assign_forcing_function("M_in", "ConstFunc", value=1.0)

        # Run very short simulation to avoid solver issues
        solution1 = model.run_model([0.0, 0.2, 0.5])
        assert solution1.states.shape[1] == 2  # A1, AUC
        assert solution1.states.shape[0] >= 3
        assert np.all(solution1.states >= 0)  # Physically reasonable

        # Test Case 2: ConstFunc + OnOff (simple switching)
        model2 = JaxModel(bodyweight_pk_model_str)
        model2.assign_forcing_function("dose_in", "ConstFunc", value=1.0)  # Constant dosing
        model2.assign_forcing_function("M_in", "OnOff", t0=0.1, t1=0.3)  # Switching bodyweight

        # Run short simulation
        solution2 = model2.run_model([0.0, 0.15, 0.35, 0.5])
        assert solution2.states.shape[1] == 2
        assert solution2.states.shape[0] >= 4

        # Verify that OnOff switching is reflected in M_current
        M_values = solution2.aux_outputs[:, 1]  # M_current from model
        assert len(np.unique(M_values)) > 1, "M_current should show switching behavior"

        # Test Case 3: Interpolate forcing function
        model3 = JaxModel(bodyweight_pk_model_str)
        model3.assign_forcing_function("dose_in", "ConstFunc", value=0.5)
        model3.assign_forcing_function(
            "M_in", "Interpolate", times=[0, 0.25, 0.5], values=[0.8, 1.0, 1.2]
        )

        # Test interpolation works in ODE context
        solution3 = model3.run_model([0.0, 0.125, 0.25, 0.375, 0.5])
        assert solution3.states.shape[1] == 2
        assert len(solution3.times) >= 5

        # Verify interpolated values are reasonable
        M_values3 = solution3.aux_outputs[:, 1]
        assert np.min(M_values3) >= 0.79  # Close to min value
        assert np.max(M_values3) <= 1.21  # Close to max value
        assert np.std(M_values3) > 0.05  # Shows variation

        # Test Case 4: Our optimized NDoses broadcasting in ODE context
        model4 = JaxModel(bodyweight_pk_model_str)
        # Test with two doses to exercise the t[..., None] broadcasting pattern
        model4.assign_forcing_function("dose_in", "NDoses", t0_list=[0.0, 0.3], duration=0.05)
        model4.assign_forcing_function("M_in", "ConstFunc", value=1.0)

        # This specifically tests our ellipsis broadcasting in real ODE context
        solution4 = model4.run_model([0.0, 0.15, 0.3, 0.45, 0.6])
        assert solution4.states.shape[0] >= 5

        # Should see dose effects (non-zero concentrations)
        C_values4 = solution4.aux_outputs[:, 0]
        assert np.any(C_values4 > 0), "Should have concentrations from NDoses"

        # Test Case 5: Cross-backend consistency (very simplified)

        scipy_model = ScipyModel(bodyweight_pk_model_str)
        scipy_model.assign_forcing_function("dose_in", "NDoses", t0_list=[0.0], duration=0.1)
        scipy_model.assign_forcing_function("M_in", "ConstFunc", value=1.0)

        # Compare at simple time points
        test_times = [0.0, 0.2, 0.5]
        jax_solution = model.run_model(test_times)
        scipy_solution = scipy_model.run_model(test_times)

        # Both should produce reasonable solutions
        jax_conc = jax_solution.aux_outputs[: len(test_times), 0]
        scipy_conc = scipy_solution.aux_outputs[: len(test_times), 0]

        assert np.all(jax_conc >= 0) and np.all(scipy_conc >= 0)
        assert np.any(jax_conc > 0) and np.any(scipy_conc > 0)  # Both show dosing effects

        # Test Case 6: JAX array operations work throughout ODE solution
        model5 = JaxModel(bodyweight_pk_model_str)
        model5.assign_forcing_function("dose_in", "OnOff", t0=0.1, t1=0.4)
        model5.assign_forcing_function("M_in", "ConstFunc", value=1.0)

        # Test with JAX arrays as input
        jax_times = jnp.array([0.0, 0.2, 0.5])
        solution5 = model5.run_model(jax_times)

        # Verify the solution is JAX-compatible
        assert hasattr(solution5.states, "__array__") or "jax" in str(type(solution5.states))
        assert solution5.states.shape[1] == 2

        # OnOff should create dosing effect
        C_values5 = solution5.aux_outputs[:, 0]
        assert np.any(C_values5 > 0), "OnOff dosing should create concentrations"


class TestUnifiedForcingFactory:
    """Test factory methods."""

    def test_backend_registration(self):
        """Test custom backend registration."""
        from src.pymcsimmod.forcing.unified import ForcingBackend

        class MockBackend(ForcingBackend):
            def tanh(self, x):
                return x  # Mock implementation

            def floor(self, x):
                return x

            def sum(self, x, axis=None):
                return x

            def asarray(self, x):
                return x

            def compile_function(self, func):
                return func

        # Register mock backend
        UnifiedForcingFactory.register_backend("mock", MockBackend)

        # Should be able to create functions with mock backend
        func = UnifiedForcingFactory.create_zerofunc(backend="mock")
        assert callable(func)

    def test_invalid_backend(self):
        """Test error handling for invalid backends."""
        with pytest.raises(ValueError, match="Unknown backend"):
            UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend="nonexistent")

    def test_backend_availability_check(self):
        """Test backend availability checking."""
        # SciPy backend should always be available
        backend = UnifiedForcingFactory._get_backend("scipy")
        assert isinstance(backend, ScipyBackend)

        # Invalid backend should raise error
        with pytest.raises(ValueError):
            UnifiedForcingFactory._get_backend("invalid")

    def test_cross_backend_consistency(self):
        """Test different backends produce consistent results."""
        # Create same function with different backends
        scipy_func = UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend=BackendType.SCIPY)

        # Test points
        test_points = [0.5, 2.0, 4.0]
        scipy_results = [scipy_func(t) for t in test_points]

        # If JAX is available, test consistency
        try:
            pytest.importorskip("jax")
            jax_func = UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend=BackendType.JAX)
            jax_results = [float(jax_func(t)) for t in test_points]

            # Results should be very close
            for scipy_val, jax_val in zip(scipy_results, jax_results, strict=False):
                assert abs(scipy_val - jax_val) < 1e-6
        except ImportError:
            # JAX not available, skip cross-backend test
            pass


class TestBackendSpecificFeatures:
    """Test backend-specific features."""

    def test_scipy_backend_features(self):
        """Test SciPy backend doesn't compile functions."""

        def dummy_func(x):
            return x * 2

        backend = ScipyBackend()
        compiled_func = backend.compile_function(dummy_func)

        # Should return same function (no compilation)
        assert compiled_func is dummy_func

    def test_jax_backend_features(self):
        """Test JAX backend compiles functions."""
        pytest.importorskip("jax", reason="JAX not available")

        def dummy_func(x):
            return x * 2

        backend = JAXBackend()
        compiled_func = backend.compile_function(dummy_func)

        # Should return different function (JIT compiled)
        assert compiled_func is not dummy_func
        assert (
            "jit" in type(compiled_func).__name__.lower() or "Pjit" in type(compiled_func).__name__
        )

    def test_error_handling_edge_cases(self):
        """Test edge cases like empty t0_list."""
        # Test NDoses with empty list
        func = UnifiedForcingFactory.create_ndoses(
            t0_list=[], duration=1.0, backend=BackendType.SCIPY
        )

        # Should not crash and return 0 (no doses)
        result = func(5.0)
        assert result == 0.0

        # Test with single dose
        func_single = UnifiedForcingFactory.create_ndoses(
            t0_list=[2.0], duration=1.0, backend=BackendType.SCIPY
        )
        assert func_single(2.5) > 0.95  # During dose
        assert abs(func_single(5.0)) < 0.01  # After dose

    def test_unavailable_backend_error(self):
        """Test error handling when backend dependencies are not available.
        ***Will need to be updated if new backends are added.***"""
        # Test TensorFlow backend availability check
        with pytest.raises(ImportError, match="TensorFlow is required for TensorFlow backend"):
            UnifiedForcingFactory.create_zerofunc(backend=BackendType.TENSORFLOW)

        # Test PyTorch backend availability check
        with pytest.raises(ImportError, match="PyTorch is required for PyTorch backend"):
            UnifiedForcingFactory.create_zerofunc(backend=BackendType.PYTORCH)

    def test_interpolated_forcing_via_create_forcing_function(self):
        """Test InterpolatedForcing through create_forcing_function method."""
        # Test with data_dict
        data_dict = {"time": [0, 1, 2], "value": [10, 20, 30]}

        func = UnifiedForcingFactory.create_forcing_function(
            "InterpolatedForcing",
            backend=BackendType.SCIPY,
            data_dict=data_dict,
            interpolation_method="linear",
        )

        assert callable(func)
        # Test interpolation works
        assert func(0.5) == 15.0  # Linear interpolation between 10 and 20
        assert func(1.5) == 25.0  # Linear interpolation between 20 and 30

        # Test with DataFrame
        import pandas as pd

        df = pd.DataFrame({"time": [0, 1, 2], "dose": [5, 15, 25]})

        func_df = UnifiedForcingFactory.create_forcing_function(
            "InterpolatedForcing",
            backend=BackendType.SCIPY,
            dataframe=df,
            time_col="time",
            value_col="dose",
            interpolation_method="linear",
        )

        assert callable(func_df)
        # Test interpolation works
        assert func_df(0.5) == 10.0  # Linear interpolation between 5 and 15
        assert func_df(1.5) == 20.0  # Linear interpolation between 15 and 25
