"""Comprehensive tests for the unified forcing function backend system."""

import numpy as np
import pytest

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


class TestScipyForcingFunctions:
    """Test scipy backend forcing functions."""
    
    def test_onoff_creation(self):
        """Test OnOff forcing function creation and behavior."""
        # Test creation
        func = UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, s=10.0, backend="scipy")
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
            "OnOff", backend="scipy", t0=1.0, t1=3.0, s=10.0
        )
        assert callable(func)
        
        # Test behavior
        assert abs(func(0.5)) < 0.01  # Before
        assert func(2.0) > 0.95       # During  
        assert abs(func(4.0)) < 0.01  # After
        
    def test_perdose_creation(self):
        """Test PerDose periodic behavior."""
        func = UnifiedForcingFactory.create_perdose(
            t0=0.0, duration=1.0, period=4.0, s=10.0, backend="scipy"
        )
        assert callable(func)
        
        # Test first dose period
        assert func(0.5) > 0.95   # During first dose
        assert abs(func(2.0)) < 0.01  # Between doses
        
        # Test second dose period  
        assert func(4.5) > 0.95   # During second dose
        assert abs(func(6.0)) < 0.01  # Between doses
        
    def test_ndoses_creation(self):
        """Test NDoses discrete dosing behavior."""
        t0_list = [1.0, 5.0, 10.0]
        func = UnifiedForcingFactory.create_ndoses(
            t0_list=t0_list, duration=1.0, s=10.0, backend="scipy"
        )
        assert callable(func)
        
        # Test individual doses
        assert func(1.5) > 0.95   # During first dose
        assert func(5.5) > 0.95   # During second dose
        assert func(10.5) > 0.95  # During third dose
        
        # Test between doses
        assert abs(func(3.0)) < 0.01  # Between first and second
        assert abs(func(12.0)) < 0.01 # After all doses
        
    def test_zerofunc_creation(self):
        """Test ZeroFunc always returns zero."""
        func = UnifiedForcingFactory.create_zerofunc(backend="scipy")
        assert callable(func)
        
        # Test various time points
        assert func(0.0) == 0.0
        assert func(5.0) == 0.0
        assert func(100.0) == 0.0
        
    def test_constantfunc_creation(self):
        """Test ConstFunc returns constant value."""
        val = 42.5
        func = UnifiedForcingFactory.create_constantfunc(val=val, backend="scipy")
        assert callable(func)
        
        # Test various time points
        assert func(0.0) == val
        assert func(5.0) == val
        assert func(100.0) == val
        
    def test_ndoses_array_inputs(self):
        """Test NDoses with array inputs for scipy backend."""
        func = UnifiedForcingFactory.create_ndoses(
            t0_list=[1.0, 5.0], duration=1.0, backend="scipy"
        )
        
        # Test with numpy array input to trigger array branch
        t_array = np.array([1.5, 3.0, 5.5])
        results = func(t_array)
        
        assert len(results) == 3
        assert results[0] > 0.95   # During first dose
        assert abs(results[1]) < 0.01  # Between doses
        assert results[2] > 0.95   # During second dose
        
    def test_array_inputs(self):
        """Test with numpy array inputs."""
        func = UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend="scipy")
        
        # Test with array input
        t_array = np.array([0.5, 2.0, 4.0])
        results = func(t_array)
        
        assert len(results) == 3
        assert abs(results[0]) < 0.01  # Before
        assert results[1] > 0.95       # During
        assert abs(results[2]) < 0.01  # After
        
    def test_parameter_validation(self):
        """Test missing required parameters raise ValueError."""
        # Test OnOff missing parameters
        with pytest.raises(ValueError, match="OnOff forcing function requires"):
            UnifiedForcingFactory.create_forcing_function("OnOff", backend="scipy")
            
        with pytest.raises(ValueError, match="OnOff forcing function requires"):
            UnifiedForcingFactory.create_forcing_function("OnOff", backend="scipy", t0=1.0)
            
        # Test PerDose missing parameters
        with pytest.raises(ValueError, match="PerDose forcing function requires"):
            UnifiedForcingFactory.create_forcing_function("PerDose", backend="scipy", t0=1.0)
            
        # Test NDoses missing parameters
        with pytest.raises(ValueError, match="NDoses forcing function requires"):
            UnifiedForcingFactory.create_forcing_function("NDoses", backend="scipy")
            
        # Test ConstFunc missing parameters
        with pytest.raises(ValueError, match="ConstFunc forcing function requires"):
            UnifiedForcingFactory.create_forcing_function("ConstFunc", backend="scipy")
            
    def test_unknown_function_type(self):
        """Test error handling for unknown function types."""
        with pytest.raises(ValueError, match="Unknown forcing function type"):
            UnifiedForcingFactory.create_forcing_function("UnknownFunc", backend="scipy")
            
    def test_convenience_functions(self):
        """Test convenience functions maintain compatibility."""
        # Test convenience functions exist and work
        func_onoff = create_onoff(t0=1.0, t1=3.0, backend="scipy")
        func_perdose = create_perdose(t0=0.0, duration=1.0, period=4.0, backend="scipy")
        func_ndoses = create_ndoses(t0_list=[1.0, 5.0], duration=1.0, backend="scipy")
        func_zero = create_zerofunc(backend="scipy")
        func_const = create_constantfunc(val=10.0, backend="scipy")
        
        # All should be callable
        assert all(callable(f) for f in [func_onoff, func_perdose, func_ndoses, func_zero, func_const])
        
        # Test basic functionality
        assert func_onoff(2.0) > 0.95
        assert func_perdose(0.5) > 0.95
        assert func_ndoses(1.5) > 0.95
        assert func_zero(5.0) == 0.0
        assert func_const(5.0) == 10.0
        
    @pytest.mark.parametrize("func_type,params", [
        ("OnOff", {"t0": 1.0, "t1": 3.0}),
        ("PerDose", {"t0": 0.0, "duration": 1.0, "period": 4.0}),
        ("NDoses", {"t0_list": [1.0, 5.0], "duration": 1.0}),
        ("ZeroFunc", {}),
        ("ConstFunc", {"value": 5.0}),
    ])
    def test_parametrized_functions(self, func_type, params):
        """Parametrized test for multiple function types."""
        func = UnifiedForcingFactory.create_forcing_function(
            func_type, backend="scipy", **params
        )
        assert callable(func)
        
        # Test function returns numeric value
        result = func(2.0)
        assert isinstance(result, (int, float, np.number))


class TestJAXForcingFunctions:
    """Test JAX backend forcing functions."""
    
    @pytest.fixture
    def check_jax_available(self):
        """Skip tests if JAX is not available."""
        pytest.importorskip("jax", reason="JAX not available")
        
    def test_jit_compilation(self, check_jax_available):
        """Test JAX functions are JIT compiled."""
        func = UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend="jax")
        
        # Check that function is JIT compiled (should have specific JAX type)
        func_type_name = type(func).__name__
        assert "Pjit" in func_type_name or "jit" in func_type_name.lower()
        
    def test_jax_array_compatibility(self, check_jax_available):
        """Test with JAX arrays."""
        import jax.numpy as jnp
        
        func = UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend="jax")
        
        # Test with JAX array
        t_jax = jnp.array([0.5, 2.0, 4.0])
        results = func(t_jax)
        
        # Results should be JAX array
        assert hasattr(results, '__array__') or 'jax' in str(type(results))
        assert len(results) == 3
        
    def test_jit_traceability(self, check_jax_available):
        """Test functions work in JIT contexts."""
        import jax
        
        func = UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend="jax")
        
        # Should work within another JIT function
        @jax.jit
        def test_wrapper(t):
            return func(t) * 2.0
            
        result = test_wrapper(2.0)
        assert result > 1.8  # Should be close to 2.0
        
    def test_all_function_types_jit_compiled(self, check_jax_available):
        """Test all function types are JIT compiled."""
        functions = [
            UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend="jax"),
            UnifiedForcingFactory.create_perdose(t0=0.0, duration=1.0, period=4.0, backend="jax"),
            UnifiedForcingFactory.create_ndoses(t0_list=[1.0, 5.0], duration=1.0, backend="jax"),
            UnifiedForcingFactory.create_zerofunc(backend="jax"),
            UnifiedForcingFactory.create_constantfunc(val=5.0, backend="jax"),
        ]
        
        for func in functions:
            func_type_name = type(func).__name__
            assert "Pjit" in func_type_name or "jit" in func_type_name.lower()
            
    def test_ndoses_jax_broadcasting(self, check_jax_available):
        """Test NDoses uses proper JAX broadcasting."""
        import jax.numpy as jnp
        
        func = UnifiedForcingFactory.create_ndoses(
            t0_list=[1.0, 5.0, 10.0], duration=1.0, backend="jax"
        )
        
        # Test with both scalar and array inputs
        scalar_result = func(1.5)
        array_result = func(jnp.array([1.5, 5.5, 10.5]))
        
        # Should work without errors
        assert isinstance(scalar_result, (int, float)) or hasattr(scalar_result, '__array__')
        assert len(array_result) == 3
        
    def test_performance_comparison(self, check_jax_available):
        """Test performance after JIT compilation."""
        import time
        import jax.numpy as jnp
        
        # Create functions
        scipy_func = UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend="scipy")
        jax_func = UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend="jax")
        
        # Large array for testing
        t_large = jnp.linspace(0, 10, 10000)
        
        # Warm up JAX function (trigger compilation)
        _ = jax_func(t_large[:100])
        
        # Both should produce similar results (not testing speed, just correctness)
        scipy_result = scipy_func(np.array([0.5, 2.0, 4.0]))
        jax_result = jax_func(jnp.array([0.5, 2.0, 4.0]))
        
        # Check results are close
        np.testing.assert_allclose(scipy_result, np.array(jax_result), rtol=1e-3, atol=1e-6)
        
    def test_jax_backend_methods(self, check_jax_available):
        """Test JAXBackend class methods directly."""
        backend = JAXBackend()
        
        # Test backend methods
        assert hasattr(backend, 'tanh')
        assert hasattr(backend, 'floor') 
        assert hasattr(backend, 'sum')
        assert hasattr(backend, 'asarray')
        assert hasattr(backend, 'compile_function')
        
        # Test basic operations
        x = backend.asarray([1.0, 2.0, 3.0])
        tanh_result = backend.tanh(x)
        sum_result = backend.sum(x)
        
        assert len(tanh_result) == 3
        assert abs(sum_result - 6.0) < 1e-6
        
    def test_jax_perdose_execution(self, check_jax_available):
        """Test JAX perdose function execution to cover floor method."""
        func = UnifiedForcingFactory.create_perdose(
            t0=0.0, duration=1.0, period=4.0, backend="jax"
        )
        
        # Actually call the function to trigger floor method
        result1 = func(0.5)  # During first dose
        result2 = func(2.0)  # Between doses  
        result3 = func(4.5)  # During second dose
        
        assert float(result1) > 0.95   # During first dose
        assert abs(float(result2)) < 0.01  # Between doses
        assert float(result3) > 0.95   # During second dose


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
        scipy_func = UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend="scipy")
        
        # Test points
        test_points = [0.5, 2.0, 4.0]
        scipy_results = [scipy_func(t) for t in test_points]
        
        # If JAX is available, test consistency
        try:
            pytest.importorskip("jax")
            jax_func = UnifiedForcingFactory.create_onoff(t0=1.0, t1=3.0, backend="jax")
            jax_results = [float(jax_func(t)) for t in test_points]
            
            # Results should be very close
            for scipy_val, jax_val in zip(scipy_results, jax_results):
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
        assert "jit" in type(compiled_func).__name__.lower() or "Pjit" in type(compiled_func).__name__
        
    def test_error_handling_edge_cases(self):
        """Test edge cases like empty t0_list."""
        # Test NDoses with empty list
        func = UnifiedForcingFactory.create_ndoses(t0_list=[], duration=1.0, backend="scipy")
        
        # Should not crash and return 0 (no doses)
        result = func(5.0)
        assert result == 0.0
        
        # Test with single dose
        func_single = UnifiedForcingFactory.create_ndoses(t0_list=[2.0], duration=1.0, backend="scipy")
        assert func_single(2.5) > 0.95  # During dose
        assert abs(func_single(5.0)) < 0.01  # After dose
        
    def test_unavailable_backend_error(self):
        """Test error handling when backend dependencies are not available.
           ***Will need to be updated if new backends are added.***"""
        # Test TensorFlow backend availability check
        with pytest.raises(ImportError, match="TensorFlow is required for TensorFlow backend"):
            UnifiedForcingFactory.create_zerofunc(backend="tensorflow")
                
        # Test PyTorch backend availability check
        with pytest.raises(ImportError, match="PyTorch is required for PyTorch backend"):
            UnifiedForcingFactory.create_zerofunc(backend="pytorch")