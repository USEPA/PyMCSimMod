"""Test base forcing functions and factory functionality."""

import pytest
import numpy as np

from src.pymcsimmod.config import BackendType
from src.pymcsimmod.forcing.base import (
    ForcingFunction,
    BackendAwareForcing, 
    OnOffForcing,
    PeriodicForcing,
    MultiDoseForcing,
    create_forcing_function
)


class TestAbstractBaseClasses:
    """Test abstract base classes."""
    
    def test_forcing_function_is_abstract(self):
        """Test that ForcingFunction cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ForcingFunction()
    
    def test_backend_aware_forcing_caching(self):
        """Test caching mechanism in BackendAwareForcing."""
        # Create a concrete subclass for testing
        class TestBackendForcing(BackendAwareForcing):
            def __init__(self):
                super().__init__()
                self.call_count = 0
                
            def _create_backend_function(self, backend: str):
                self.call_count += 1
                return lambda t: t * 2  # Simple test function
                
            def get_switch_times(self, t_start: float, t_end: float) -> list[float]:
                return []
        
        forcing = TestBackendForcing()
        
        # First call should create function
        func1 = forcing.create_function("scipy")
        assert forcing.call_count == 1
        
        # Second call should use cached function
        func2 = forcing.create_function("scipy")
        assert forcing.call_count == 1  # No additional call
        assert func1 is func2  # Same object
        
        # Different backend should create new function
        func3 = forcing.create_function("jax")
        assert forcing.call_count == 2
        assert func3 is not func1  # Different object


class TestOnOffForcing:
    """Test OnOffForcing functionality."""
    
    def test_basic_construction(self):
        """Test basic OnOffForcing construction."""
        forcing = OnOffForcing(t0=1.0, t1=5.0, s=10.0)
        
        assert forcing.t0 == 1.0
        assert forcing.t1 == 5.0
        assert forcing.s == 10.0
        
    def test_default_steepness(self):
        """Test default steepness parameter."""
        forcing = OnOffForcing(t0=1.0, t1=5.0)
        assert forcing.s == 10.0
        
    def test_function_creation_scipy(self):
        """Test function creation for scipy backend."""
        forcing = OnOffForcing(t0=2.0, t1=6.0, s=5.0)
        func = forcing.create_function("scipy")
        
        # Test function behavior
        assert callable(func)
        
        # Test values before, during, and after on period
        assert func(0.0) < 0.1  # Should be near 0 before t0
        assert func(4.0) > 0.9  # Should be near 1 during on period
        assert func(8.0) < 0.1  # Should be near 0 after t1
        
    def test_function_creation_jax(self):
        """Test function creation for JAX backend."""
        pytest.importorskip("jax", reason="JAX not available")
        
        forcing = OnOffForcing(t0=1.0, t1=3.0)
        func = forcing.create_function("jax")
        
        assert callable(func)
        
        # Test basic functionality
        result = func(2.0)  # Should be in "on" period
        assert result > 0.5  # Should be high during on period
        
    def test_function_caching(self):
        """Test that functions are cached properly."""
        forcing = OnOffForcing(t0=1.0, t1=5.0)
        
        func1 = forcing.create_function("scipy")
        func2 = forcing.create_function("scipy")
        
        assert func1 is func2  # Should be same cached object
        
    def test_get_switch_times_full_overlap(self):
        """Test switch times when forcing period fully overlaps simulation."""
        forcing = OnOffForcing(t0=2.0, t1=8.0)
        
        switch_times = forcing.get_switch_times(0.0, 10.0)
        assert switch_times == [2.0, 8.0]
        
    def test_get_switch_times_partial_overlap(self):
        """Test switch times with partial overlap."""
        forcing = OnOffForcing(t0=2.0, t1=8.0)
        
        # Only t0 in range
        switch_times = forcing.get_switch_times(0.0, 5.0)
        assert switch_times == [2.0]
        
        # Only t1 in range
        switch_times = forcing.get_switch_times(5.0, 10.0)
        assert switch_times == [8.0]
        
    def test_get_switch_times_no_overlap(self):
        """Test switch times with no overlap."""
        forcing = OnOffForcing(t0=2.0, t1=8.0)
        
        # Before forcing period
        switch_times = forcing.get_switch_times(0.0, 1.0)
        assert switch_times == []
        
        # After forcing period
        switch_times = forcing.get_switch_times(10.0, 15.0)
        assert switch_times == []


class TestPeriodicForcing:
    """Test PeriodicForcing functionality."""
    
    def test_basic_construction(self):
        """Test basic PeriodicForcing construction."""
        forcing = PeriodicForcing(t0=0.0, duration=1.0, period=24.0, s=5.0)
        
        assert forcing.t0 == 0.0
        assert forcing.duration == 1.0
        assert forcing.period == 24.0
        assert forcing.s == 5.0
        
    def test_default_steepness(self):
        """Test default steepness parameter."""
        forcing = PeriodicForcing(t0=0.0, duration=1.0, period=24.0)
        assert forcing.s == 10.0
        
    def test_function_creation_scipy(self):
        """Test function creation for scipy backend."""
        forcing = PeriodicForcing(t0=0.0, duration=2.0, period=10.0)
        func = forcing.create_function("scipy")
        
        assert callable(func)
        
        # Test periodicity
        # Should be high during dose periods (0-2, 10-12, 20-22, etc.)
        assert func(1.0) > 0.5  # First dose period
        assert func(11.0) > 0.5  # Second dose period
        assert func(21.0) > 0.5  # Third dose period
        
        # Should be low between dose periods
        assert func(5.0) < 0.5  # Between doses
        assert func(15.0) < 0.5  # Between doses
        
    def test_get_switch_times_multiple_periods(self):
        """Test switch times calculation for multiple periods."""
        forcing = PeriodicForcing(t0=0.0, duration=2.0, period=10.0)
        
        switch_times = forcing.get_switch_times(0.0, 25.0)
        
        # Expected: dose starts at 0, 10, 20 and ends at 2, 12, 22
        expected = [0.0, 2.0, 10.0, 12.0, 20.0, 22.0]
        assert switch_times == expected
        
    def test_get_switch_times_partial_periods(self):
        """Test switch times with partial periods in range."""
        forcing = PeriodicForcing(t0=1.0, duration=3.0, period=10.0)
        
        # Range covers partial periods
        switch_times = forcing.get_switch_times(5.0, 15.0)
        
        # Expected: second period starts at 11, ends at 14
        # First period ends at 4 (before range), third period starts at 21 (after range)
        expected = [11.0, 14.0]
        assert switch_times == expected
        
    def test_get_switch_times_no_periods_in_range(self):
        """Test switch times when no periods are in range."""
        forcing = PeriodicForcing(t0=10.0, duration=2.0, period=20.0)
        
        switch_times = forcing.get_switch_times(0.0, 5.0)
        assert switch_times == []


class TestMultiDoseForcing:
    """Test MultiDoseForcing functionality."""
    
    def test_basic_construction(self):
        """Test basic MultiDoseForcing construction."""
        t0_list = [1.0, 5.0, 10.0]
        forcing = MultiDoseForcing(t0_list=t0_list, duration=2.0, s=8.0)
        
        assert forcing.t0_list == t0_list
        assert forcing.duration == 2.0
        assert forcing.s == 8.0
        
    def test_default_steepness(self):
        """Test default steepness parameter."""
        forcing = MultiDoseForcing(t0_list=[0.0, 10.0], duration=1.0)
        assert forcing.s == 10.0
        
    def test_function_creation_scipy(self):
        """Test function creation for scipy backend."""
        t0_list = [2.0, 8.0, 15.0]
        forcing = MultiDoseForcing(t0_list=t0_list, duration=1.5)
        func = forcing.create_function("scipy")
        
        assert callable(func)
        
        # Test function behavior during dose periods
        assert func(2.5) > 0.5  # During first dose (2-3.5)
        assert func(8.5) > 0.5  # During second dose (8-9.5)
        assert func(15.5) > 0.5  # During third dose (15-16.5)
        
        # Test function behavior between doses
        assert func(5.0) < 0.5  # Between first and second dose
        assert func(12.0) < 0.5  # Between second and third dose
        
    def test_get_switch_times_all_doses_in_range(self):
        """Test switch times when all doses are in range."""
        t0_list = [2.0, 6.0, 12.0]
        forcing = MultiDoseForcing(t0_list=t0_list, duration=1.5)
        
        switch_times = forcing.get_switch_times(0.0, 20.0)
        
        # Expected: doses start at 2, 6, 12 and end at 3.5, 7.5, 13.5
        expected = [2.0, 3.5, 6.0, 7.5, 12.0, 13.5]
        assert switch_times == expected
        
    def test_get_switch_times_partial_doses_in_range(self):
        """Test switch times with only some doses in range."""
        t0_list = [1.0, 5.0, 10.0, 20.0]
        forcing = MultiDoseForcing(t0_list=t0_list, duration=2.0)
        
        switch_times = forcing.get_switch_times(4.0, 15.0)
        
        # Only second and third doses should be included
        # Second dose: 5-7, third dose: 10-12
        expected = [5.0, 7.0, 10.0, 12.0]
        assert switch_times == expected
        
    def test_get_switch_times_empty_dose_list(self):
        """Test switch times with empty dose list."""
        forcing = MultiDoseForcing(t0_list=[], duration=1.0)
        
        switch_times = forcing.get_switch_times(0.0, 10.0)
        assert switch_times == []
        
    def test_get_switch_times_dose_ends_outside_range(self):
        """Test switch times when dose starts in range but ends outside."""
        t0_list = [8.0]
        forcing = MultiDoseForcing(t0_list=t0_list, duration=5.0)  # Dose ends at 13.0
        
        switch_times = forcing.get_switch_times(0.0, 10.0)
        
        # Dose starts at 8 (in range) but ends at 13 (outside range)
        expected = [8.0]  # Only start time should be included
        assert switch_times == expected


class TestFactoryFunction:
    """Test create_forcing_function factory."""
    
    def test_create_onoff_forcing(self):
        """Test creating OnOffForcing through factory."""
        forcing = create_forcing_function("onoff", t0=1.0, t1=5.0, s=8.0)
        
        assert isinstance(forcing, OnOffForcing)
        assert forcing.t0 == 1.0
        assert forcing.t1 == 5.0
        assert forcing.s == 8.0
        
    def test_create_periodic_forcing_perdose(self):
        """Test creating PeriodicForcing through factory with 'perdose'."""
        forcing = create_forcing_function("perdose", t0=2.0, duration=1.0, period=12.0)
        
        assert isinstance(forcing, PeriodicForcing)
        assert forcing.t0 == 2.0
        assert forcing.duration == 1.0
        assert forcing.period == 12.0
        
    def test_create_periodic_forcing_periodic(self):
        """Test creating PeriodicForcing through factory with 'periodic'."""
        forcing = create_forcing_function("periodic", t0=0.0, duration=2.0, period=24.0, s=5.0)
        
        assert isinstance(forcing, PeriodicForcing)
        assert forcing.s == 5.0
        
    def test_create_multidose_forcing_ndoses(self):
        """Test creating MultiDoseForcing through factory with 'ndoses'."""
        t0_list = [1.0, 3.0, 7.0]
        forcing = create_forcing_function("ndoses", t0_list=t0_list, duration=0.5)
        
        assert isinstance(forcing, MultiDoseForcing)
        assert forcing.t0_list == t0_list
        assert forcing.duration == 0.5
        
    def test_create_multidose_forcing_multidose(self):
        """Test creating MultiDoseForcing through factory with 'multidose'."""
        t0_list = [2.0, 8.0]
        forcing = create_forcing_function("multidose", t0_list=t0_list, duration=1.5, s=15.0)
        
        assert isinstance(forcing, MultiDoseForcing)
        assert forcing.s == 15.0
        
    def test_create_interpolated_forcing(self):
        """Test creating InterpolatedForcing through factory."""
        times = [0, 1, 2, 5]
        values = [10, 20, 30, 50]
        
        forcing = create_forcing_function("interpolated", times=times, values=values)
        
        from src.pymcsimmod.forcing.interpolated import InterpolatedForcing
        assert isinstance(forcing, InterpolatedForcing)
        np.testing.assert_array_equal(forcing.times, times)
        np.testing.assert_array_equal(forcing.values, values)
        
    def test_create_interpolated_forcing_with_method(self):
        """Test creating InterpolatedForcing with specific interpolation method."""
        forcing = create_forcing_function(
            "interpolated", 
            times=[0, 1, 2], 
            values=[5, 10, 15],
            interpolation_method="cubic"
        )
        
        from src.pymcsimmod.forcing.interpolated import InterpolatedForcing
        assert isinstance(forcing, InterpolatedForcing)
        assert forcing.interpolation_method == "cubic"
        
    def test_unknown_forcing_type_error(self):
        """Test error for unknown forcing type."""
        with pytest.raises(ValueError, match="Unknown forcing type: unknown_type"):
            create_forcing_function("unknown_type")
            
        # Check that error message lists available types
        with pytest.raises(ValueError, match="Available:.*onoff.*perdose.*periodic.*ndoses.*multidose.*interpolated"):
            create_forcing_function("invalid")


class TestForcingIntegration:
    """Test integration between different forcing types and backends."""
    
    def test_all_forcing_types_work_with_scipy(self):
        """Test that all forcing types work with scipy backend."""
        forcings = [
            create_forcing_function("onoff", t0=1.0, t1=3.0),
            create_forcing_function("perdose", t0=0.0, duration=1.0, period=10.0),
            create_forcing_function("ndoses", t0_list=[1.0, 5.0], duration=0.5),
            create_forcing_function("interpolated", times=[0, 2, 4], values=[0, 10, 20])
        ]
        
        for forcing in forcings:
            func = forcing.create_function("scipy")
            assert callable(func)
            
            # Test that function can be called
            result = func(2.0)
            assert isinstance(result, (int, float, np.number, np.ndarray))
            
    def test_all_forcing_types_have_switch_times(self):
        """Test that all forcing types implement get_switch_times."""
        forcings = [
            create_forcing_function("onoff", t0=2.0, t1=6.0),
            create_forcing_function("perdose", t0=1.0, duration=1.0, period=5.0),
            create_forcing_function("ndoses", t0_list=[3.0, 8.0], duration=1.0),
            create_forcing_function("interpolated", times=[0, 5, 10], values=[1, 2, 3])
        ]
        
        for forcing in forcings:
            switch_times = forcing.get_switch_times(0.0, 15.0)
            assert isinstance(switch_times, list)
            assert all(isinstance(t, (int, float)) for t in switch_times)
            
    def test_forcing_function_caching_across_types(self):
        """Test that caching works correctly for different forcing types."""
        forcing_configs = [
            ("onoff", {"t0": 1.0, "t1": 4.0}),
            ("perdose", {"t0": 0.0, "duration": 1.0, "period": 8.0}),
            ("ndoses", {"t0_list": [2.0, 6.0], "duration": 1.0})
        ]
        
        for forcing_type, kwargs in forcing_configs:
            forcing = create_forcing_function(forcing_type, **kwargs)
            
            # Test caching within same backend
            func1 = forcing.create_function("scipy")
            func2 = forcing.create_function("scipy")
            assert func1 is func2
            
            # Test different backends create different functions
            try:
                func3 = forcing.create_function("jax")
                assert func3 is not func1
            except (ImportError, ValueError):
                # JAX might not be available or supported for all forcing types
                pass


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_forcing_function_with_invalid_parameters(self):
        """Test that invalid parameters are handled appropriately."""
        # Test OnOffForcing with invalid order
        forcing = OnOffForcing(t0=5.0, t1=2.0)  # t1 < t0, should still work
        func = forcing.create_function("scipy")
        assert callable(func)
        
    def test_empty_multidose_forcing(self):
        """Test MultiDoseForcing with empty dose list."""
        forcing = MultiDoseForcing(t0_list=[], duration=1.0)
        func = forcing.create_function("scipy")
        
        # Function should work but always return low values
        assert func(5.0) < 0.1
        
    def test_zero_duration_forcing(self):
        """Test forcing functions with zero duration."""
        forcing = MultiDoseForcing(t0_list=[2.0], duration=0.0)
        func = forcing.create_function("scipy")
        
        # Should still create a function
        assert callable(func)
        
    def test_negative_steepness(self):
        """Test forcing functions with negative steepness."""
        forcing = OnOffForcing(t0=1.0, t1=3.0, s=-5.0)  # Unusual but should work
        func = forcing.create_function("scipy")
        assert callable(func)


class TestBackendParametrizedFunctionality:
    """Test all forcing types with all available backends using parametrization."""
    
    def _is_numeric_result(self, result):
        """Check if result is a numeric type (handles JAX arrays)."""
        # Handle regular Python/NumPy types
        if isinstance(result, (int, float, np.number, np.ndarray)):
            return True
        
        # Handle JAX arrays
        try:
            # Check if it's a JAX array by looking for the Array class or array-like behavior
            if hasattr(result, '__array__') or 'jax' in str(type(result)) or 'Array' in str(type(result)):
                return True
        except:
            pass
            
        return False
    # List all available backends
    @pytest.fixture(params=["scipy", "jax"])
    def backend(self, request):
        """Parametrized backend fixture."""
        backend_name = request.param
        if backend_name == "jax":
            pytest.importorskip("jax", reason="JAX not available")
        return backend_name
    
    @pytest.mark.parametrize("forcing_config", [
        ("onoff", {"t0": 1.0, "t1": 4.0, "s": 10.0}),
        ("perdose", {"t0": 0.0, "duration": 1.0, "period": 5.0, "s": 10.0}),
        ("ndoses", {"t0_list": [1.0, 6.0], "duration": 1.0, "s": 10.0}),
    ])
    def test_all_forcing_types_all_backends(self, backend, forcing_config):
        """Test all forcing types work with all backends."""
        forcing_type, params = forcing_config
        
        # Create forcing function
        forcing = create_forcing_function(forcing_type, **params)
        func = forcing.create_function(backend)
        
        # Basic functionality test
        assert callable(func)
        result = func(2.0)
        assert self._is_numeric_result(result)
        
    def test_onoff_forcing_all_backends(self, backend):
        """Test OnOffForcing behavior across all backends."""
        forcing = OnOffForcing(t0=2.0, t1=6.0, s=10.0)
        func = forcing.create_function(backend)
        
        # Test before activation
        assert func(1.0) < 0.1
        
        # Test during activation
        assert func(4.0) > 0.9
        
        # Test after activation
        assert func(8.0) < 0.1
        
    def test_periodic_forcing_all_backends(self, backend):
        """Test PeriodicForcing behavior across all backends."""
        forcing = PeriodicForcing(t0=0.0, duration=2.0, period=10.0, s=10.0)
        func = forcing.create_function(backend)
        
        # Test first dose period
        assert func(1.0) > 0.9  # During first dose
        
        # Test between doses
        assert func(5.0) < 0.1  # Between doses
        
        # Test second dose period
        assert func(11.0) > 0.9  # During second dose
        
    def test_multidose_forcing_all_backends(self, backend):
        """Test MultiDoseForcing behavior across all backends."""
        forcing = MultiDoseForcing(t0_list=[2.0, 8.0, 15.0], duration=1.5, s=10.0)
        func = forcing.create_function(backend)
        
        # Test during doses
        assert func(2.5) > 0.9  # During first dose
        assert func(8.5) > 0.9  # During second dose
        assert func(15.5) > 0.9  # During third dose
        
        # Test between doses
        assert func(5.0) < 0.1  # Between first and second
        assert func(12.0) < 0.1  # Between second and third
        
    def test_switch_times_consistency_all_backends(self, backend):
        """Test that switch times are consistent regardless of backend."""
        forcings = [
            OnOffForcing(t0=2.0, t1=8.0),
            PeriodicForcing(t0=0.0, duration=2.0, period=10.0),
            MultiDoseForcing(t0_list=[2.0, 6.0, 12.0], duration=1.5)
        ]
        
        for forcing in forcings:
            # Switch times should be same regardless of backend
            switch_times = forcing.get_switch_times(0.0, 15.0)
            assert isinstance(switch_times, list)
            assert all(isinstance(t, (int, float)) for t in switch_times)
            
            # Creating function should not affect switch times
            _ = forcing.create_function(backend)
            switch_times_after = forcing.get_switch_times(0.0, 15.0)
            assert switch_times == switch_times_after
            
    def test_array_input_all_backends(self, backend):
        """Test array inputs work with all backends."""
        forcing = OnOffForcing(t0=1.0, t1=3.0)
        func = forcing.create_function(backend)
        
        # Test with array input
        t_array = np.array([0.5, 2.0, 4.0])
        results = func(t_array)
        
        # Should get array results
        if hasattr(results, '__len__'):
            assert len(results) == 3
        
        # Check approximate behavior
        if hasattr(results, '__iter__'):
            results_list = list(results)
            assert results_list[0] < 0.1  # Before
            assert results_list[1] > 0.9  # During  
            assert results_list[2] < 0.1  # After
            
    def test_caching_all_backends(self, backend):
        """Test function caching works with all backends."""
        forcing = OnOffForcing(t0=1.0, t1=3.0)
        
        # First call creates function
        func1 = forcing.create_function(backend)
        
        # Second call should return cached function
        func2 = forcing.create_function(backend) 
        
        assert func1 is func2  # Should be same cached object
        
    @pytest.mark.parametrize("forcing_data", [
        (OnOffForcing, {"t0": 1.0, "t1": 3.0}),
        (PeriodicForcing, {"t0": 0.0, "duration": 1.0, "period": 5.0}),
        (MultiDoseForcing, {"t0_list": [1.0, 4.0], "duration": 1.0})
    ])
    def test_forcing_classes_all_backends(self, backend, forcing_data):
        """Test forcing classes directly with all backends."""
        forcing_class, params = forcing_data
        
        forcing = forcing_class(**params)
        func = forcing.create_function(backend)
        
        # Basic tests
        assert callable(func)
        
        # Test that it produces numeric output
        result = func(2.0)
        assert self._is_numeric_result(result)
        
        # Test that different backends can coexist in cache
        other_backend = "jax" if backend == "scipy" else "scipy"
        try:
            if other_backend == "jax":
                pytest.importorskip("jax", reason="JAX not available")
            func_other = forcing.create_function(other_backend) 
            # Should be different cached functions
            assert func is not func_other
        except (ImportError, ValueError):
            # Other backend not available, skip
            pass