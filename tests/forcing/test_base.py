"""Test base forcing functions and factory functionality."""

import pytest

from src.pymcsimmod.forcing.base import (
    BackendAwareForcing,
    ForcingFunction,
    create_forcing_function,
)
from src.pymcsimmod.forcing.unified import UnifiedForcingFactory


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

    def test_backend_aware_forcing_invalidate_cache(self):
        """Test that invalidate_cache clears the cached functions."""

        class SimpleForcing(BackendAwareForcing):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            def _create_backend_function(self, backend: str):
                self.call_count += 1
                return lambda t: t

            def get_switch_times(self, t_start, t_end):
                return []

        f = SimpleForcing()
        f.create_function("scipy")
        assert f.call_count == 1
        f.invalidate_cache()
        f.create_function("scipy")
        assert f.call_count == 2  # Should be recreated after cache invalidation


class TestOnOffForcingViaFactory:
    """Test OnOff forcing functionality via UnifiedForcingFactory."""

    def test_basic_function_creation(self):
        """Test that on-off forcing function is created correctly."""
        func = UnifiedForcingFactory.create_onoff(1.0, 5.0, s=10.0, backend="scipy")
        assert callable(func)

    def test_on_off_values(self):
        """Test that on-off forcing function returns correct values."""
        func = UnifiedForcingFactory.create_onoff(1.0, 5.0, s=10.0, backend="scipy")
        # In the middle should be ~1.0
        assert abs(func(3.0) - 1.0) < 0.05
        # Before t0 should be ~0.0
        assert func(0.0) < 0.05
        # After t1 should be ~0.0
        assert func(6.0) < 0.05

    def test_switch_times_via_extract(self):
        """Test switch time extraction via UnifiedForcingFactory."""
        from src.pymcsimmod.forcing.unified import extract_forcing_switch_times

        ff_spec = {"function": "OnOff", "kwargs": {"t0": 1.0, "t1": 5.0, "s": 10.0}}
        times = extract_forcing_switch_times({"input": ff_spec}, 0.0, 10.0)
        assert 1.0 in times
        assert 5.0 in times

    def test_switch_times_full_overlap(self):
        """Test switch times fully within range."""
        from src.pymcsimmod.forcing.unified import extract_forcing_switch_times

        ff_spec = {"function": "OnOff", "kwargs": {"t0": 2.0, "t1": 8.0, "s": 10.0}}
        times = sorted(extract_forcing_switch_times({"input": ff_spec}, 0.0, 10.0))
        assert times == [2.0, 8.0]

    def test_switch_times_partial_overlap(self):
        """Test switch times partially within range."""
        from src.pymcsimmod.forcing.unified import extract_forcing_switch_times

        ff_spec = {"function": "OnOff", "kwargs": {"t0": 2.0, "t1": 8.0, "s": 10.0}}

        times = sorted(extract_forcing_switch_times({"input": ff_spec}, 0.0, 5.0))
        assert times == [2.0]

        times = sorted(extract_forcing_switch_times({"input": ff_spec}, 5.0, 10.0))
        assert times == [8.0]

    def test_switch_times_no_overlap(self):
        """Test switch times outside range."""
        from src.pymcsimmod.forcing.unified import extract_forcing_switch_times

        ff_spec = {"function": "OnOff", "kwargs": {"t0": 2.0, "t1": 8.0, "s": 10.0}}
        times = extract_forcing_switch_times({"input": ff_spec}, 0.0, 1.0)
        assert times == set()


class TestPeriodicForcingViaFactory:
    """Test PerDose/Periodic forcing functionality via UnifiedForcingFactory."""

    def test_basic_function_creation(self):
        """Test that periodic forcing function is created correctly."""
        func = UnifiedForcingFactory.create_perdose(0.0, 1.0, 10.0, s=10.0, backend="scipy")
        assert callable(func)

    def test_periodic_function_values(self):
        """Test that periodic forcing function returns correct values."""
        func = UnifiedForcingFactory.create_perdose(0.0, 1.0, 10.0, s=10.0, backend="scipy")
        # In the middle of first dose should be ~1.0
        assert abs(func(0.5) - 1.0) < 0.05
        # Between doses should be ~0.0
        assert func(5.0) < 0.05

    def test_switch_times_multiple_periods(self):
        """Test switch time extraction over multiple periods."""
        from src.pymcsimmod.forcing.unified import extract_forcing_switch_times

        ff_spec = {
            "function": "PerDose",
            "kwargs": {"t0": 0.0, "duration": 2.0, "period": 8.0, "s": 10.0},
        }
        times = sorted(extract_forcing_switch_times({"input": ff_spec}, 0.0, 25.0))
        # On times: 0, 8, 16, 24; Off times: 2, 10, 18, 26 (26 outside range)
        expected = [0.0, 2.0, 8.0, 10.0, 16.0, 18.0, 24.0]
        assert times == expected

    def test_switch_times_partial_periods(self):
        """Test switch time extraction over a subset of periods."""
        from src.pymcsimmod.forcing.unified import extract_forcing_switch_times

        ff_spec = {
            "function": "PerDose",
            "kwargs": {"t0": 0.0, "duration": 2.0, "period": 8.0, "s": 10.0},
        }
        times = sorted(extract_forcing_switch_times({"input": ff_spec}, 5.0, 15.0))
        expected = [8.0, 10.0]
        assert times == expected

    def test_switch_times_no_periods_in_range(self):
        """Test switch times when no periods fall in range."""
        from src.pymcsimmod.forcing.unified import extract_forcing_switch_times

        ff_spec = {
            "function": "PerDose",
            "kwargs": {"t0": 10.0, "duration": 2.0, "period": 8.0, "s": 10.0},
        }
        times = extract_forcing_switch_times({"input": ff_spec}, 0.0, 5.0)
        assert len(times) == 0


class TestMultiDoseForcingViaFactory:
    """Test NDoses/MultiDose forcing functionality via UnifiedForcingFactory."""

    def test_basic_function_creation(self):
        """Test that multi-dose forcing function is created correctly."""
        func = UnifiedForcingFactory.create_ndoses([0.0, 10.0, 20.0], 1.0, s=10.0, backend="scipy")
        assert callable(func)

    def test_switch_times_all_doses_in_range(self):
        """Test switch time extraction with all doses in range."""
        from src.pymcsimmod.forcing.unified import extract_forcing_switch_times

        ff_spec = {
            "function": "NDoses",
            "kwargs": {"t0_list": [0.0, 5.0, 10.0], "duration": 2.0, "s": 10.0},
        }
        times = sorted(extract_forcing_switch_times({"input": ff_spec}, 0.0, 20.0))
        expected = [0.0, 2.0, 5.0, 7.0, 10.0, 12.0]
        assert times == expected

    def test_switch_times_partial_doses_in_range(self):
        """Test switch time extraction with partial doses in range."""
        from src.pymcsimmod.forcing.unified import extract_forcing_switch_times

        ff_spec = {
            "function": "NDoses",
            "kwargs": {"t0_list": [0.0, 5.0, 10.0], "duration": 2.0, "s": 10.0},
        }
        times = sorted(extract_forcing_switch_times({"input": ff_spec}, 4.0, 15.0))
        expected = [5.0, 7.0, 10.0, 12.0]
        assert times == expected

    def test_switch_times_empty_dose_list(self):
        """Test switch time extraction with empty dose list."""
        from src.pymcsimmod.forcing.unified import extract_forcing_switch_times

        ff_spec = {
            "function": "NDoses",
            "kwargs": {"t0_list": [], "duration": 2.0, "s": 10.0},
        }
        times = extract_forcing_switch_times({"input": ff_spec}, 0.0, 10.0)
        assert len(times) == 0

    def test_switch_times_dose_ends_outside_range(self):
        """Test switch time extraction when dose end falls outside range."""
        from src.pymcsimmod.forcing.unified import extract_forcing_switch_times

        ff_spec = {
            "function": "NDoses",
            "kwargs": {"t0_list": [0.0, 5.0], "duration": 7.0, "s": 10.0},
        }
        times = sorted(extract_forcing_switch_times({"input": ff_spec}, 0.0, 10.0))
        # Dose 1: on=0 (in range), off=7 (in range)
        # Dose 2: on=5 (in range), off=12 (outside range)
        expected = [0.0, 5.0, 7.0]
        assert times == expected


class TestCreateForcingFunction:
    """Test the factory function create_forcing_function."""

    def test_create_interpolated_forcing(self):
        """Test creating an interpolated forcing function."""
        from src.pymcsimmod.forcing.interpolated import InterpolatedForcing

        forcing = create_forcing_function(
            "interpolated", times=[0, 1, 2, 5, 10], values=[10, 20, 30, 40, 50]
        )
        assert isinstance(forcing, InterpolatedForcing)

    def test_create_interpolated_forcing_alias(self):
        """Test 'interp' alias for interpolated forcing."""
        from src.pymcsimmod.forcing.interpolated import InterpolatedForcing

        forcing = create_forcing_function("interp", times=[0, 1, 2], values=[10, 20, 30])
        assert isinstance(forcing, InterpolatedForcing)

    def test_unknown_forcing_type_raises_error(self):
        """Test that unknown forcing type raises ValueError with guidance."""
        with pytest.raises(ValueError, match="Unknown forcing type"):
            create_forcing_function("unknown_type")

    def test_error_message_includes_guidance(self):
        """Test that error message for old types includes guidance to use assign_forcing_function."""
        with pytest.raises(ValueError, match="assign_forcing_function"):
            create_forcing_function("onoff", t0=0.0, t1=5.0)


class TestAllForcingTypesViaSwitchTimes:
    """Test that all forcing types produce valid switch times via extract_forcing_switch_times."""

    @pytest.mark.parametrize(
        "func_name,kwargs",
        [
            ("OnOff", {"t0": 1.0, "t1": 5.0, "s": 10.0}),
            ("PerDose", {"t0": 0.0, "duration": 1.0, "period": 5.0, "s": 10.0}),
            ("NDoses", {"t0_list": [0.0, 5.0, 10.0], "duration": 1.0, "s": 10.0}),
        ],
    )
    def test_all_forcing_types_have_switch_times(self, func_name, kwargs):
        """Test that all forcing types extract valid switch times."""
        from src.pymcsimmod.forcing.unified import extract_forcing_switch_times

        ff_spec = {"function": func_name, "kwargs": kwargs}
        times = extract_forcing_switch_times({"input": ff_spec}, 0.0, 15.0)
        assert isinstance(times, set)
        assert all(isinstance(t, int | float) for t in times)


class TestSwitchTimesConsistency:
    """Test switch time consistency across backends."""

    @pytest.mark.parametrize("backend", ["scipy", "jax"])
    def test_switch_times_consistency_all_backends(self, backend):
        """Test that switch time extraction is backend-agnostic."""
        from src.pymcsimmod.forcing.unified import extract_forcing_switch_times

        # Switch time extraction doesn't depend on backend (it's from the spec dict)
        ff_spec = {"function": "OnOff", "kwargs": {"t0": 2.0, "t1": 8.0, "s": 10.0}}
        times = sorted(extract_forcing_switch_times({"input": ff_spec}, 0.0, 10.0))
        assert times == [2.0, 8.0]
