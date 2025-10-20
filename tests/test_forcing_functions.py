"""Tests for forcing function implementations."""

import numpy as np
import pytest


class TestScipyForcingFunctions:
    """Tests for scipy forcing function implementations."""

    def test_create_onoff(self):
        """Test scipy OnOff forcing function creation."""
        from pymcsimmod.forcing.scipy_functions import create_onoff

        onoff_func = create_onoff(1.0, 3.0, 10.0)

        # Test at different time points
        assert onoff_func(0.0) < 0.1  # Before start
        assert onoff_func(2.0) > 0.9  # During
        assert onoff_func(4.0) < 0.1  # After end

        # Test vectorized operation
        times = np.array([0.0, 2.0, 4.0])
        results = onoff_func(times)
        assert isinstance(results, np.ndarray)
        assert len(results) == len(times)

    def test_create_perdose(self):
        """Test scipy PerDose forcing function creation."""
        from pymcsimmod.forcing.scipy_functions import create_perdose

        perdose_func = create_perdose(0.0, 1.0, 24.0, 10.0)

        # Test at different time points
        assert perdose_func(0.5) > 0.9  # During first dose
        assert perdose_func(12.0) < 0.1  # Between doses
        assert perdose_func(24.5) > 0.9  # During second dose
        assert perdose_func(48.5) > 0.9  # During third dose

    def test_create_ndoses(self):
        """Test scipy NDoses forcing function creation."""
        from pymcsimmod.forcing.scipy_functions import create_ndoses

        t0_list = [0.0, 8.0, 16.0]
        ndoses_func = create_ndoses(t0_list, 1.0, 10.0)

        # Test at different time points
        assert ndoses_func(0.5) > 0.9  # During first dose
        assert ndoses_func(4.0) < 0.1  # Between doses
        assert ndoses_func(8.5) > 0.9  # During second dose
        assert ndoses_func(16.5) > 0.9  # During third dose

    def test_create_zerofunc(self):
        """Test scipy ZeroFunc creation."""
        from pymcsimmod.forcing.scipy_functions import create_zerofunc

        zero_func = create_zerofunc()

        # Should always return 0
        assert zero_func(0.0) == 0.0
        assert zero_func(10.0) == 0.0
        assert zero_func(-5.0) == 0.0

        # Test with arrays
        times = np.array([1.0, 2.0, 3.0])
        results = zero_func(times)
        np.testing.assert_array_equal(results, np.zeros_like(times))

    def test_constfunc_scipy(self):
        """Test scipy ConstFunc creation and functionality."""
        from pymcsimmod.models.scipy_model import ScipyModel

        # Create a simple test using ScipyModel static method
        const_func = ScipyModel.ConstFunc(value=42.0)

        # Should always return the constant value
        assert const_func(0.0) == 42.0
        assert const_func(10.0) == 42.0
        assert const_func(-5.0) == 42.0

        # Test with arrays
        times = np.array([1.0, 2.0, 3.0])
        results = const_func(times)
        np.testing.assert_array_equal(results, np.full_like(times, 42.0))

        # Test different values
        const_func_2 = ScipyModel.ConstFunc(value=3.14)
        assert const_func_2(1.0) == 3.14

    def test_smoothing_parameter_effect(self):
        """Test that smoothing parameter affects transition sharpness."""
        from pymcsimmod.forcing.scipy_functions import create_onoff

        # Test different smoothing parameters
        onoff_smooth = create_onoff(1.0, 2.0, 1.0)  # Low s = smooth
        onoff_sharp = create_onoff(1.0, 2.0, 100.0)  # High s = sharp

        # At the transition point, smooth should be closer to 0.5
        mid_point = 1.5
        smooth_val = onoff_smooth(mid_point)
        sharp_val = onoff_sharp(mid_point)

        # With low s, the transition should be more gradual
        assert abs(smooth_val - 0.5) < abs(sharp_val - 0.5)


@pytest.mark.skipif(
    not pytest.importorskip("jax", reason="JAX not available"), reason="JAX not available"
)
class TestJaxForcingFunctions:
    """Tests for JAX forcing function implementations."""

    def test_create_onoff_jax(self):
        """Test JAX OnOff forcing function creation."""
        import jax.numpy as jnp

        from pymcsimmod.forcing.jax_functions import create_onoff

        onoff_func = create_onoff(1.0, 3.0, 10.0)

        # Test at different time points
        assert onoff_func(jnp.array(0.0)) < 0.1  # Before start
        assert onoff_func(jnp.array(2.0)) > 0.9  # During
        assert onoff_func(jnp.array(4.0)) < 0.1  # After end

    def test_create_perdose_jax(self):
        """Test JAX PerDose forcing function creation."""
        import jax.numpy as jnp

        from pymcsimmod.forcing.jax_functions import create_perdose

        perdose_func = create_perdose(0.0, 1.0, 24.0, 10.0)

        # Test at different time points
        assert perdose_func(jnp.array(0.5)) > 0.9  # During first dose
        assert perdose_func(jnp.array(12.0)) < 0.1  # Between doses
        assert perdose_func(jnp.array(24.5)) > 0.9  # During second dose

    def test_create_ndoses_jax(self):
        """Test JAX NDoses forcing function creation."""
        import jax.numpy as jnp

        from pymcsimmod.forcing.jax_functions import create_ndoses

        t0_list = [0.0, 8.0, 16.0]
        ndoses_func = create_ndoses(t0_list, 1.0, 10.0)

        # Test at different time points
        assert ndoses_func(jnp.array(0.5)) > 0.9  # During first dose
        assert ndoses_func(jnp.array(4.0)) < 0.1  # Between doses
        assert ndoses_func(jnp.array(8.5)) > 0.9  # During second dose

    def test_create_zerofunc_jax(self):
        """Test JAX ZeroFunc creation."""
        from pymcsimmod.forcing.jax_functions import create_zerofunc

        zero_func = create_zerofunc()

        # Should always return 0
        result = zero_func(1.0)  # JAX functions expect scalars
        assert result == 0.0

    def test_constfunc_jax(self):
        """Test JAX ConstFunc creation and functionality."""
        from pymcsimmod.models.jax_model import EqxModel

        # Create a simple test using EqxModel static method
        const_func = EqxModel.ConstFunc(value=42.0)

        # Should always return the constant value
        assert const_func(0.0) == 42.0
        assert const_func(10.0) == 42.0
        assert const_func(-5.0) == 42.0

        # Test different values
        const_func_2 = EqxModel.ConstFunc(value=3.14)
        assert const_func_2(1.0) == 3.14

    def test_jax_jit_compatibility(self):
        """Test that JAX forcing functions are JIT-compatible."""
        import jax
        import jax.numpy as jnp

        from pymcsimmod.forcing.jax_functions import create_ndoses, create_onoff, create_perdose

        # Test OnOff JIT compatibility
        onoff_func = create_onoff(1.0, 3.0, 10.0)
        jitted_onoff = jax.jit(onoff_func)

        t_test = jnp.array(2.0)
        result1 = onoff_func(t_test)
        result2 = jitted_onoff(t_test)
        np.testing.assert_allclose(result1, result2, rtol=1e-6)

        # Test PerDose JIT compatibility
        perdose_func = create_perdose(0.0, 1.0, 24.0, 10.0)
        jitted_perdose = jax.jit(perdose_func)

        result1 = perdose_func(t_test)
        result2 = jitted_perdose(t_test)
        np.testing.assert_allclose(result1, result2, rtol=1e-6)

        # Test NDoses JIT compatibility
        ndoses_func = create_ndoses([0.0, 24.0], 1.0, 10.0)
        jitted_ndoses = jax.jit(ndoses_func)

        result1 = ndoses_func(t_test)
        result2 = jitted_ndoses(t_test)
        np.testing.assert_allclose(result1, result2, rtol=1e-6)


class TestForcingFunctionBase:
    """Tests for forcing function base classes."""

    def test_onoff_forcing_class(self):
        """Test OnOffForcing class."""
        from pymcsimmod.forcing.base import OnOffForcing

        forcing = OnOffForcing(t0=1.0, t1=3.0, s=10.0)

        # Test scipy backend
        scipy_func = forcing.create_function("scipy")
        assert callable(scipy_func)
        assert scipy_func(2.0) > 0.9  # Should be active

        # Test switch times
        switch_times = forcing.get_switch_times(0.0, 5.0)
        assert 1.0 in switch_times
        assert 3.0 in switch_times

    @pytest.mark.skipif(
        not pytest.importorskip("jax", reason="JAX not available"), reason="JAX not available"
    )
    def test_onoff_forcing_jax_backend(self):
        """Test OnOffForcing with JAX backend."""
        import jax.numpy as jnp

        from pymcsimmod.forcing.base import OnOffForcing

        forcing = OnOffForcing(t0=1.0, t1=3.0, s=10.0)

        # Test JAX backend
        jax_func = forcing.create_function("jax")
        assert callable(jax_func)
        result = jax_func(jnp.array(2.0))
        assert result > 0.9  # Should be active

    def test_periodic_forcing_class(self):
        """Test PeriodicForcing class."""
        from pymcsimmod.forcing.base import PeriodicForcing

        forcing = PeriodicForcing(t0=0.0, duration=1.0, period=24.0, s=10.0)

        # Test scipy backend
        scipy_func = forcing.create_function("scipy")
        assert callable(scipy_func)
        assert scipy_func(0.5) > 0.9  # During first dose
        assert scipy_func(12.0) < 0.1  # Between doses

        # Test switch times
        switch_times = forcing.get_switch_times(0.0, 50.0)
        assert 0.0 in switch_times  # First dose start
        assert 1.0 in switch_times  # First dose end
        assert 24.0 in switch_times  # Second dose start

    def test_multidose_forcing_class(self):
        """Test MultiDoseForcing class."""
        from pymcsimmod.forcing.base import MultiDoseForcing

        t0_list = [0.0, 8.0, 16.0]
        forcing = MultiDoseForcing(t0_list=t0_list, duration=1.0, s=10.0)

        # Test scipy backend
        scipy_func = forcing.create_function("scipy")
        assert callable(scipy_func)
        assert scipy_func(0.5) > 0.9  # During first dose
        assert scipy_func(4.0) < 0.1  # Between doses
        assert scipy_func(8.5) > 0.9  # During second dose

        # Test switch times
        switch_times = forcing.get_switch_times(0.0, 20.0)
        assert 0.0 in switch_times  # First dose start
        assert 1.0 in switch_times  # First dose end
        assert 8.0 in switch_times  # Second dose start
        assert 16.0 in switch_times  # Third dose start

    def test_invalid_backend(self):
        """Test error handling for invalid backends."""
        from pymcsimmod.forcing.base import OnOffForcing

        forcing = OnOffForcing(t0=1.0, t1=3.0, s=10.0)

        with pytest.raises(ValueError, match="Unknown backend"):
            forcing.create_function("invalid_backend")

    def test_switch_times_outside_range(self):
        """Test switch times calculation when events are outside range."""
        from pymcsimmod.forcing.base import OnOffForcing

        forcing = OnOffForcing(t0=10.0, t1=20.0, s=10.0)

        # Test range that doesn't include the forcing function
        switch_times = forcing.get_switch_times(0.0, 5.0)
        assert len(switch_times) == 0

        # Test range that partially includes
        switch_times = forcing.get_switch_times(15.0, 25.0)
        assert 20.0 in switch_times
        assert 10.0 not in switch_times

    def test_forcing_function_parameters(self):
        """Test that forcing function parameters are properly handled."""
        from pymcsimmod.forcing.base import OnOffForcing

        # Test different s values
        forcing_smooth = OnOffForcing(t0=1.0, t1=2.0, s=1.0)
        forcing_sharp = OnOffForcing(t0=1.0, t1=2.0, s=100.0)

        func_smooth = forcing_smooth.create_function("scipy")
        func_sharp = forcing_sharp.create_function("scipy")

        # At the transition point, smooth should be closer to 0.5
        mid_point = 1.5
        smooth_val = func_smooth(mid_point)
        sharp_val = func_sharp(mid_point)

        assert abs(smooth_val - 0.5) < abs(sharp_val - 0.5)


class TestForcingFunctionIntegration:
    """Integration tests for forcing functions with models."""

    def test_forcing_function_with_scipy_model(self):
        """Test forcing functions integrated with ScipyModel."""
        from pymcsimmod.models.scipy_model import ScipyModel

        model_str = """
        States = {
            A
        };

        Inputs = {
            dose
        };

        # Parameters defined outside blocks with default values
        ke = 0.1;

        Initialize {
            A = 0.0;
        }

        Dynamics {
            dt(A) = dose - A * ke;
        }

        End.
        """

        model = ScipyModel(model_str)

        # Set up forcing function
        model.forcing_functions["dose"] = model.OnOff(1.0, 3.0, 10.0)

        times = np.linspace(0, 5, 51)
        result = model.run_model(times)

        # Check that the dose affects the system
        dose_start_idx = np.argmin(np.abs(times - 1.0))
        dose_end_idx = np.argmin(np.abs(times - 3.0))

        # During dosing, A should increase
        assert result.states[dose_end_idx, 0] > result.states[dose_start_idx, 0]

    @pytest.mark.skipif(
        not pytest.importorskip("jax", reason="JAX not available"), reason="JAX not available"
    )
    def test_forcing_function_with_jax_model(self):
        """Test forcing functions integrated with JaxModel."""
        from pymcsimmod.models.jax_model import JaxModel

        model_str = """
        States = {
            A
        };

        Inputs = {
            dose
        };

        # Parameters defined outside blocks with default values
        ke = 0.1;

        Initialize {
            A = 1.0;
        }

        Dynamics {
            dt(A) = dose - A * ke;
        }

        End.
        """

        model = JaxModel(model_str)

        # Set up forcing function using static method
        model.forcing_functions["dose"] = {"function": "ZeroFunc", "args": (), "kwargs": {}}

        times = np.linspace(0, 5, 51)
        result = model.run_model(times)

        # With zero dosing and ke=0.1, A should decay exponentially from 1.0
        assert result.states[0, 0] == 1.0  # Initial value
        assert result.states[-1, 0] < result.states[0, 0]  # Should decay


class TestInterpolatedForcing:
    """Tests for interpolated forcing functions."""

    @pytest.fixture
    def sample_data(self):
        """Sample time-value data for testing."""
        return {"times": [0, 1, 2, 5, 10], "values": [10, 15, 20, 35, 50]}

    @pytest.fixture
    def sample_dataframe(self, sample_data):
        """Sample DataFrame for testing."""
        import pandas as pd

        return pd.DataFrame({"time": sample_data["times"], "bodyweight": sample_data["values"]})

    def test_basic_initialization(self, sample_data):
        """Test basic initialization of InterpolatedForcing."""
        from pymcsimmod.forcing.interpolated import InterpolatedForcing

        forcing = InterpolatedForcing(sample_data["times"], sample_data["values"])

        assert len(forcing.times) == 5
        assert len(forcing.values) == 5
        assert forcing.interpolation_method == "linear"
        assert not forcing.bounds_error
        assert forcing.fill_value == "extrapolate"

    def test_initialization_with_options(self, sample_data):
        """Test initialization with custom options."""
        from pymcsimmod.forcing.interpolated import InterpolatedForcing

        forcing = InterpolatedForcing(
            sample_data["times"],
            sample_data["values"],
            interpolation_method="cubic",
            bounds_error=True,
            fill_value=0.0,
        )

        assert forcing.interpolation_method == "cubic"
        assert forcing.bounds_error
        assert forcing.fill_value == 0.0

    def test_from_dataframe(self, sample_dataframe):
        """Test creating InterpolatedForcing from DataFrame."""
        from pymcsimmod.forcing.interpolated import InterpolatedForcing

        forcing = InterpolatedForcing.from_dataframe(
            sample_dataframe, time_col="time", value_col="bodyweight"
        )

        assert len(forcing.times) == 5
        assert len(forcing.values) == 5
        np.testing.assert_array_equal(forcing.times, [0, 1, 2, 5, 10])
        np.testing.assert_array_equal(forcing.values, [10, 15, 20, 35, 50])

    def test_scipy_function_creation(self, sample_data):
        """Test scipy interpolation function creation."""
        from pymcsimmod.forcing.interpolated import InterpolatedForcing

        forcing = InterpolatedForcing(sample_data["times"], sample_data["values"])
        func = forcing.create_function("scipy")

        # Test exact points
        assert func(0) == 10
        assert func(1) == 15
        assert func(10) == 50

        # Test interpolated point
        result = func(0.5)
        assert 10 < result < 15

    @pytest.mark.skipif(
        not pytest.importorskip("jax", reason="JAX not available"), reason="JAX not available"
    )
    def test_jax_function_creation(self, sample_data):
        """Test JAX interpolation function creation."""
        import jax.numpy as jnp

        from pymcsimmod.forcing.interpolated import InterpolatedForcing

        forcing = InterpolatedForcing(sample_data["times"], sample_data["values"])
        func = forcing.create_function("jax")

        # Test exact points
        assert func(jnp.array(0.0)) == 10
        assert func(jnp.array(1.0)) == 15
        assert func(jnp.array(10.0)) == 50

        # Test interpolated point
        result = func(jnp.array(0.5))
        assert 10 < result < 15

    def test_data_sorting(self):
        """Test that unsorted data gets sorted automatically."""
        from pymcsimmod.forcing.interpolated import InterpolatedForcing

        # Unsorted data
        times = [5, 1, 10, 0, 2]
        values = [35, 15, 50, 10, 20]

        forcing = InterpolatedForcing(times, values)

        # Should be sorted
        np.testing.assert_array_equal(forcing.times, [0, 1, 2, 5, 10])
        np.testing.assert_array_equal(forcing.values, [10, 15, 20, 35, 50])

    def test_duplicate_times_error(self):
        """Test that duplicate times raise an error."""
        from pymcsimmod.forcing.interpolated import InterpolatedForcing

        times = [0, 1, 1, 2]  # Duplicate at t=1
        values = [10, 15, 20, 25]

        with pytest.raises(ValueError, match="Duplicate time points"):
            InterpolatedForcing(times, values)

    def test_switch_times(self, sample_data):
        """Test switch times calculation."""
        from pymcsimmod.forcing.interpolated import InterpolatedForcing

        forcing = InterpolatedForcing(sample_data["times"], sample_data["values"])

        # Full range
        switch_times = forcing.get_switch_times(0, 10)
        expected = [0, 1, 2, 5, 10]
        assert switch_times == expected

        # Partial range
        switch_times = forcing.get_switch_times(1.5, 6)
        expected = [2, 5]
        assert switch_times == expected

    def test_model_integration_scipy(self, sample_data):
        """Test integration with ScipyModel."""
        from pathlib import Path

        from pymcsimmod import ScipyModel

        model_file = Path("tests/data/pk1_dosing.model")
        model = ScipyModel(model_file)

        # Assign interpolated forcing function
        model.assign_forcing_function(
            "BW_input", times=sample_data["times"], values=sample_data["values"]
        )

        # Check that the forcing function was stored correctly
        assert "BW_input" in model.forcing_functions
        assert model.forcing_functions["BW_input"]["function"] == "InterpolatedForcing"

        # Run the model
        times = np.linspace(0, 10, 50)
        solution = model.run_model(times)

        # Basic checks
        assert len(solution.times) >= 50  # Allow for switch times
        assert hasattr(solution, "states")

    @pytest.mark.skipif(
        not pytest.importorskip("jax", reason="JAX not available"), reason="JAX not available"
    )
    def test_model_integration_jax(self, sample_data):
        """Test integration with JaxModel."""
        from pathlib import Path

        from pymcsimmod import JaxModel

        model_file = Path("tests/data/pk1_dosing.model")
        model = JaxModel(model_file)

        # Assign interpolated forcing function
        model.assign_forcing_function(
            "BW_input", times=sample_data["times"], values=sample_data["values"]
        )

        # Check that the forcing function was stored correctly
        assert "BW_input" in model.forcing_functions
        assert model.forcing_functions["BW_input"]["function"] == "InterpolatedForcing"

        # Run the model
        times = np.linspace(0, 10, 50)
        solution = model.run_model(times)

        # Basic checks
        assert len(solution.times) >= 50  # Allow for switch times
        assert hasattr(solution, "states")

    def test_constant_func_integration_scipy(self):
        """Test ConstFunc integration with ScipyModel."""
        from pathlib import Path

        from pymcsimmod import ScipyModel

        model_file = Path("tests/data/pk1_dosing.model")
        model = ScipyModel(model_file)

        # Assign ConstFunc
        model.assign_forcing_function("BW_input", "ConstFunc", value=0.75)

        # Check storage
        assert "BW_input" in model.forcing_functions
        assert model.forcing_functions["BW_input"]["function"] == "ConstFunc"
        assert model.forcing_functions["BW_input"]["kwargs"]["value"] == 0.75

        # Run model
        times = np.linspace(0, 5, 50)
        solution = model.run_model(times)
        assert len(solution.times) >= 50  # Allow for switch times

    @pytest.mark.skipif(
        not pytest.importorskip("jax", reason="JAX not available"), reason="JAX not available"
    )
    def test_constant_func_integration_jax(self):
        """Test ConstFunc integration with JaxModel."""
        from pathlib import Path

        from pymcsimmod import JaxModel

        model_file = Path("tests/data/pk1_dosing.model")
        model = JaxModel(model_file)

        # Assign ConstFunc
        model.assign_forcing_function("BW_input", "ConstFunc", value=1.0)

        # Check storage
        assert "BW_input" in model.forcing_functions
        assert model.forcing_functions["BW_input"]["function"] == "ConstFunc"
        assert model.forcing_functions["BW_input"]["kwargs"]["value"] == 1.0

        # Run model
        times = np.linspace(0, 5, 50)
        solution = model.run_model(times)
        assert len(solution.times) >= 50  # Allow for switch times
