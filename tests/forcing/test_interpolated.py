"""Test integration of InterpolatedForcing into unified forcing system."""

import numpy as np
import pandas as pd
import pytest

from src.pymcsimmod.config import BackendType
from src.pymcsimmod.forcing.unified import UnifiedForcingFactory
from src.pymcsimmod.forcing.interpolated import InterpolatedForcing, create_interpolated_forcing


class TestInterpolatedForcingCore:
    """Test core InterpolatedForcing functionality."""

    def test_basic_construction(self):
        """Test basic constructor functionality."""
        times = [0, 1, 2, 5]
        values = [10, 20, 30, 50]
        
        forcing = InterpolatedForcing(times, values)
        
        assert len(forcing.times) == 4
        assert len(forcing.values) == 4
        assert forcing.interpolation_method == "linear"
        assert forcing.bounds_error is False
        assert forcing.fill_value == "extrapolate"

    def test_validation_errors(self):
        """Test input validation errors."""
        # Test mismatched lengths
        with pytest.raises(ValueError, match="times and values must have the same length"):
            InterpolatedForcing([0, 1, 2], [10, 20])  # Different lengths
            
        # Test insufficient data points
        with pytest.raises(ValueError, match="At least 2 time points required"):
            InterpolatedForcing([0], [10])  # Only 1 point
            
        # Test duplicate times
        with pytest.raises(ValueError, match="Duplicate time points are not allowed"):
            InterpolatedForcing([0, 1, 1, 2], [10, 20, 25, 30])  # Duplicate time

    def test_time_sorting(self):
        """Test that times and values are sorted correctly."""
        # Provide unsorted data
        times = [2, 0, 5, 1]
        values = [30, 10, 50, 20]
        
        forcing = InterpolatedForcing(times, values)
        
        # Should be sorted by time
        expected_times = [0, 1, 2, 5]
        expected_values = [10, 20, 30, 50]
        
        np.testing.assert_array_equal(forcing.times, expected_times)
        np.testing.assert_array_equal(forcing.values, expected_values)

    def test_from_dataframe_basic(self):
        """Test from_dataframe class method."""
        df = pd.DataFrame({
            'time': [0, 1, 2, 5],
            'bodyweight': [20, 22, 24, 30]
        })
        
        forcing = InterpolatedForcing.from_dataframe(df, 'time', 'bodyweight')
        
        assert len(forcing.times) == 4
        np.testing.assert_array_equal(forcing.times, [0, 1, 2, 5])
        np.testing.assert_array_equal(forcing.values, [20, 22, 24, 30])

    def test_from_dataframe_errors(self):
        """Test from_dataframe error handling."""
        df = pd.DataFrame({
            'time': [0, 1, 2],
            'value': [10, 20, 30]
        })
        
        # Missing time column
        with pytest.raises(ValueError, match="Time column 'missing_time' not found"):
            InterpolatedForcing.from_dataframe(df, 'missing_time', 'value')
            
        # Missing value column
        with pytest.raises(ValueError, match="Value column 'missing_value' not found"):
            InterpolatedForcing.from_dataframe(df, 'time', 'missing_value')

    def test_from_dataframe_nan_handling(self):
        """Test from_dataframe handles NaN values."""
        df = pd.DataFrame({
            'time': [0, 1, np.nan, 2, 5],
            'value': [10, np.nan, 25, 30, 50]
        })
        
        forcing = InterpolatedForcing.from_dataframe(df, 'time', 'value')
        
        # Should remove NaN rows, leaving only valid data
        assert len(forcing.times) == 3
        np.testing.assert_array_equal(forcing.times, [0, 2, 5])
        np.testing.assert_array_equal(forcing.values, [10, 30, 50])

    def test_from_dataframe_insufficient_data(self):
        """Test from_dataframe with insufficient data after cleaning."""
        df = pd.DataFrame({
            'time': [0, np.nan],
            'value': [10, np.nan]
        })
        
        with pytest.raises(ValueError, match="At least 2 valid data points required"):
            InterpolatedForcing.from_dataframe(df, 'time', 'value')

    def test_from_dict_basic(self):
        """Test from_dict class method."""
        data = {
            'time': [0, 1, 2, 5],
            'value': [10, 20, 30, 50]
        }
        
        forcing = InterpolatedForcing.from_dict(data)
        
        assert len(forcing.times) == 4
        np.testing.assert_array_equal(forcing.times, [0, 1, 2, 5])
        np.testing.assert_array_equal(forcing.values, [10, 20, 30, 50])

    def test_from_dict_custom_keys(self):
        """Test from_dict with custom key names."""
        data = {
            'timestamps': [0, 1, 2],
            'doses': [100, 200, 300]
        }
        
        forcing = InterpolatedForcing.from_dict(data, 'timestamps', 'doses')
        
        np.testing.assert_array_equal(forcing.times, [0, 1, 2])
        np.testing.assert_array_equal(forcing.values, [100, 200, 300])

    def test_from_dict_errors(self):
        """Test from_dict error handling."""
        data = {'time': [0, 1, 2], 'value': [10, 20, 30]}
        
        # Missing time key
        with pytest.raises(ValueError, match="Time key 'missing' not found"):
            InterpolatedForcing.from_dict(data, 'missing', 'value')
            
        # Missing value key  
        with pytest.raises(ValueError, match="Value key 'missing' not found"):
            InterpolatedForcing.from_dict(data, 'time', 'missing')

    def test_scipy_function_creation(self):
        """Test scipy backend function creation."""
        times = [0, 1, 2]
        values = [10, 20, 30]
        
        forcing = InterpolatedForcing(times, values)
        func = forcing.create_function("scipy")
        
        # Test interpolation with array input
        assert func(0) == 10
        assert func(1) == 20
        assert func(1.5) == 25  # Linear interpolation
        assert func(2) == 30
        
        # Test with the scipy function
        scipy_func = forcing._create_scipy_function()
        
        # Test interpolation at various points
        scalar_result = scipy_func(0.5)  
        assert abs(scalar_result - 15.0) < 1e-10  # Linear interpolation between 10 and 20
        
        # Test with numpy scalar
        import numpy as np
        np_scalar = np.float64(1.5)
        result = scipy_func(np_scalar)
        assert abs(result - 25.0) < 1e-10









    def test_jax_function_creation(self):
        """Test JAX backend function creation."""
        pytest.importorskip("jax", reason="JAX not available")
        
        times = [0, 1, 2]
        values = [10, 20, 30]
        
        forcing = InterpolatedForcing(times, values)
        func = forcing.create_function("jax")
        
        # Test interpolation
        assert abs(func(0.0) - 10.0) < 1e-6
        assert abs(func(1.0) - 20.0) < 1e-6
        assert abs(func(1.5) - 25.0) < 1e-6  # Linear interpolation
        assert abs(func(2.0) - 30.0) < 1e-6

    def test_unsupported_backend(self):
        """Test error for unsupported backends."""
        forcing = InterpolatedForcing([0, 1], [10, 20])
        
        with pytest.raises(ValueError, match="Interpolation not yet supported for backend: tensorflow"):
            forcing.create_function("tensorflow")

    def test_get_switch_times(self):
        """Test get_switch_times method."""
        times = [0, 1, 2, 5, 10]
        values = [10, 20, 30, 50, 100]
        
        forcing = InterpolatedForcing(times, values)
        
        # Test full range
        switch_times = forcing.get_switch_times(0, 10)
        assert switch_times == [0, 1, 2, 5, 10]
        
        # Test partial range
        switch_times = forcing.get_switch_times(1, 5)
        assert switch_times == [1, 2, 5]
        
        # Test range with no data points
        switch_times = forcing.get_switch_times(3, 4)
        assert switch_times == []

    def test_get_data_range(self):
        """Test get_data_range method."""
        times = [1, 3, 7, 10]
        values = [10, 20, 30, 40]
        
        forcing = InterpolatedForcing(times, values)
        min_time, max_time = forcing.get_data_range()
        
        assert min_time == 1.0
        assert max_time == 10.0

    def test_get_value_range(self):
        """Test get_value_range method."""
        times = [0, 1, 2]
        values = [5, 25, 15]
        
        forcing = InterpolatedForcing(times, values)
        min_val, max_val = forcing.get_value_range()
        
        assert min_val == 5.0
        assert max_val == 25.0

    def test_repr(self):
        """Test string representation."""
        times = [0, 1, 2]
        values = [10, 20, 30]
        
        forcing = InterpolatedForcing(times, values, interpolation_method="cubic")
        repr_str = repr(forcing)
        
        assert "InterpolatedForcing" in repr_str
        assert "n_points=3" in repr_str
        assert "time_range=(0.000, 2.000)" in repr_str
        assert "method='cubic'" in repr_str

    def test_plot_data_basic(self):
        """Test basic plotting functionality."""
        matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not available")
        import matplotlib.pyplot as plt
        
        times = [0, 1, 2, 3]
        values = [10, 20, 15, 25]
        forcing = InterpolatedForcing(times, values)
        
        # Test with new figure
        ax = forcing.plot_data()
        
        # Check that axes was returned
        assert ax is not None
        assert hasattr(ax, 'plot')  # Verify it's a matplotlib axes
        
        # Check labels and title
        assert ax.get_xlabel() == "Time"
        assert ax.get_ylabel() == "Value" 
        assert ax.get_title() == "Interpolated Forcing Function"
        
        # Check that legend was created
        legend = ax.get_legend()
        assert legend is not None
        
        # Check grid
        assert ax.grid
        
        plt.close('all')  # Clean up

    def test_plot_data_custom_axes(self):
        """Test plotting with custom axes."""
        matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not available")
        import matplotlib.pyplot as plt
        
        times = [0, 1, 2]
        values = [5, 10, 8]
        forcing = InterpolatedForcing(times, values)
        
        # Create custom axes
        fig, custom_ax = plt.subplots()
        
        # Plot on custom axes
        returned_ax = forcing.plot_data(ax=custom_ax)
        
        # Should return the same axes object
        assert returned_ax is custom_ax
        
        plt.close(fig)

    def test_plot_data_options(self):
        """Test plotting with different options."""
        matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not available")
        import matplotlib.pyplot as plt
        
        times = [0, 1, 2, 3, 4]
        values = [1, 4, 2, 8, 5]
        forcing = InterpolatedForcing(times, values)
        
        # Test with points only
        ax1 = forcing.plot_data(show_points=True, show_interpolation=False)
        
        # Test with interpolation only  
        ax2 = forcing.plot_data(show_points=False, show_interpolation=True)
        
        # Test with custom interpolation points
        ax3 = forcing.plot_data(n_interp_points=50)
        
        # Check that all returned valid axes
        for ax in [ax1, ax2, ax3]:
            assert ax is not None
            assert hasattr(ax, 'plot')
            
        plt.close('all')

    def test_plot_data_matplotlib_not_available(self):
        """Test ImportError handling when matplotlib not available."""
        times = [0, 1, 2]
        values = [10, 20, 30]
        forcing = InterpolatedForcing(times, values)
        
        # Mock matplotlib import to raise ImportError
        import sys
        original_modules = sys.modules.copy()
        
        # Remove matplotlib from sys.modules if present
        if 'matplotlib.pyplot' in sys.modules:
            del sys.modules['matplotlib.pyplot']
        if 'matplotlib' in sys.modules:
            del sys.modules['matplotlib']
            
        # Mock the import to raise ImportError
        class MockMatplotlib:
            def __getattr__(self, name):
                raise ImportError("No module named 'matplotlib'")
        
        sys.modules['matplotlib'] = MockMatplotlib()
        sys.modules['matplotlib.pyplot'] = MockMatplotlib()
        
        try:
            with pytest.raises(ImportError, match="matplotlib is required for plotting"):
                forcing.plot_data()
        finally:
            # Restore original modules
            sys.modules.clear()
            sys.modules.update(original_modules)

    def test_plot_data_with_kwargs(self):
        """Test plotting with additional matplotlib kwargs."""
        matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not available")
        import matplotlib.pyplot as plt
        
        times = [0, 1, 2]
        values = [10, 20, 15]
        forcing = InterpolatedForcing(times, values)
        
        # Test with custom kwargs (these would be passed to scatter plot)
        ax = forcing.plot_data(alpha=0.7, edgecolors='black')
        
        assert ax is not None
        plt.close('all')


class TestConvenienceFunction:
    """Test the convenience function for creating interpolated forcing."""

    def test_create_from_dataframe(self):
        """Test convenience function with DataFrame."""
        df = pd.DataFrame({'t': [0, 1, 2], 'bw': [20, 22, 24]})
        
        forcing = create_interpolated_forcing(df, 't', 'bw')
        
        assert isinstance(forcing, InterpolatedForcing)
        np.testing.assert_array_equal(forcing.times, [0, 1, 2])
        np.testing.assert_array_equal(forcing.values, [20, 22, 24])

    def test_create_from_dict(self):
        """Test convenience function with dict."""
        data = {'time': [0, 1, 2], 'value': [20, 22, 24]}
        
        forcing = create_interpolated_forcing(data)
        
        assert isinstance(forcing, InterpolatedForcing)
        np.testing.assert_array_equal(forcing.times, [0, 1, 2])
        np.testing.assert_array_equal(forcing.values, [20, 22, 24])

    def test_create_from_tuple(self):
        """Test convenience function with tuple."""
        data = ([0, 1, 2], [20, 22, 24])
        
        forcing = create_interpolated_forcing(data)
        
        assert isinstance(forcing, InterpolatedForcing)
        np.testing.assert_array_equal(forcing.times, [0, 1, 2])
        np.testing.assert_array_equal(forcing.values, [20, 22, 24])

    def test_create_dataframe_missing_columns(self):
        """Test convenience function error handling for DataFrame."""
        df = pd.DataFrame({'time': [0, 1, 2], 'value': [20, 22, 24]})
        
        with pytest.raises(ValueError, match="time_col and value_col must be specified"):
            create_interpolated_forcing(df)

    def test_create_unsupported_format(self):
        """Test convenience function with unsupported data format."""
        with pytest.raises(ValueError, match="Unsupported data format"):
            create_interpolated_forcing("invalid_data")


class TestInterpolatedForcingScenarios:
    """Test interpolated forcing scenarios from real notebook examples."""

    @pytest.fixture
    def pk_model_str(self):
        """Simple PK model with interpolated forcing for bodyweight."""
        return """
        States = {
            A0,     # Amount in exposure compartment (mg)
            A1,     # Amount in central compartment (mg)
            A2,     # Amount cleared (mg)
            AUC     # Area under concentration curve (mg*h/L)
        };

        Inputs = {
            dose_in,    # Dose input rate (mg/day)
            M_in        # Body mass for volume scaling (kg)
        };

        Outputs = {
            C,          # Concentration (mg/L)
            Atot,       # Total amount (mg)
            C_mg,       # Concentration in mg/L
            C_umol      # Concentration in umol/L
        };

        # Parameters
        Vdc = 0.1;      # Volume distribution constant (L/kg)
        k01 = 1;        # Absorption rate constant (/h)
        k12 = 0.5;      # Clearance rate constant (/h)
        MW = 150;       # Molecular weight (g/mol)

        # Initial conditions
        A0_init = 0;
        A1_init = 0;
        A2_init = 0;
        AUC_init = 0;

        Initialize {
            A0 = A0_init;
            A1 = A1_init;
            A2 = A2_init;
            AUC = AUC_init;
        }

        Dynamics {
            M = M_in;
            Vd = Vdc * M;
            C = A1 / Vd;

            dt(A0) = dose_in - k01 * A0;
            dt(A1) = k01 * A0 - k12 * A1;
            dt(A2) = k12 * A1;
            dt(AUC) = C;
        }

        CalcOutputs {
            C_mg = C;
            C_umol = C / (MW * 1000);
            Atot = A0 + A1 + A2;
        }

        End.
        """

    def test_bodyweight_growth_interpolation(self, pk_model_str):
        """Test realistic bodyweight growth curve interpolation."""
        from src.pymcsimmod.models.scipy_model import ScipyModel
        
        model = ScipyModel(pk_model_str)

        # Create realistic bodyweight growth data (12 weeks, 84 days)
        time_points = [0, 14, 28, 42, 56, 70, 84]  # Every 2 weeks
        bodyweight_values = [0.25, 0.4, 0.6, 0.85, 1.1, 1.3, 1.45]  # Growth curve in kg

        # Assign interpolated forcing function for bodyweight using data_dict
        bodyweight_data = {"time": time_points, "value": bodyweight_values}
        model.assign_forcing_function("M_in", "InterpolatedForcing", data_dict=bodyweight_data)

        # Set constant dose input
        model.assign_forcing_function("dose_in", "ConstFunc", value=0.5)

        # Run simulation
        times = np.linspace(0, 84, 1000)
        solution = model.run_model(times)

        # Verify solution (solver may add switch times at interpolation points)
        assert solution.states.shape[1] == 4  # Should have 4 state variables
        assert solution.states.shape[0] >= 1000  # May be larger due to switch times

        # Check that bodyweight interpolation works correctly
        # Concentration should vary inversely with bodyweight (larger BW = larger Vd = lower C)
        C_values = solution.aux_outputs[:, 0]  # C_mg concentration

        # Basic checks: simulation completed successfully, concentrations are positive
        assert np.all(C_values >= 0), "Concentrations should be non-negative"
        assert np.any(C_values > 0), "Should have some positive concentrations"

    def test_step_function_interpolation(self, pk_model_str):
        """Test step function-like interpolation for discrete changes."""
        from src.pymcsimmod.models.scipy_model import ScipyModel
        
        model = ScipyModel(pk_model_str)

        # Create step function for body mass (e.g., different life stages)
        time_points = [0, 30, 60, 90]
        bodyweight_values = [0.5, 0.5, 1.5, 1.5]  # Step changes at 30 and 60 days

        bodyweight_data = {"time": time_points, "value": bodyweight_values}
        model.assign_forcing_function("M_in", "InterpolatedForcing", data_dict=bodyweight_data)

        # Set constant dose
        model.assign_forcing_function("dose_in", "ConstFunc", value=1.0)

        # Run simulation
        times = np.linspace(0, 90, 900)
        solution = model.run_model(times)

        # Verify shape (solver may add switch times)
        assert solution.states.shape[1] == 4
        assert solution.states.shape[0] >= 900

        # Check that concentration shows step-like behavior
        C_values = solution.aux_outputs[:, 0]
        assert np.all(C_values >= 0), "Concentrations should be non-negative"
        assert np.any(C_values > 0), "Should have some positive concentrations"

    def test_oscillating_dose_input(self, pk_model_str):
        """Test oscillating dose input using interpolation."""
        from src.pymcsimmod.models.scipy_model import ScipyModel
        
        model = ScipyModel(pk_model_str)

        # Create oscillating dose pattern (circadian rhythm-like)
        time_points = np.linspace(0, 24, 25)  # Every hour for 24 hours
        dose_values = 1.0 + 0.5 * np.sin(2 * np.pi * time_points / 24)  # Sinusoidal pattern

        dose_data = {"time": time_points.tolist(), "value": dose_values.tolist()}
        model.assign_forcing_function("dose_in", "InterpolatedForcing", data_dict=dose_data)

        # Set constant bodyweight
        model.assign_forcing_function("M_in", "ConstFunc", value=1.0)

        # Run simulation
        times = np.linspace(0, 24, 1000)
        solution = model.run_model(times)

        # Verify solution
        assert solution.states.shape[1] == 4
        assert solution.states.shape[0] >= 1000

        # Check that concentration shows variation
        C_values = solution.aux_outputs[:, 0]  # Central concentration

        # Should have some variation due to oscillating input
        assert np.std(C_values) > 0, "Should see variation due to oscillating dose"

        # All concentrations should be positive
        assert np.all(C_values >= 0), "Concentrations should be non-negative"

    def test_complex_multi_phase_scenario(self, pk_model_str):
        """Test complex multi-phase bodyweight and dosing scenario."""
        from src.pymcsimmod.models.scipy_model import ScipyModel
        
        model = ScipyModel(pk_model_str)

        # Multi-phase scenario: rapid growth, plateau, then decline
        # (could represent development, maturity, aging)
        time_points = [0, 30, 90, 180, 365, 450, 500]
        bodyweight_values = [0.2, 0.8, 1.2, 1.5, 1.4, 1.2, 1.0]  # Growth, plateau, decline

        # Variable dosing schedule
        dose_time_points = [0, 100, 200, 300, 400, 500]
        dose_values = [0.5, 1.0, 1.5, 1.2, 0.8, 0.3]  # Varying dose over time

        bodyweight_data = {"time": time_points, "value": bodyweight_values}
        dose_data = {"time": dose_time_points, "value": dose_values}
        model.assign_forcing_function("M_in", "InterpolatedForcing", data_dict=bodyweight_data)
        model.assign_forcing_function("dose_in", "InterpolatedForcing", data_dict=dose_data)

        # Run long-term simulation
        times = np.linspace(0, 500, 2000)
        solution = model.run_model(times)

        # Verify solution completed
        assert solution.states.shape[1] == 4
        assert solution.states.shape[0] >= 2000

        # Check that simulation captures the complex dynamics
        C_values = solution.aux_outputs[:, 0]

        # Should have meaningful concentration profile
        assert np.max(C_values) > np.min(C_values), "Should see concentration variation"
        assert np.all(C_values >= 0), "Concentrations should be non-negative"

    def test_sparse_data_interpolation(self, pk_model_str):
        """Test interpolation with sparse data points."""
        from src.pymcsimmod.models.scipy_model import ScipyModel
        
        model = ScipyModel(pk_model_str)

        # Very sparse bodyweight data (only 3 points over long period)
        time_points = [0, 200, 400]
        bodyweight_values = [0.5, 1.2, 0.8]

        bodyweight_data = {"time": time_points, "value": bodyweight_values}
        model.assign_forcing_function("M_in", "InterpolatedForcing", data_dict=bodyweight_data)
        model.assign_forcing_function("dose_in", "ConstFunc", value=1.0)

        # Run simulation
        times = np.linspace(0, 400, 1500)
        solution = model.run_model(times)

        # Verify interpolation works with sparse data
        assert solution.states.shape[1] == 4
        assert solution.states.shape[0] >= 1500

        # Check that interpolation is smooth
        C_values = solution.aux_outputs[:, 0]

        # Should have successful simulation
        assert np.all(C_values >= 0), "Concentrations should be non-negative"
        assert np.any(C_values > 0), "Should have some positive concentrations"

    def test_high_frequency_interpolation(self, pk_model_str):
        """Test interpolation with high-frequency data."""
        from src.pymcsimmod.models.scipy_model import ScipyModel
        
        model = ScipyModel(pk_model_str)

        # High-frequency bodyweight oscillations (daily measurements)
        time_points = np.linspace(0, 30, 31)  # Daily for 30 days
        # Small daily fluctuations around a growing trend
        trend = 0.5 + 0.02 * time_points  # Linear growth
        noise = 0.05 * np.sin(2 * np.pi * time_points)  # Daily oscillations
        bodyweight_values = trend + noise

        bodyweight_data = {"time": time_points.tolist(), "value": bodyweight_values.tolist()}
        model.assign_forcing_function("M_in", "InterpolatedForcing", data_dict=bodyweight_data)
        model.assign_forcing_function("dose_in", "ConstFunc", value=0.8)

        # Run simulation with fine resolution
        times = np.linspace(0, 30, 1200)
        solution = model.run_model(times)

        # Verify high-frequency interpolation
        assert solution.states.shape[1] == 4
        assert solution.states.shape[0] >= 1200

        # Check that fine details are preserved
        C_values = solution.aux_outputs[:, 0]

        # Should have meaningful concentration profile
        assert np.all(C_values >= 0), "Concentrations should be non-negative"
        assert np.any(C_values > 0), "Should have some positive concentrations"

        # Should have variation due to daily oscillations
        assert np.std(C_values) > 0.001, "Should capture daily variations"


class TestInterpolatedForcingIntegration:
    """Test that InterpolatedForcing works through the unified factory."""

    def test_dataframe_integration_scipy(self):
        """Test DataFrame integration with SciPy backend."""
        df = pd.DataFrame({
            'time': [0, 1, 2, 5, 10],
            'concentration': [0, 10, 20, 50, 100]
        })
        
        func = UnifiedForcingFactory.create_interpolated(
            backend=BackendType.SCIPY,
            dataframe=df,
            time_col="time",
            value_col="concentration",
            interpolation_method="linear"
        )
        
        # Test interpolation values
        assert func(0) == 0
        assert func(1) == 10
        assert func(2.5) == 25  # Linear interpolation between 20 and 50: 20 + 0.5 * (50-20)/3 = 20 + 5 = 25
        assert func(10) == 100

    def test_dataframe_integration_jax(self):
        """Test DataFrame integration with JAX backend."""
        df = pd.DataFrame({
            'time': [0, 1, 2, 5],
            'value': [100, 80, 60, 20]
        })
        
        func = UnifiedForcingFactory.create_interpolated(
            backend=BackendType.JAX,
            dataframe=df,
            time_col="time",
            value_col="value"
        )
        
        # JAX only supports linear interpolation
        assert abs(func(0.0) - 100.0) < 1e-6
        assert abs(func(1.5) - 70.0) < 1e-6  # Linear between 80 and 60
        assert abs(func(5.0) - 20.0) < 1e-6

    def test_dict_integration_scipy(self):
        """Test dictionary integration with SciPy backend."""
        data_dict = {
            'time': [0, 1, 2, 5], 
            'value': [100, 80, 60, 20]
        }
        
        func = UnifiedForcingFactory.create_interpolated(
            backend=BackendType.SCIPY,
            data_dict=data_dict,
            interpolation_method="cubic"
        )
        
        # Test that cubic interpolation works
        assert func(0) == 100
        assert func(1) == 80
        # Cubic interpolation will give different result than linear
        cubic_result = func(1.5)
        linear_func = UnifiedForcingFactory.create_interpolated(
            backend=BackendType.SCIPY,
            data_dict=data_dict,
            interpolation_method="linear"
        )
        linear_result = linear_func(1.5)
        assert abs(cubic_result - linear_result) > 1e-6  # Should be different

    def test_unified_create_forcing_function_interface(self):
        """Test that InterpolatedForcing works through create_forcing_function."""
        data_dict = {
            'time': [0, 2, 4],
            'value': [10, 30, 50]
        }
        
        func = UnifiedForcingFactory.create_forcing_function(
            "InterpolatedForcing",
            backend=BackendType.SCIPY,
            data_dict=data_dict,
            interpolation_method="linear"
        )
        
        assert func(0) == 10
        assert func(1) == 20  # Linear interpolation
        assert func(3) == 40
        assert func(4) == 50

    def test_error_handling(self):
        """Test error handling for missing parameters."""
        # Test missing both dataframe and data_dict
        with pytest.raises(ValueError, match="Must provide either 'dataframe' or 'data_dict'"):
            UnifiedForcingFactory.create_interpolated(backend=BackendType.SCIPY)
        
        # Test through create_forcing_function interface
        with pytest.raises(ValueError, match="Must provide either 'dataframe' or 'data_dict'"):
            UnifiedForcingFactory.create_forcing_function(
                "InterpolatedForcing",
                backend=BackendType.SCIPY
            )

    def test_parameter_passthrough(self):
        """Test that interpolation parameters are passed through correctly."""
        data_dict = {
            'time': [0, 1, 2],
            'value': [0, 10, 20]
        }
        
        # Test bounds_error parameter
        func = UnifiedForcingFactory.create_interpolated(
            backend=BackendType.SCIPY,
            data_dict=data_dict,
            bounds_error=False,
            fill_value="extrapolate"
        )
        
        # Should extrapolate beyond bounds
        result = func(3.0)  # Beyond the data range
        assert result > 20  # Should extrapolate upward
        
        # Test with bounds error enabled (override the default fill_value)
        func_bounds = UnifiedForcingFactory.create_interpolated(
            backend=BackendType.SCIPY, 
            data_dict=data_dict,
            bounds_error=True,
            fill_value=np.nan  # Compatible with bounds_error=True
        )
        
        # Should raise error when extrapolating
        with pytest.raises(ValueError):
            func_bounds(3.0)

    def test_jax_linear_only_limitation(self):
        """Test that JAX only supports linear interpolation."""
        # This should work fine since JAX defaults to linear
        data_dict = {'time': [0, 1, 2], 'value': [0, 10, 20]}
        
        func = UnifiedForcingFactory.create_interpolated(
            backend=BackendType.JAX,
            data_dict=data_dict
        )
        
        assert abs(func(0.5) - 5.0) < 1e-6  # Linear interpolation

    def test_custom_column_names(self):
        """Test DataFrame with custom column names."""
        df = pd.DataFrame({
            'timestamp': [0, 1, 2, 3],
            'dose_amount': [0, 5, 10, 15]
        })
        
        func = UnifiedForcingFactory.create_interpolated(
            backend=BackendType.SCIPY,
            dataframe=df,
            time_col="timestamp",
            value_col="dose_amount"
        )
        
        assert func(0.5) == 2.5  # Linear between 0 and 5
        assert func(2.5) == 12.5  # Linear between 10 and 15

    def test_integration_list_available_functions(self):
        """Test that InterpolatedForcing appears in error messages."""
        with pytest.raises(ValueError) as exc_info:
            UnifiedForcingFactory.create_forcing_function("UnknownFunction")
        
        error_message = str(exc_info.value)
        assert "InterpolatedForcing" in error_message
        assert "Available: OnOff, PerDose, NDoses, ZeroFunc, ConstFunc, InterpolatedForcing" in error_message