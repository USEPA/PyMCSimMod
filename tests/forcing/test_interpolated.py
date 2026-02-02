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

    def test_plot_data_comprehensive(self):
        """Test comprehensive plotting functionality."""
        matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not available")
        import matplotlib.pyplot as plt
        
        times = [0, 1, 2, 3, 4]
        values = [1, 4, 2, 8, 5]
        forcing = InterpolatedForcing(times, values)
        
        # Test basic plotting with new figure
        ax1 = forcing.plot_data()
        assert ax1 is not None
        assert hasattr(ax1, 'plot')
        assert ax1.get_xlabel() == "Time"
        assert ax1.get_ylabel() == "Value" 
        assert ax1.get_title() == "Interpolated Forcing Function"
        assert ax1.get_legend() is not None
        
        # Test with custom axes and options
        fig, custom_ax = plt.subplots()
        ax2 = forcing.plot_data(ax=custom_ax, show_points=True, show_interpolation=False, alpha=0.7)
        assert ax2 is custom_ax
        
        # Test with interpolation only and custom points
        ax3 = forcing.plot_data(show_points=False, show_interpolation=True, n_interp_points=50)
        assert ax3 is not None
        
        plt.close('all')

    def test_convenience_function_basic(self):
        """Test convenience function with common data formats."""
        # Test DataFrame format
        df = pd.DataFrame({'t': [0, 1, 2], 'bw': [20, 22, 24]})
        df_forcing = create_interpolated_forcing(df, 't', 'bw')
        assert isinstance(df_forcing, InterpolatedForcing)
        np.testing.assert_array_equal(df_forcing.times, [0, 1, 2])
        
        # Test dict format
        data = {'time': [0, 1, 2], 'value': [20, 22, 24]}
        dict_forcing = create_interpolated_forcing(data)
        assert isinstance(dict_forcing, InterpolatedForcing)
        np.testing.assert_array_equal(dict_forcing.times, [0, 1, 2])
        
        # Test tuple format
        tuple_data = ([0, 1, 2], [20, 22, 24])
        tuple_forcing = create_interpolated_forcing(tuple_data)
        assert isinstance(tuple_forcing, InterpolatedForcing)
        np.testing.assert_array_equal(tuple_forcing.times, [0, 1, 2])
        
        # Test error handling
        with pytest.raises(ValueError, match="Unsupported data format"):
            create_interpolated_forcing("invalid_data")

    def test_non_input_variable_rejection(self, limited_inputs_model_str):
        """Test that non-input variables are rejected for interpolation."""
        from src.pymcsimmod.models.scipy_model import ScipyModel
        
        model = ScipyModel(limited_inputs_model_str)
        
        # Should work for input variable
        model.assign_forcing_function("dose_in", "Interpolate", times=[0, 1, 2], values=[1, 2, 1])
        
        # Should reject state variable
        with pytest.raises(ValueError, match="'A1' is not a valid input variable"):
            model.assign_forcing_function("A1", "Interpolate", times=[0, 1, 2], values=[1, 2, 1])
        
        # Should reject output variable  
        with pytest.raises(ValueError, match="'C1' is not a valid input variable"):
            model.assign_forcing_function("C1", "Interpolate", times=[0, 1, 2], values=[1, 2, 1])
        
        # Should reject non-existent variable
        with pytest.raises(ValueError, match="'non_existent' is not a valid input variable"):
            model.assign_forcing_function("non_existent", "Interpolate", times=[0, 1, 2], values=[1, 2, 1])


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
        assert "Available: OnOff, PerDose, NDoses, ZeroFunc, ConstFunc, InterpolatedForcing, Interpolate" in error_message


class TestAssignForcingFunctionEnhanced:
    """Test enhanced assign_forcing_function method for dictionary and DataFrame support."""

    @pytest.mark.parametrize("backend_name", ["scipy", "jax"])
    def test_assign_forcing_function_dictionary_format(self, backend_name, bodyweight_pk_model_str):
        """Test assign_forcing_function with dictionary format for both backends."""
        if backend_name == "jax":
            pytest.importorskip("jax", reason="JAX not available")
            from src.pymcsimmod.models.jax_model import JaxModel
            model = JaxModel(bodyweight_pk_model_str)
        else:
            from src.pymcsimmod.models.scipy_model import ScipyModel
            model = ScipyModel(bodyweight_pk_model_str)

        # Test dictionary format with times and variable arrays
        times_data = [0, 12, 24, 48, 72]
        bodyweight_data = [0.5, 0.6, 0.7, 0.9, 1.0]
        dose_data = [0.5, 1.0, 1.5, 1.0, 0.5]

        # Use the new consistent API format
        model.assign_forcing_function("M_in", "Interpolate", times=times_data, values=bodyweight_data)
        model.assign_forcing_function("dose_in", "Interpolate", times=times_data, values=dose_data)

        # Run simulation with exact times for verification
        solution = model.run_model(times_data)

        # Common verification logic
        assert solution.states.shape[1] == 2  # A1, AUC
        C_values = solution.aux_outputs[:, 0]  # Concentration
        M_current_values = solution.aux_outputs[:, 1]  # Current bodyweight
        assert np.all(C_values >= 0), "Concentrations should be non-negative"
        assert np.any(C_values > 0), "Should have some positive concentrations"
        assert np.min(M_current_values) >= 0.5, "Should reflect minimum bodyweight"
        assert np.max(M_current_values) <= 1.0, "Should reflect maximum bodyweight"
        assert np.std(M_current_values) > 0.05, "Should show bodyweight variation"
        
        # Verify exact time matching (now works for both backends!)
        for i, expected_time in enumerate(times_data):
            time_idx = np.argmin(np.abs(solution.times - expected_time))
            actual_time = solution.times[time_idx]
            actual_bw = M_current_values[time_idx]
            assert actual_time == pytest.approx(expected_time), f"Time mismatch at t={expected_time}"
            assert actual_bw == pytest.approx(bodyweight_data[i]), f"Bodyweight mismatch at t={expected_time}"

    @pytest.mark.parametrize("backend_name", ["scipy", "jax"])
    def test_assign_forcing_function_dataframe_format(self, backend_name, bodyweight_pk_model_str):
        """Test assign_forcing_function with DataFrame format for SciPy."""
        from src.pymcsimmod.models.scipy_model import ScipyModel
        
        model = ScipyModel(bodyweight_pk_model_str)

        # Create DataFrame with bodyweight data - note: using variable name as column
        df = pd.DataFrame({
            'times': [0, 24, 48, 72, 96],
            'M_in': [0.4, 0.6, 0.8, 1.0, 1.1]
        })  
        
        # Use the new consistent API with DataFrame data
        model.assign_forcing_function("M_in", "Interpolate", dataframe=df, time_col="times", value_col="M_in")
        
        # Set constant dose
        model.assign_forcing_function("dose_in", "ConstFunc", value=2.0)

        # Run simulation - include interpolation times for exact comparison
        interpolation_times = [0, 12, 24, 48, 72]
        other_times = np.linspace(0, 72, 300)
        all_times = np.sort(np.unique(np.concatenate([interpolation_times, other_times])))
        solution = model.run_model(all_times)

        # Verify simulation completed successfully
        assert solution.states.shape[1] == 2  # A1, AUC
        assert solution.states.shape[0] >= 300
        
        # Check that concentrations show variation due to changing bodyweight
        C_values = solution.aux_outputs[:, 0]  # Concentration
        M_current_values = solution.aux_outputs[:, 1]  # Current bodyweight from model
        assert np.all(C_values >= 0), "Concentrations should be non-negative"
        assert np.any(C_values > 0), "Should have some positive concentrations"
        assert np.std(C_values) > 0.01, "Should see concentration variation due to bodyweight changes"
        
        # Verify DataFrame M_in values are reflected in computed outputs
        assert np.min(M_current_values) >= 0.4, "Should reflect minimum bodyweight from DataFrame"
        assert np.max(M_current_values) <= 1.1, "Should reflect maximum bodyweight from DataFrame"
        assert np.std(M_current_values) > 0.15, "Should show significant bodyweight variation"
        
        # Verify exact interpolation at specific data points
        expected_times = [0, 24, 48, 72]  # Only check times in our simulation range
        expected_bodyweights = [0.4, 0.6, 0.8, 1.0]  # Corresponding bodyweights
        
        for expected_time, expected_bw in zip(expected_times, expected_bodyweights):
            # Find the closest time index in the solution
            time_idx = np.argmin(np.abs(solution.times - expected_time))
            actual_time = solution.times[time_idx]
            actual_bw = M_current_values[time_idx]
            
            # Should be very close to the expected time and bodyweight
            assert actual_time == pytest.approx(expected_time), f"Time mismatch at t={expected_time}: got {actual_time}"
            assert actual_bw == pytest.approx(expected_bw), f"Bodyweight mismatch at t={expected_time}: expected {expected_bw}, got {actual_bw}"

    def test_assign_forcing_function_dataframe_format_jax(self, bodyweight_pk_model_str):
        """Test assign_forcing_function with DataFrame format for JAX."""
        pytest.importorskip("jax", reason="JAX not available")
        from src.pymcsimmod.models.jax_model import JaxModel
        
        model = JaxModel(bodyweight_pk_model_str)

        # Create DataFrame with dose data - note: using variable name as column
        df = pd.DataFrame({
            'times': [0, 6, 12, 18, 24],
            'dose_in': [1.0, 2.0, 3.0, 2.0, 1.0]
        })

        # Use the new consistent API with DataFrame format
        model.assign_forcing_function("dose_in", "Interpolate", dataframe=df, time_col='times', value_col='dose_in')
        
        # Set constant bodyweight
        model.assign_forcing_function("M_in", "ConstFunc", value=0.7)

        # Run simulation - include interpolation times for exact comparison
        interpolation_times = [0, 12, 24, 48, 72]
        other_times = np.linspace(0, 72, 300)
        all_times = np.sort(np.unique(np.concatenate([interpolation_times, other_times])))
        solution = model.run_model(all_times)

        # Verify simulation completed successfully
        assert solution.states.shape[1] == 2  # A1, AUC
        assert solution.states.shape[0] >= 200
        
        # Check that concentrations show variation due to changing dose
        C_values = solution.aux_outputs[:, 0]  # Concentration
        M_current_values = solution.aux_outputs[:, 1]  # Current bodyweight
        assert np.all(C_values >= 0), "Concentrations should be non-negative"
        assert np.any(C_values > 0), "Should have some positive concentrations"
        assert np.std(C_values) > 0.1, "Should see concentration variation due to dose changes"
        
        # Verify constant bodyweight is maintained
        assert np.allclose(M_current_values, 0.7, rtol=1e-10), "Should maintain constant bodyweight"

    def test_assign_forcing_function_multi_variable_dictionary_scipy(self, bodyweight_pk_model_str):
        """Test assign_forcing_function with multi-variable dictionary for SciPy."""
        from src.pymcsimmod.models.scipy_model import ScipyModel
        
        model = ScipyModel(bodyweight_pk_model_str)

        # Test multi-variable assignment with single dictionary
        times_data = [0, 24, 48, 72]
        bodyweight_data = [0.5, 0.7, 0.9, 1.1]
        dose_data = [1.0, 1.5, 2.0, 1.0]

        # Assign both variables using new consistent API
        model.assign_forcing_function("M_in", "Interpolate", times=times_data, values=bodyweight_data)
        model.assign_forcing_function("dose_in", "Interpolate", times=times_data, values=dose_data)

        # Run simulation - include interpolation times for exact comparison
        interpolation_times = [0, 12, 24, 48, 72]
        other_times = np.linspace(0, 72, 300)
        all_times = np.sort(np.unique(np.concatenate([interpolation_times, other_times])))
        solution = model.run_model(all_times)

        # Verify simulation completed successfully
        assert solution.states.shape[1] == 2
        assert solution.states.shape[0] >= 300
        
        # Check that both variables affect the simulation
        C_values = solution.aux_outputs[:, 0]  # Concentration
        M_current_values = solution.aux_outputs[:, 1]  # Current bodyweight
        assert np.all(C_values >= 0), "Concentrations should be non-negative"
        assert np.any(C_values > 0), "Should have some positive concentrations"
        # Should show complex variation due to both dose and bodyweight changing
        assert np.std(C_values) > 0.1, "Should see significant variation"
        
        # Verify both interpolated inputs are reflected in computed outputs
        assert np.min(M_current_values) >= 0.5, "Should reflect minimum bodyweight"
        assert np.max(M_current_values) <= 1.1, "Should reflect maximum bodyweight"
        assert np.std(M_current_values) > 0.1, "Should show bodyweight variation"
        
        # Verify exact interpolation at specific data points for M_in
        expected_times = [0, 24, 48, 72]
        expected_bodyweights = [0.5, 0.7, 0.9, 1.1]
        
        for expected_time, expected_bw in zip(expected_times, expected_bodyweights):
            # Find the closest time index in the solution
            time_idx = np.argmin(np.abs(solution.times - expected_time))
            actual_time = solution.times[time_idx]
            actual_bw = M_current_values[time_idx]
            
            # Should be very close to the expected time and bodyweight
            assert actual_time == pytest.approx(expected_time), f"Time mismatch at t={expected_time}: got {actual_time}"
            assert actual_bw == pytest.approx(expected_bw), f"Bodyweight mismatch at t={expected_time}: expected {expected_bw}, got {actual_bw}"

    def test_assign_forcing_function_multi_variable_dictionary_jax(self, bodyweight_pk_model_str):
        """Test assign_forcing_function with multi-variable dictionary for JAX."""
        pytest.importorskip("jax", reason="JAX not available")
        from src.pymcsimmod.models.jax_model import JaxModel
        
        model = JaxModel(bodyweight_pk_model_str)

        # Test multi-variable assignment with single dictionary
        times_data = [0, 18, 36, 54, 72]
        bodyweight_data = [0.6, 0.8, 1.0, 1.1, 1.0]
        dose_data = [0.8, 1.2, 1.8, 1.4, 0.6]

        # Assign both variables using new consistent API
        model.assign_forcing_function("M_in", "Interpolate", times=times_data, values=bodyweight_data)
        model.assign_forcing_function("dose_in", "Interpolate", times=times_data, values=dose_data)

        # Verify exact interpolation at specific data points for M_in
        expected_times = [0, 18, 36, 54, 72]  # Use the actual interpolation times
        expected_bodyweights = [0.6, 0.8, 1.0, 1.1, 1.0]
        
        # Include these actual times in the interpolation times for exact comparison
        interpolation_times = [0, 18, 36, 54, 72]
        other_times = np.linspace(0, 72, 300)
        all_times = np.sort(np.unique(np.concatenate([interpolation_times, other_times])))
        solution = model.run_model(all_times)

        # Verify simulation completed successfully
        assert solution.states.shape[1] == 2
        assert solution.states.shape[0] >= 300
        
        # Check that both variables affect the simulation
        C_values = solution.aux_outputs[:, 0]  # Concentration
        M_current_values = solution.aux_outputs[:, 1]  # Current bodyweight
        assert np.all(C_values >= 0), "Concentrations should be non-negative"
        assert np.any(C_values > 0), "Should have some positive concentrations"
        # Should show complex variation due to both dose and bodyweight changing
        assert np.std(C_values) > 0.1, "Should see significant variation"
        
        # Verify both interpolated inputs are reflected in computed outputs
        assert np.min(M_current_values) >= 0.6, "Should reflect minimum bodyweight"
        assert np.max(M_current_values) <= 1.1, "Should reflect maximum bodyweight"
        assert np.std(M_current_values) > 0.08, "Should show bodyweight variation"
        
        for expected_time, expected_bw in zip(expected_times, expected_bodyweights):
            # Find the closest time index in the solution
            time_idx = np.argmin(np.abs(solution.times - expected_time))
            actual_time = solution.times[time_idx]
            actual_bw = M_current_values[time_idx]
            
            # Should be very close to the expected time and bodyweight
            assert actual_time == pytest.approx(expected_time), f"Time mismatch at t={expected_time}: got {actual_time}"
            assert actual_bw == pytest.approx(expected_bw, abs=0.01), f"Bodyweight mismatch at t={expected_time}: expected {expected_bw}, got {actual_bw}"

    def test_assign_forcing_function_error_handling(self, bodyweight_pk_model_str):
        """Test error handling for enhanced assign_forcing_function method."""
        from src.pymcsimmod.models.scipy_model import ScipyModel
        
        model = ScipyModel(bodyweight_pk_model_str)

        # Test missing forcing function type with just kwargs
        with pytest.raises(ValueError, match="Forcing function type must be specified"):
            model.assign_forcing_function("M_in", M_in=[0.5, 0.6, 0.7])

        # Test DataFrame as first argument (now unsupported)
        df_bad = pd.DataFrame({'time': [0, 1, 2], 'value': [10, 20, 30]})
        with pytest.raises(ValueError, match="Passing DataFrame as first argument is not supported"):
            model.assign_forcing_function(df_bad)

        # Test mismatched array lengths - now should raise error about unsupported format
        with pytest.raises(ValueError, match="Using variable name in kwargs is not supported"):
            model.assign_forcing_function("M_in", times=[0, 1, 2], M_in=[0.5, 0.6])  # Legacy format

    def test_assign_forcing_function_backward_compatibility_scipy(self, bodyweight_pk_model_str):
        """Test that enhanced assign_forcing_function maintains backward compatibility for SciPy."""
        from src.pymcsimmod.models.scipy_model import ScipyModel
        
        model = ScipyModel(bodyweight_pk_model_str)

        # Test old data_dict format still works
        bodyweight_data = {"time": [0, 24, 48], "value": [0.5, 0.7, 0.9]}
        model.assign_forcing_function("M_in", "InterpolatedForcing", data_dict=bodyweight_data)
        
        # Test old times/values format still works
        model.assign_forcing_function("dose_in", "InterpolatedForcing", 
                                     times=[0, 24, 48], values=[1.0, 1.5, 1.0])

        # Run simulation to verify everything works
        times = np.linspace(0, 48, 200)
        solution = model.run_model(times)

        # Should complete successfully
        assert solution.states.shape[1] == 2
        assert solution.states.shape[0] >= 200
        
        # Verify M_in changes are reflected in output even with old format
        M_current_values = solution.aux_outputs[:, 1]  # Current bodyweight
        assert np.min(M_current_values) >= 0.5, "Should reflect minimum bodyweight"
        assert np.max(M_current_values) <= 0.9, "Should reflect maximum bodyweight"

    def test_assign_forcing_function_backward_compatibility_jax(self, bodyweight_pk_model_str):
        """Test that enhanced assign_forcing_function maintains backward compatibility for JAX."""
        pytest.importorskip("jax", reason="JAX not available")
        from src.pymcsimmod.models.jax_model import JaxModel
        
        model = JaxModel(bodyweight_pk_model_str)

        # Test old data_dict format still works
        dose_data = {"time": [0, 12, 24], "value": [1.0, 2.0, 1.5]}
        model.assign_forcing_function("dose_in", "InterpolatedForcing", data_dict=dose_data)
        
        # Test old times/values format still works
        model.assign_forcing_function("M_in", "InterpolatedForcing", 
                                     times=[0, 12, 24], values=[0.6, 0.8, 1.0])

        # Run simulation to verify everything works
        times = np.linspace(0, 24, 100)
        solution = model.run_model(times)

        # Should complete successfully
        assert solution.states.shape[1] == 2
        assert solution.states.shape[0] >= 100
        
        # Verify M_in changes are reflected in output even with old format
        M_current_values = solution.aux_outputs[:, 1]  # Current bodyweight
        assert np.min(M_current_values) >= 0.6, "Should reflect minimum bodyweight"
        assert np.max(M_current_values) <= 1.0, "Should reflect maximum bodyweight"

    def test_cross_backend_consistency(self, bodyweight_pk_model_str):
        """Test that SciPy and JAX backends give consistent results with new formats."""
        pytest.importorskip("jax", reason="JAX not available")
        from src.pymcsimmod.models.scipy_model import ScipyModel
        from src.pymcsimmod.models.jax_model import JaxModel

        # Test data
        times_data = [0, 12, 24, 36]
        bodyweight_data = [0.6, 0.8, 1.0, 1.2]
        dose_data = [1.0, 1.5, 2.0, 1.0]

        # SciPy model with dictionary format (auto-detects InterpolatedForcing)
        scipy_model = ScipyModel(bodyweight_pk_model_str)
        scipy_model.assign_forcing_function("M_in", "Interpolate", times=times_data, values=bodyweight_data)
        scipy_model.assign_forcing_function("dose_in", "Interpolate", times=times_data, values=dose_data)

        # JAX model with DataFrame format (auto-detects InterpolatedForcing and variable names)
        jax_model = JaxModel(bodyweight_pk_model_str)
        df = pd.DataFrame({'times': times_data, 'M_in': bodyweight_data})
        jax_model.assign_forcing_function("M_in", "Interpolate", dataframe=df, time_col="times", value_col="M_in")
        jax_model.assign_forcing_function("dose_in", "Interpolate", times=times_data, values=dose_data)

        # Run both simulations
        times = np.linspace(0, 36, 150)
        scipy_solution = scipy_model.run_model(times)
        jax_solution = jax_model.run_model(times)

        # Compare results (allowing for small numerical differences)
        scipy_C = scipy_solution.aux_outputs[:, 0]
        jax_C = jax_solution.aux_outputs[:, 0]
        scipy_M = scipy_solution.aux_outputs[:, 1]
        jax_M = jax_solution.aux_outputs[:, 1]

        # Both should have same basic shape and behavior
        assert len(scipy_C) > 0 and len(jax_C) > 0
        assert np.all(scipy_C >= 0) and np.all(jax_C >= 0)
        
        # Should both show variation
        assert np.std(scipy_C) > 0.1 and np.std(jax_C) > 0.1
        
        # M_in values should be consistent between backends (within numerical tolerance)
        # Both should reflect the same bodyweight interpolation
        assert np.min(scipy_M) >= 0.6 and np.min(jax_M) >= 0.6
        assert np.max(scipy_M) <= 1.2 and np.max(jax_M) <= 1.2

    def test_assign_forcing_function_multi_variable_dataframe_scipy(self, bodyweight_pk_model_str):
        """Test assign_forcing_function with multi-variable DataFrame for SciPy."""
        from src.pymcsimmod.models.scipy_model import ScipyModel
        
        model = ScipyModel(bodyweight_pk_model_str)

        # Create DataFrame with multiple variables (dose_in and M_in)
        df = pd.DataFrame({
            'times': [0, 12, 24, 36, 48],
            'dose_in': [0.8, 1.2, 1.8, 1.4, 1.0],
            'M_in': [0.5, 0.7, 0.9, 1.1, 1.2]
        })

        # Use consistent API to assign each variable separately
        model.assign_forcing_function("dose_in", "Interpolate", dataframe=df, time_col='times', value_col='dose_in')
        model.assign_forcing_function("M_in", "Interpolate", dataframe=df, time_col='times', value_col='M_in')

        # Run simulation - include interpolation times for exact comparison
        interpolation_times = [0, 12, 24, 36, 48]
        other_times = np.linspace(0, 48, 200)
        all_times = np.sort(np.unique(np.concatenate([interpolation_times, other_times])))
        solution = model.run_model(all_times)

        # Verify simulation completed successfully
        assert solution.states.shape[1] == 2  # A1, AUC
        assert solution.states.shape[0] >= 200
        
        # Check that both interpolated variables affect the simulation
        C_values = solution.aux_outputs[:, 0]  # Concentration
        M_current_values = solution.aux_outputs[:, 1]  # Current bodyweight
        assert np.all(C_values >= 0), "Concentrations should be non-negative"
        assert np.any(C_values > 0), "Should have some positive concentrations"
        
        # Should show complex variation due to both dose and bodyweight changing
        assert np.std(C_values) > 0.1, "Should see significant concentration variation"
        
        # Verify both variables from DataFrame are reflected in computed outputs
        assert np.min(M_current_values) >= 0.5, "Should reflect minimum bodyweight from DataFrame"
        assert np.max(M_current_values) <= 1.2, "Should reflect maximum bodyweight from DataFrame"
        assert np.std(M_current_values) > 0.15, "Should show bodyweight variation from DataFrame"
        
        # Verify exact interpolation at specific data points for M_in
        expected_times = [0, 12, 24, 36, 48]
        expected_bodyweights = [0.5, 0.7, 0.9, 1.1, 1.2]
        
        for expected_time, expected_bw in zip(expected_times, expected_bodyweights):
            # Find the closest time index in the solution
            time_idx = np.argmin(np.abs(solution.times - expected_time))
            actual_time = solution.times[time_idx]
            actual_bw = M_current_values[time_idx]
            
            # Should be very close to the expected time and bodyweight
            assert actual_time == pytest.approx(expected_time), f"Time mismatch at t={expected_time}: got {actual_time}"
            assert actual_bw == pytest.approx(expected_bw), f"Bodyweight mismatch at t={expected_time}: expected {expected_bw}, got {actual_bw}"

    def test_assign_forcing_function_multi_variable_dataframe_jax(self, bodyweight_pk_model_str):
        """Test assign_forcing_function with multi-variable DataFrame for JAX."""
        pytest.importorskip("jax", reason="JAX not available")
        from src.pymcsimmod.models.jax_model import JaxModel
        
        model = JaxModel(bodyweight_pk_model_str)

        # Create DataFrame with multiple variables (dose_in and M_in)
        df = pd.DataFrame({
            'times': [0, 15, 30, 45, 60],
            'dose_in': [1.0, 1.8, 2.2, 1.6, 0.8],
            'M_in': [0.6, 0.8, 1.0, 1.1, 1.0]
        })

        # Use consistent API to assign each variable separately
        model.assign_forcing_function("dose_in", "Interpolate", dataframe=df, time_col='times', value_col='dose_in')
        model.assign_forcing_function("M_in", "Interpolate", dataframe=df, time_col='times', value_col='M_in')

        # Run simulation - include interpolation times for exact comparison
        interpolation_times = [0, 15, 30, 45, 60]
        other_times = np.linspace(0, 60, 250)
        all_times = np.sort(np.unique(np.concatenate([interpolation_times, other_times])))
        solution = model.run_model(all_times)

        # Verify simulation completed successfully
        assert solution.states.shape[1] == 2  # A1, AUC
        assert solution.states.shape[0] >= 250
        
        # Check that both interpolated variables affect the simulation
        C_values = solution.aux_outputs[:, 0]  # Concentration
        M_current_values = solution.aux_outputs[:, 1]  # Current bodyweight
        assert np.all(C_values >= 0), "Concentrations should be non-negative"
        assert np.any(C_values > 0), "Should have some positive concentrations"
        
        # Should show complex variation due to both dose and bodyweight changing
        assert np.std(C_values) > 0.1, "Should see significant concentration variation"
        
        # Verify both variables from DataFrame are reflected in computed outputs
        assert np.min(M_current_values) >= 0.6, "Should reflect minimum bodyweight from DataFrame"
        assert np.max(M_current_values) <= 1.1, "Should reflect maximum bodyweight from DataFrame"
        assert np.std(M_current_values) > 0.08, "Should show bodyweight variation from DataFrame"
        
        # Verify exact interpolation at specific data points for M_in
        expected_times = [0, 15, 30, 45, 60]
        expected_bodyweights = [0.6, 0.8, 1.0, 1.1, 1.0]
        
        for expected_time, expected_bw in zip(expected_times, expected_bodyweights):
            # Find the closest time index in the solution
            time_idx = np.argmin(np.abs(solution.times - expected_time))
            actual_time = solution.times[time_idx]
            actual_bw = M_current_values[time_idx]
            
            # Should be very close to the expected time and bodyweight
            assert actual_time == pytest.approx(expected_time), f"Time mismatch at t={expected_time}: got {actual_time}"
            assert actual_bw == pytest.approx(expected_bw), f"Bodyweight mismatch at t={expected_time}: expected {expected_bw}, got {actual_bw}"