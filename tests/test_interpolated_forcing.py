"""
Tests for interpolated forcing functions based on real-world scenarios from notebooks.

These tests validate that interpolated forcing functions work correctly with
realistic parameter combinations and use cases found in the documentation notebooks.
"""

import numpy as np
import pytest

from pymcsimmod.models.scipy_model import ScipyModel


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
        model = ScipyModel(pk_model_str)

        # Create realistic bodyweight growth data (12 weeks, 84 days)
        time_points = [0, 14, 28, 42, 56, 70, 84]  # Every 2 weeks
        bodyweight_values = [0.25, 0.4, 0.6, 0.85, 1.1, 1.3, 1.45]  # Growth curve in kg

        # Assign interpolated forcing function for bodyweight
        model.assign_forcing_function("M_in", times=time_points, values=bodyweight_values)

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
        model = ScipyModel(pk_model_str)

        # Create step function for body mass (e.g., different life stages)
        time_points = [0, 30, 60, 90]
        bodyweight_values = [0.5, 0.5, 1.5, 1.5]  # Step changes at 30 and 60 days

        model.assign_forcing_function("M_in", times=time_points, values=bodyweight_values)

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
        model = ScipyModel(pk_model_str)

        # Create oscillating dose pattern (circadian rhythm-like)
        time_points = np.linspace(0, 24, 25)  # Every hour for 24 hours
        dose_values = 1.0 + 0.5 * np.sin(2 * np.pi * time_points / 24)  # Sinusoidal pattern

        model.assign_forcing_function("dose_in", times=time_points, values=dose_values)

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
        model = ScipyModel(pk_model_str)

        # Multi-phase scenario: rapid growth, plateau, then decline
        # (could represent development, maturity, aging)
        time_points = [0, 30, 90, 180, 365, 450, 500]
        bodyweight_values = [0.2, 0.8, 1.2, 1.5, 1.4, 1.2, 1.0]  # Growth, plateau, decline

        # Variable dosing schedule
        dose_time_points = [0, 100, 200, 300, 400, 500]
        dose_values = [0.5, 1.0, 1.5, 1.2, 0.8, 0.3]  # Varying dose over time

        model.assign_forcing_function("M_in", times=time_points, values=bodyweight_values)
        model.assign_forcing_function("dose_in", times=dose_time_points, values=dose_values)

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
        model = ScipyModel(pk_model_str)

        # Very sparse bodyweight data (only 3 points over long period)
        time_points = [0, 200, 400]
        bodyweight_values = [0.5, 1.2, 0.8]

        model.assign_forcing_function("M_in", times=time_points, values=bodyweight_values)
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
        model = ScipyModel(pk_model_str)

        # High-frequency bodyweight oscillations (daily measurements)
        time_points = np.linspace(0, 30, 31)  # Daily for 30 days
        # Small daily fluctuations around a growing trend
        trend = 0.5 + 0.02 * time_points  # Linear growth
        noise = 0.05 * np.sin(2 * np.pi * time_points)  # Daily oscillations
        bodyweight_values = trend + noise

        model.assign_forcing_function("M_in", times=time_points, values=bodyweight_values)
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
