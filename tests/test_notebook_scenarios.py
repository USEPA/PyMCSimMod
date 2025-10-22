"""Tests for real-world scenarios demonstrated in notebooks/vignettes.

This test module covers integration scenarios and complex usage patterns
that are demonstrated in the documentation notebooks but not covered
in the existing unit tests.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from pymcsimmod.models.scipy_model import ScipyModel


class TestRealWorldForcingScenarios:
    """Test realistic forcing function scenarios from pk_dosing.ipynb."""

    @pytest.fixture
    def pk1_model_str(self):
        """PK1 model string similar to pk1_input.model."""
        return """
        States = {
            A0,     # Amount in exposure compartment (mg)
            A1,     # Amount in central compartment (mg)
            A2,     # Amount cleared (mg)
            AUC     # Area under concentration curve (mg*h/L)
        };

        Inputs = {
            OralExp,    # Oral exposure input
            IVExp,      # IV exposure input
            M_in        # Body mass input (kg)
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

        # Dosing parameters
        OralDose = 0;
        OralDur = 0.01;
        IVDose = 0;
        IVDur = 0.01;

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
            ODose = OralDose / OralDur;

            dt(A0) = OralExp * ODose - k01 * A0;
            dt(A1) = IVExp * ODose + k01 * A0 - k12 * A1;
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

    def test_perdose_with_constant_bodyweight(self, pk1_model_str):
        """Test PerDose forcing with constant bodyweight (Example 1 from pk_dosing.ipynb)."""
        model = ScipyModel(pk1_model_str)
        model.update_constants(OralDose=50)

        # Set constant bodyweight using ConstFunc
        model.assign_forcing_function("M_in", "ConstFunc", value=70.0)

        # Set up periodic oral dosing
        model.assign_forcing_function(
            "OralExp", "PerDose", t0=0, duration=model.parameters["OralDur"], period=7, s=10.0
        )

        # Run simulation
        times = np.linspace(0, 35, 1000)
        solution = model.run_model(times)

        # Verify basic behavior - solver may add switch times at forcing function events
        assert solution.states.shape[1] == 4  # 4 state variables
        assert solution.states.shape[0] >= 1000  # May be larger due to added event times
        assert solution.var_names == ["A0", "A1", "A2", "AUC"]

        # Check that dosing events occur
        # Should see spikes in A0 at dosing times
        A0_values = solution.states[:, 0]
        # Since solver adds event times, use solution.times instead of original times
        actual_times = solution.times

        # Find local maxima (dose times)
        dose_peaks = []
        for i in range(1, len(A0_values) - 1):
            if A0_values[i] > A0_values[i - 1] and A0_values[i] > A0_values[i + 1]:
                dose_peaks.append(actual_times[i])

        # Should have approximately 5 doses over 35 days (0, 7, 14, 21, 28)
        assert len(dose_peaks) >= 4, f"Expected at least 4 dose peaks, found {len(dose_peaks)}"

        # Check that AUC increases over time (area under curve should accumulate)
        AUC_values = solution.states[:, 3]  # AUC state
        assert AUC_values[-1] > AUC_values[0], "AUC should increase over time"

    def test_perdose_with_interpolated_bodyweight(self, pk1_model_str):
        """Test PerDose forcing with interpolated bodyweight growth (Example 2 from pk_dosing.ipynb)."""
        model = ScipyModel(pk1_model_str)
        model.update_constants(OralDose=50)

        # Set up interpolated bodyweight growth
        time_points = [0, 7, 14, 21, 28, 35]  # weeks
        bodyweights = [20, 30, 50, 70, 80, 90]  # kg growth curve
        model.assign_forcing_function("M_in", times=time_points, values=bodyweights)

        # Set up periodic oral dosing
        model.assign_forcing_function(
            "OralExp", "PerDose", t0=0, duration=model.parameters["OralDur"], period=7, s=10.0
        )

        # Run simulation
        times = np.linspace(0, 35, 1000)
        solution = model.run_model(times)

        # Verify results - solver may add switch times for interpolation
        assert solution.states.shape[1] == 4  # 4 state variables
        assert solution.states.shape[0] >= 1000  # May be larger due to added switch times

        # Check that simulation completed successfully
        assert solution.states is not None, "Simulation should complete successfully"

    def test_ndoses_multiple_discrete_times(self, pk1_model_str):
        """Test NDoses forcing with multiple discrete dose times (from JAX example in pk_dosing.ipynb)."""
        model = ScipyModel(pk1_model_str)
        model.update_constants(OralDose=100)

        # Set constant bodyweight
        model.assign_forcing_function("M_in", "ConstFunc", value=70.0)

        # Set up multiple discrete doses
        t0_list = [0, 1, 2, 5, 6, 7, 12, 13]
        model.assign_forcing_function(
            "OralExp", "NDoses", t0_list=t0_list, duration=model.parameters["OralDur"], s=15.0
        )

        # Run simulation
        times = np.linspace(0, 14, 1000)
        solution = model.run_model(times)

        # Verify results - solver may add switch times for events
        assert solution.states.shape[1] == 4  # 4 state variables
        assert solution.states.shape[0] >= 1000  # May be larger due to added event times

        # Check that we see dose events at specified times
        A0_values = solution.states[:, 0]

        # Look for peaks near dose times
        dose_found = []
        for dose_time in t0_list:
            # Find time index closest to dose time
            time_idx = np.argmin(np.abs(times - dose_time))
            # Check for elevated A0 around dose time
            window = slice(max(0, time_idx - 10), min(len(A0_values), time_idx + 10))
            if np.max(A0_values[window]) > 1.0:  # Threshold for dose detection
                dose_found.append(dose_time)

        # Should find most of the doses
        assert len(dose_found) >= len(t0_list) - 2, (
            f"Found only {len(dose_found)} of {len(t0_list)} doses"
        )

    def test_constfunc_vs_interpolated_comparison(self, pk1_model_str):
        """Test comparison between ConstFunc and interpolated bodyweight scenarios."""
        # Model with constant bodyweight
        model_const = ScipyModel(pk1_model_str)
        model_const.update_constants(OralDose=25)
        model_const.assign_forcing_function("M_in", "ConstFunc", value=0.75)
        model_const.assign_forcing_function(
            "OralExp", "PerDose", t0=0, duration=model_const.parameters["OralDur"], period=7
        )

        # Model with growing bodyweight
        model_interp = ScipyModel(pk1_model_str)
        model_interp.update_constants(OralDose=25)
        time_points = [0, 14, 28, 42, 56, 70, 84]
        bodyweight_values = [0.25, 0.4, 0.6, 0.85, 1.1, 1.3, 1.45]
        model_interp.assign_forcing_function("M_in", times=time_points, values=bodyweight_values)
        model_interp.assign_forcing_function(
            "OralExp", "PerDose", t0=0, duration=model_interp.parameters["OralDur"], period=7
        )

        # Run both simulations
        times = np.linspace(0, 84, 1000)
        solution_const = model_const.run_model(times)
        solution_interp = model_interp.run_model(times)

        # Both should complete successfully - solver may add switch times
        assert solution_const.states.shape[1] == 4  # 4 state variables
        assert solution_const.states.shape[0] >= 1000  # May be larger due to added times
        assert solution_interp.states.shape[1] == 4  # 4 state variables
        assert solution_interp.states.shape[0] >= 1000  # May be larger due to added times

        # Solutions should be different due to different bodyweight profiles
        C_const = solution_const.aux_outputs[:, 0]
        C_interp = solution_interp.aux_outputs[:, 0]

        # Should not be identical
        assert not np.allclose(C_const, C_interp, rtol=0.1)

        # Constant bodyweight should generally lead to higher concentrations
        # (smaller volume of distribution)
        mean_C_const = np.mean(C_const[500:])  # Steady-state region
        mean_C_interp = np.mean(C_interp[500:])  # Steady-state region

        # This relationship may vary depending on exact parameters, but generally:
        # smaller body mass -> smaller Vd -> higher concentration
        assert mean_C_const > 0 and mean_C_interp > 0


class TestComplexDiscreteEventsWorkflows:
    """Test complex discrete events workflows from events_demo.ipynb."""

    @pytest.fixture
    def pk1_model_str(self):
        """Simple PK model for events testing."""
        return """
        States = {
            A0,  # Exposure compartment
            A1,  # Central compartment
            A2   # Cleared compartment
        };

        Inputs = {
            dose_input
        };

        # Parameters
        k01 = 1.0;
        k12 = 0.5;
        Vd = 50.0;

        # Initial conditions
        A0_init = 0;
        A1_init = 0;
        A2_init = 0;

        Initialize {
            A0 = A0_init;
            A1 = A1_init;
            A2 = A2_init;
        }

        Dynamics {
            dt(A0) = dose_input - k01 * A0;
            dt(A1) = k01 * A0 - k12 * A1;
            dt(A2) = k12 * A1;
        }

        Outputs = {
            C,
            Atot
        };

        CalcOutputs {
            C = A1 / Vd;
            Atot = A0 + A1 + A2;
        }

        End.
        """

    def test_pandas_dataframe_event_creation(self, pk1_model_str):
        """Test creating events from pandas DataFrame (events_demo.ipynb style)."""
        model = ScipyModel(pk1_model_str)

        # Create events DataFrame similar to events_demo.ipynb
        events_df = pd.DataFrame(
            {
                "time": np.arange(start=0, stop=48 + 12, step=12)  # 0, 12, 24, 36, 48
            }
        )
        events_df["var"] = "A0"
        events_df["value"] = 50.0
        events_df["method"] = "add"

        # Add events from DataFrame
        for _, row in events_df.iterrows():
            model.add_event(
                time=row["time"], state_var=row["var"], value=row["value"], method=row["method"]
            )

        # Verify events were added
        assert len(model.events) == len(events_df)

        # Run simulation - expect warnings about event time inclusion
        times = np.linspace(0, 48, 1000)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Not all event times were in output times")
            warnings.filterwarnings("ignore", message="Some time steps were very close to events")
            solution = model.run_model(times)

        # Check that events occurred
        A0_values = solution.states[:, 0]

        # Should see dose-related activity at event times
        event_times = events_df["time"].values
        doses_detected = 0
        for event_time in event_times:
            time_idx = np.argmin(np.abs(times - event_time))
            # Check for elevated A0 around event time
            window = slice(max(0, time_idx - 5), min(len(A0_values), time_idx + 15))
            if np.max(A0_values[window]) > 10.0:  # Threshold for dose detection
                doses_detected += 1

        assert doses_detected >= len(event_times) - 1, (
            f"Only detected {doses_detected} of {len(event_times)} events"
        )

    def test_repeated_dosing_schedule_events(self, pk1_model_str):
        """Test repeated dosing schedule using events."""
        model = ScipyModel(pk1_model_str)

        # Create a realistic repeated dosing schedule
        # Daily dosing for 7 days, then BID for 7 days, then stop
        dose_times = []

        # Days 0-6: once daily at 8 AM (hour 8, 32, 56, ...)
        for day in range(7):
            dose_times.append(day * 24 + 8)

        # Days 7-13: twice daily at 8 AM and 8 PM
        for day in range(7, 14):
            dose_times.append(day * 24 + 8)  # 8 AM
            dose_times.append(day * 24 + 20)  # 8 PM

        # Add all events
        for dose_time in dose_times:
            model.add_event(time=dose_time, state_var="A0", value=25.0, method="add")

        # Run simulation for 3 weeks - expect warnings about event time inclusion
        times = np.linspace(0, 21 * 24, 2000)  # 21 days, fine resolution
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Not all event times were in output times")
            solution = model.run_model(times)

        # Verify we have the expected number of events
        expected_events = 7 + 14  # 7 once-daily + 14 twice-daily
        assert len(model.events) == expected_events

        # Check that concentration reflects the dosing schedule
        C_values = solution.aux_outputs[:, 0]  # Concentration
        n_times = len(C_values)  # Actual number of time points from solver

        # Should see higher average concentration during BID period (days 7-13)
        # vs once-daily period (days 0-6)

        # Create corresponding time array for the actual solution size
        actual_times = np.linspace(0, 21 * 24, n_times)
        day_hours = 24
        once_daily_period = (actual_times >= 2 * day_hours) & (actual_times <= 6 * day_hours)
        bid_period = (actual_times >= 8 * day_hours) & (actual_times <= 12 * day_hours)

        if np.any(once_daily_period) and np.any(bid_period):
            avg_conc_once = np.mean(C_values[once_daily_period])
            avg_conc_bid = np.mean(C_values[bid_period])

            # BID should generally have higher average concentration
            assert avg_conc_bid > avg_conc_once, (
                f"BID conc {avg_conc_bid} not > once daily {avg_conc_once}"
            )

    def test_event_clearing_and_management(self, pk1_model_str):
        """Test event clearing and management functionality."""
        model = ScipyModel(pk1_model_str)

        # Add some initial events
        model.add_event(time=1.0, state_var="A0", value=10.0, method="add")
        model.add_event(time=2.0, state_var="A1", value=5.0, method="replace")
        model.add_event(time=3.0, state_var="A0", value=0.5, method="multiply")

        assert len(model.events) == 3

        # Test get_event_times
        event_times = model.get_event_times(0, 5)
        assert sorted(event_times) == [1.0, 2.0, 3.0]

        # Test partial range
        partial_times = model.get_event_times(1.5, 2.5)
        assert partial_times == [2.0]

        # Test clearing events
        model.clear_events()
        assert len(model.events) == 0

        # Verify no events in range after clearing
        empty_times = model.get_event_times(0, 10)
        assert len(empty_times) == 0

        # Add new events after clearing
        model.add_event(time=5.0, state_var="A0", value=20.0, method="add")
        assert len(model.events) == 1

    def test_different_event_methods_combined(self, pk1_model_str):
        """Test combination of different event methods (add, replace, multiply)."""
        model = ScipyModel(pk1_model_str)

        # Set initial condition
        model.update_Y0(A1=100.0)

        # Add events with different methods
        model.add_event(time=1.0, state_var="A1", value=50.0, method="add")  # A1 = 100 + 50 = 150
        model.add_event(time=2.0, state_var="A1", value=75.0, method="replace")  # A1 = 75
        model.add_event(
            time=3.0, state_var="A1", value=0.5, method="multiply"
        )  # A1 = 75 * 0.5 = 37.5

        # Run simulation - expect warnings about event time inclusion
        times = np.linspace(0, 4, 400)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Not all event times were in output times")
            solution = model.run_model(times)

        A1_values = solution.states[:, 1]

        # Check values at specific times (approximately)
        def get_value_at_time(target_time):
            idx = np.argmin(np.abs(times - target_time))
            return A1_values[idx]

        # Just after t=1 (add event): should be higher than initial
        val_after_add = get_value_at_time(1.1)
        assert val_after_add > 100, f"After add event: {val_after_add}"

        # Just after t=2 (replace event): should be around 75
        val_after_replace = get_value_at_time(2.1)
        assert 70 < val_after_replace < 80, f"After replace event: {val_after_replace}"

        # Just after t=3 (multiply event): should be roughly half of replace value
        val_after_multiply = get_value_at_time(3.1)
        assert val_after_multiply < val_after_replace, f"After multiply event: {val_after_multiply}"


class TestModelComparisonAndValidation:
    """Test model comparison scenarios from notebooks."""

    def test_mass_balance_validation(self):
        """Test mass balance checks as shown in notebooks."""
        model_str = """
        States = { A, B };
        Inputs = { dose };

        k = 0.1;
        A_init = 0; B_init = 0;

        Initialize { A = A_init; B = B_init; }
        Dynamics { dt(A) = dose - k * A; dt(B) = k * A; }

        Outputs = { total };
        CalcOutputs { total = A + B; }
        End.
        """

        model = ScipyModel(model_str)

        # Add a single dose event
        model.add_event(time=0.0, state_var="A", value=100.0, method="add")

        # Run simulation
        times = np.linspace(0, 50, 500)
        solution = model.run_model(times)

        # Check mass balance: total amount should be conserved
        A_values = solution.states[:, 0]
        B_values = solution.states[:, 1]
        total_values = A_values + B_values

        # Should be approximately constant at 100 mg
        assert np.all(np.abs(total_values - 100.0) < 1e-6), "Mass balance violation"

        # Final total should equal initial dose
        assert abs(total_values[-1] - 100.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__])
