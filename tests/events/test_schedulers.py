"""Tests for event schedulers submodule."""

import numpy as np
import pandas as pd
import pytest

from pymcsimmod.events import NDoses, OnOff, PerDoses, create_event_scheduler
from pymcsimmod.events.base import DataFrameEventScheduler
from pymcsimmod.models.scipy_model import ScipyModel


def test_ndoses_scheduler():
    """Test NDoses event scheduler."""
    scheduler = NDoses(t0_list=[1.0, 5.0, 10.0], value=100.0, method="add")

    # Within range [0, 8]
    events = scheduler.get_events(state_var="A", t_start=0.0, t_end=8.0)
    assert len(events) == 2
    assert events[0].time == 1.0
    assert events[0].value == 100.0
    assert events[0].method == "add"
    assert events[1].time == 5.0

    # With list of values matching t0_list
    scheduler_list = NDoses(t0_list=[1.0, 5.0, 10.0], value=[50.0, 60.0, 70.0], method="replace")
    events_list = scheduler_list.get_events(state_var="A", t_start=0.0, t_end=12.0)
    assert len(events_list) == 3
    assert events_list[0].value == 50.0
    assert events_list[1].value == 60.0
    assert events_list[2].value == 70.0


def test_perdoses_scheduler():
    """Test PerDoses event scheduler."""
    # Test without limits
    scheduler = PerDoses(t0=1.0, period=2.0, value=10.0, method="add")
    events = scheduler.get_events(state_var="A", t_start=0.0, t_end=6.0)
    # Expected times: 1.0, 3.0, 5.0
    assert len(events) == 3
    assert [e.time for e in events] == [1.0, 3.0, 5.0]

    # Test with n limits
    scheduler_n = PerDoses(t0=1.0, period=2.0, value=10.0, method="add", n=2)
    events_n = scheduler_n.get_events(state_var="A", t_start=0.0, t_end=10.0)
    # Expected times: 1.0, 3.0
    assert len(events_n) == 2
    assert [e.time for e in events_n] == [1.0, 3.0]

    # Test with t_end limits
    scheduler_tend = PerDoses(t0=1.0, period=2.0, value=10.0, method="add", t_end=4.0)
    events_tend = scheduler_tend.get_events(state_var="A", t_start=0.0, t_end=10.0)
    # Expected times: 1.0, 3.0
    assert len(events_tend) == 2
    assert [e.time for e in events_tend] == [1.0, 3.0]


def test_onoff_scheduler():
    """Test OnOff event scheduler."""
    # Test add method
    scheduler_add = OnOff(t0=2.0, t1=5.0, value=10.0, method="add")
    events_add = scheduler_add.get_events(state_var="A", t_start=0.0, t_end=10.0)
    assert len(events_add) == 2
    assert events_add[0].time == 2.0
    assert events_add[0].value == 10.0
    assert events_add[0].method == "add"
    assert events_add[1].time == 5.0
    assert events_add[1].value == -10.0
    assert events_add[1].method == "add"

    # Test replace method with default off
    scheduler_replace = OnOff(t0=2.0, t1=5.0, value=10.0, method="replace")
    events_replace = scheduler_replace.get_events(state_var="A", t_start=0.0, t_end=10.0)
    assert len(events_replace) == 2
    assert events_replace[0].value == 10.0
    assert events_replace[0].method == "replace"
    assert events_replace[1].value == 0.0
    assert events_replace[1].method == "replace"

    # Test multiply method with default off
    scheduler_multiply = OnOff(t0=2.0, t1=5.0, value=4.0, method="multiply")
    events_multiply = scheduler_multiply.get_events(state_var="A", t_start=0.0, t_end=10.0)
    assert len(events_multiply) == 2
    assert events_multiply[0].value == 4.0
    assert events_multiply[1].value == 0.25

    # Test custom value_off
    scheduler_off = OnOff(t0=2.0, t1=5.0, value=10.0, method="add", value_off=5.0)
    events_off = scheduler_off.get_events(state_var="A", t_start=0.0, t_end=10.0)
    assert len(events_off) == 2
    assert events_off[0].value == 10.0
    assert events_off[1].value == 5.0


def test_dataframe_scheduler():
    """Test DataFrameEventScheduler."""
    df = pd.DataFrame(
        {
            "time": [1.0, 3.0, 5.0],
            "value": [10.0, 20.0, 30.0],
            "state_var": ["A", "B", "A"],
            "method": ["add", "replace", "add"],
        }
    )

    scheduler = DataFrameEventScheduler(df=df)

    # Query for state_var "A"
    events_a = scheduler.get_events(state_var="A", t_start=0.0, t_end=10.0)
    assert len(events_a) == 2
    assert events_a[0].time == 1.0
    assert events_a[0].value == 10.0
    assert events_a[0].state_var == "A"
    assert events_a[1].time == 5.0
    assert events_a[1].value == 30.0

    # Query for state_var "B"
    events_b = scheduler.get_events(state_var="B", t_start=0.0, t_end=10.0)
    assert len(events_b) == 1
    assert events_b[0].time == 3.0
    assert events_b[0].value == 20.0
    assert events_b[0].method == "replace"


def test_create_event_scheduler():
    """Test create_event_scheduler factory function."""
    s1 = create_event_scheduler("NDoses", t0_list=[1, 2], value=5)
    assert isinstance(s1, NDoses)

    s2 = create_event_scheduler("PerDoses", t0=0, period=24, value=10)
    assert isinstance(s2, PerDoses)

    s3 = create_event_scheduler("OnOff", t0=0, t1=10, value=5)
    assert isinstance(s3, OnOff)

    df = pd.DataFrame({"time": [1], "value": [2]})
    s4 = create_event_scheduler("DataFrame", df=df)
    assert isinstance(s4, DataFrameEventScheduler)

    with pytest.raises(ValueError, match="Unknown event scheduler type"):
        create_event_scheduler("InvalidType")


def test_model_scheduler_integration_scipy():
    """Test integrating event schedulers with ScipyModel."""
    model_str = """
    States = {
        A
    };
    Initialize {
        A = 0.0;
    }
    Dynamics {
        dt(A) = 0.0;
    }
    End.
    """
    model = ScipyModel(model_str)

    # Assign an NDoses event scheduler using string name
    model.assign_event("A", "NDoses", t0_list=[2.0, 7.0], value=10.0, method="add")

    # Add another scheduler using add_event_scheduler method
    sec_sched = PerDoses(t0=4.0, period=2.0, value=5.0, method="add", n=2)
    model.add_event_scheduler("A", sec_sched)

    times = np.linspace(0, 10, 11)
    result = model.run_model(times)

    # Expected times:
    # t=2.0: +10 (from NDoses) -> A = 10.0
    # t=4.0: +5 (from PerDoses) -> A = 15.0
    # t=6.0: +5 (from PerDoses) -> A = 20.0
    # t=7.0: +10 (from NDoses) -> A = 30.0

    # Find values at exact non-boundary times to be robust to pre/post event reporting
    # of different backends at boundary times
    a_vals = result.dataframe["A"].values
    idx_3 = np.where(np.abs(result.times - 3.0) < 1e-12)[0][0]
    idx_5 = np.where(np.abs(result.times - 5.0) < 1e-12)[0][0]
    idx_8 = np.where(np.abs(result.times - 8.0) < 1e-12)[0][0]

    np.testing.assert_allclose(a_vals[idx_3], 10.0)
    np.testing.assert_allclose(a_vals[idx_5], 15.0)
    np.testing.assert_allclose(a_vals[idx_8], 30.0)

    # Test clear_events removes schedulers
    model.clear_events()
    assert len(model.events) == 0
    assert len(model._event_schedulers) == 0

    # Run again and ensure state remains at 0.0
    result_clear = model.run_model(times)
    np.testing.assert_allclose(result_clear.dataframe["A"].values, 0.0)


def test_model_scheduler_integration_jax():
    """Test integrating event schedulers with JaxModel."""
    pytest.importorskip("jax")
    from pymcsimmod.models.jax_model import JaxModel

    model_str = """
    States = {
        A
    };
    Initialize {
        A = 0.0;
    }
    Dynamics {
        dt(A) = 0.0;
    }
    End.
    """
    model = JaxModel(model_str)

    # Assign an NDoses event scheduler using string name
    model.assign_event("A", "NDoses", t0_list=[2.0, 7.0], value=10.0, method="add")

    # Add another scheduler using add_event_scheduler method
    sec_sched = PerDoses(t0=4.0, period=2.0, value=5.0, method="add", n=2)
    model.add_event_scheduler("A", sec_sched)

    times = np.linspace(0, 10, 11)
    result = model.run_model(times)

    # Expected times:
    # t=2.0: +10 (from NDoses) -> A = 10.0
    # t=4.0: +5 (from PerDoses) -> A = 15.0
    # t=6.0: +5 (from PerDoses) -> A = 20.0
    # t=7.0: +10 (from NDoses) -> A = 30.0

    # Find values at exact non-boundary times to be robust to pre/post event reporting
    # of different backends at boundary times
    a_vals = result.dataframe["A"].values
    idx_3 = np.where(np.abs(result.times - 3.0) < 1e-12)[0][0]
    idx_5 = np.where(np.abs(result.times - 5.0) < 1e-12)[0][0]
    idx_8 = np.where(np.abs(result.times - 8.0) < 1e-12)[0][0]

    np.testing.assert_allclose(a_vals[idx_3], 10.0)
    np.testing.assert_allclose(a_vals[idx_5], 15.0)
    np.testing.assert_allclose(a_vals[idx_8], 30.0)

    # Test clear_events removes schedulers
    model.clear_events()
    assert len(model.events) == 0
    assert len(model._event_schedulers) == 0

    # Run again and ensure state remains at 0.0
    result_clear = model.run_model(times)
    np.testing.assert_allclose(result_clear.dataframe["A"].values, 0.0)
