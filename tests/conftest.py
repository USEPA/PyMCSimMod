from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pytest

from pymcsimmod.models.events import DiscreteEvent
from pymcsimmod.models.scipy_model import ScipyModel
from pymcsimmod.utils.backends import detect_available_backends


# --- Data and Path Fixtures ---
@pytest.fixture(scope="session")
def data_path() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / "data"


# --- Backend Fixtures ---
@pytest.fixture(scope="session")
def available_backends() -> Dict[str, bool]:
    """Detect and cache available backends for the test session."""
    return detect_available_backends()


@pytest.fixture(scope="session")
def has_scipy(available_backends) -> bool:
    """Whether scipy backend is available."""
    return available_backends["scipy"]


@pytest.fixture(scope="session")  
def has_jax(available_backends) -> bool:
    """Whether JAX backend is available.""" 
    return available_backends["jax"]


@pytest.fixture
def skip_if_no_scipy(has_scipy):
    """Skip test if scipy backend is not available."""
    if not has_scipy:
        pytest.skip("Scipy backend not available")


@pytest.fixture
def skip_if_no_jax(has_jax):
    """Skip test if JAX backend is not available."""
    if not has_jax:
        pytest.skip("JAX backend not available")


# --- Model String Fixtures ---
@pytest.fixture(scope="session")
def simple_pk_model_str() -> str:
    """Simple pharmacokinetic model string used across multiple tests."""
    return """
    States = {
        A
    };

    Inputs = {
        dose
    };

    Outputs = {
        A_out
    };

    # Parameters defined outside blocks with default values
    ka = 1.0;
    ke = 0.1;

    Initialize {
        A = 0.0;
    }

    Dynamics {
        dt(A) = dose - ke * A;
    }

    CalcOutputs {
        A_out = A;
    }

    End.
    """


@pytest.fixture(scope="session")
def complex_pk_model_str() -> str:
    """More complex pharmacokinetic model for advanced testing."""
    return """
    States = {
        A0,    # Amount in exposure compartment (mg)
        A1,    # Amount in central compartment (mg)
        AUC    # Area under curve
    };

    Inputs = {
        dose
    };

    Outputs = {
        C,      # Concentration in central compartment (mg/L)
        Atot    # Total amount in system (mg)
    };

    # Parameters
    ka = 1.0;      # Absorption rate constant (1/h)
    ke = 0.1;      # Elimination rate constant (1/h)
    V = 10.0;      # Volume of distribution (L)

    Initialize {
        A0 = 0.0;
        A1 = 0.0;
        AUC = 0.0;
    }

    Dynamics {
        dt(A0) = dose - ka * A0;
        dt(A1) = ka * A0 - ke * A1;
        dt(AUC) = A1 / V;
    }

    CalcOutputs {
        C = A1 / V;
        Atot = A0 + A1;
    }

    End.
    """


@pytest.fixture(scope="session")
def pred_prey_model_str() -> str:
    """Predator-prey model for ecological dynamics testing."""
    return """
    States = {
        prey,
        predator
    };

    Outputs = {
        prey_out,
        predator_out
    };

    # Parameters
    alpha = 1.0;    # Prey growth rate
    beta = 0.1;     # Predation rate  
    gamma = 1.5;    # Predator efficiency
    delta = 0.075;  # Predator death rate

    Initialize {
        prey = 10.0;
        predator = 5.0;
    }

    Dynamics {
        dt(prey) = alpha * prey - beta * prey * predator;
        dt(predator) = delta * prey * predator - gamma * predator;
    }

    CalcOutputs {
        prey_out = prey;
        predator_out = predator;
    }

    End.
    """


@pytest.fixture(scope="session")
def minimal_model_str() -> str:
    """Minimal model for basic functionality testing."""
    return """
    States = { A };
    
    # Add a parameter so tests can modify it
    decay_rate = 0.1;
    
    Initialize { A = 1.0; }
    
    Dynamics { dt(A) = -decay_rate * A; }
    
    End.
    """


# --- Time and Data Fixtures ---
@pytest.fixture
def standard_times() -> np.ndarray:
    """Standard time array for model testing."""
    return np.linspace(0, 10, 101)


@pytest.fixture
def short_times() -> np.ndarray:
    """Short time array for quick tests."""
    return np.linspace(0, 5, 11)


@pytest.fixture
def event_times() -> List[float]:
    """Standard event times for discrete event testing."""
    return [1.0, 3.0, 7.0, 9.0]


# --- Model Instance Fixtures ---
@pytest.fixture
def simple_scipy_model(simple_pk_model_str, skip_if_no_scipy) -> ScipyModel:
    """Create a scipy model instance for testing."""
    return ScipyModel(simple_pk_model_str)


@pytest.fixture
def complex_scipy_model(complex_pk_model_str, skip_if_no_scipy) -> ScipyModel:
    """Create a complex scipy model instance for testing."""
    return ScipyModel(complex_pk_model_str)


@pytest.fixture
def simple_jax_model(simple_pk_model_str, skip_if_no_jax):
    """Create a JAX model instance for testing."""
    from pymcsimmod.models.jax_model import JaxModel
    return JaxModel(simple_pk_model_str)


@pytest.fixture
def complex_jax_model(complex_pk_model_str, skip_if_no_jax):
    """Create a complex JAX model instance for testing."""
    from pymcsimmod.models.jax_model import JaxModel
    return JaxModel(complex_pk_model_str)


# --- Event Fixtures ---
@pytest.fixture
def sample_events(event_times) -> List[DiscreteEvent]:
    """Create sample discrete events for testing."""
    return [
        DiscreteEvent(time=event_times[0], state_var="A", value=10.0),
        DiscreteEvent(time=event_times[1], state_var="A", value=5.0),
        DiscreteEvent(time=event_times[2], state_var="A", value=15.0),
    ]


@pytest.fixture
def single_event() -> DiscreteEvent:
    """Single discrete event for basic testing."""
    return DiscreteEvent(time=2.0, state_var="A", value=8.0)


# --- Parameter and Initial Condition Fixtures ---
@pytest.fixture
def standard_parameters() -> Dict[str, float]:
    """Standard parameter set for testing."""
    return {
        "ka": 1.0,
        "ke": 0.1,
        "V": 10.0,
    }


@pytest.fixture
def alternative_parameters() -> Dict[str, float]:
    """Alternative parameter set for comparison testing.""" 
    return {
        "ka": 2.0,
        "ke": 0.5,
        "V": 20.0,
    }


@pytest.fixture
def standard_initial_conditions() -> Dict[str, float]:
    """Standard initial conditions for testing."""
    return {
        "A": 0.0,
        "A0": 0.0,
        "A1": 0.0,
        "AUC": 0.0,
    }


@pytest.fixture
def nonzero_initial_conditions() -> Dict[str, float]:
    """Non-zero initial conditions for testing."""
    return {
        "A": 10.0,
        "A0": 5.0,
        "A1": 15.0,
        "AUC": 0.0,
    }


# --- Forcing Function Data Fixtures ---
@pytest.fixture
def onoff_forcing_params() -> Dict[str, float]:
    """Standard OnOff forcing function parameters."""
    return {"t0": 1.0, "t1": 5.0, "s": 10.0}


@pytest.fixture
def perdose_forcing_params() -> Dict[str, float]:
    """Standard PerDose forcing function parameters."""
    return {"t0": 0.0, "duration": 1.0, "period": 24.0, "s": 10.0}


@pytest.fixture
def ndoses_forcing_params() -> Dict:
    """Standard NDoses forcing function parameters."""
    return {"t0_list": [1.0, 25.0, 49.0], "duration": 1.0, "s": 10.0}


@pytest.fixture
def interpolation_data() -> Dict[str, List[float]]:
    """Standard interpolation data for forcing functions."""
    return {
        "times": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        "values": [0.0, 5.0, 10.0, 8.0, 3.0, 0.0]
    }
