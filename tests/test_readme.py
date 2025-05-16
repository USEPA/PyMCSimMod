"""These tests are designed to mirror code in README.md to check that our README works.

It is manually synced, and this file should be updated when the README.md is updated.
"""

from textwrap import dedent

from pymcsimmod import parser
from pymcsimmod.model import Model


def test_quickstart(data_path):
    # ensure the parser works and returns a model instance
    text_model = """
    States = {
        x,        # Number of rabbits (1000s).
        y,        # Number of foxes (1000s).
    };

    Outputs = {};
    Inputs = {};

    alpha = 0.67;   # Birth rate of rabbits (1/d).
    beta = 1.33;    # Death rate of rabbits (1/d per 1000 foxes).
    gamma = 1.00;   # Birth rate of foxes (1/d per 1000 rabbits).
    delta = 1.00;   # Death rate of foxes (1/d).

    Initialize {
        # Assign an initial value for each state variable.
        x = 1.00;     # Initial number of rabbits (1000s).
        y = 0.75;     # Initial number of foxes (1000s).
    }
    Dynamics {
        # Time rate of change (ODE) for each state variable.
        dt(x) = alpha * x - beta * x * y;
        dt(y) = gamma * x * y - delta * y;
    }

    End.
    """
    model = parser.ModelParser().parse(dedent(text_model))
    assert isinstance(model, Model)
