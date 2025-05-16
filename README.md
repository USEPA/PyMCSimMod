# PyMCSimMod

PyMCSimMod is a Python package that facilitates ordinary differential equation (ODE) modeling. It allows one to perform simulations for ODE models that are encoded in the [GNU MCSim](https://www.gnu.org/software/mcsim/) model specification language.

## Quickstart

PyMCSimMod requires Python 3.13+. Install the application:

```bash
pip install pymcsimmod
```

The `ModelParser` object in this library is able to parse a model file into a datastructure usable in python. We can define a model in the MCSim model format, and parse it.

```python
from pymcsimmod import parser

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
model = parser.ModelParser().parse(text_model)
```

Review the documentation in the `docs/notebook` folder for details on how to use.

## Developer setup

Install [uv](https://docs.astral.sh/uv/) and make it available and on your path. Then:

```bash
# clone project
git clone git@github.com:USEPA/PyMCSimMod.git
cd pymcsimmod

# create virtual environment and activate
uv sync --all-extras

# run assorted commands
uv run poe --help
uv run poe test    # run tests
uv run poe lint    # identify formatting errors
uv run poe format  # fix formatting errors when possible
uv run poe build   # build a python wheel
```

GitHub Actions are enabled to execute whenever code is pushed to check code formatting and successful tests. In addition, when code is pushed to the `main` branch, a wheel artifact is generated and available in the pipeline results.

## Disclaimer

The United States Environmental Protection Agency (EPA) GitHub project code is provided on an "as is" basis and the user assumes responsibility for its use.  EPA has relinquished control of the information and no longer has responsibility to protect the integrity, confidentiality, or availability of the information.  Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by EPA.  The EPA seal and logo shall not be used in any manner to imply endorsement of any commercial product or activity by EPA or the United States Government.
