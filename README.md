# PyMCSimMod

PyMCSimMod is a Python package that facilitates ordinary differential equation (ODE) modeling. It allows one to perform simulations for ODE models that are encoded in the [GNU MCSim](https://www.gnu.org/software/mcsim/) model specification language.

## Quickstart

Install [uv](https://docs.astral.sh/uv/) and make it available and on your path. Then:

```bash
# update and install
uv sync

# test our CLI
uv run pymcsimmod --help
uv run pymcsimmod hello
uv run pymcsimmod hello --name Andy
uv run pymcsimmod bottles --num 20
```

NOTE: this is a standard python package that you can install in other ways; we suggest using uv to make things easier, but you can always use standard pip and manage a virtual environment however you prefer.

## Developer setup

Make sure you have `uv` available on your path. Then:

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

## Parsing a model file

The `ModelParser` object in this library is able to parse a model file into a datastructure usable in python.

```python
from pathlib import Path

from pymcsimmod import parser

model_path = Path(parser.__file__).parents[2] / "tests/data/pred_prey.model"

model = parser.ModelParser().parse(model_path.read_text())

# model is now set to a datastructure representing the contents of the model file
```

## Disclaimer

The United States Environmental Protection Agency (EPA) GitHub project code is provided on an "as is" basis and the user assumes responsibility for its use.  EPA has relinquished control of the information and no longer has responsibility to protect the integrity, confidentiality, or availability of the information.  Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by EPA.  The EPA seal and logo shall not be used in any manner to imply endorsement of any commercial product or activity by EPA or the United States Government.
