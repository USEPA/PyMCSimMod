from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

notebooks = list((Path(__file__).parents[1] / "docs" / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("nb", notebooks, ids=[nb.name for nb in notebooks])
def test_notebooks(nb: Path):
    """Test that all notebooks run without error."""
    nb_text = nbformat.reads(nb.read_text(encoding="utf-8"), as_version=4)
    executor = ExecutePreprocessor(timeout=30)
    try:
        executor.preprocess(nb_text, {"metadata": {"path": str(nb.parent.absolute())}})
    except CellExecutionError as err:
        pytest.fail(f"Error executing the notebook {nb}:\n{err}")
