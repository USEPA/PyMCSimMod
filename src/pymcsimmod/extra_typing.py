from collections.abc import Sequence
from typing import Annotated

import numpy as np
from pydantic_core import core_schema

type NumericArrayType = np.ndarray | Sequence[int | float]


class NumericArrayCheck:
    """Checks that data is 1D numeric NumPy array"""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler) -> core_schema.CoreSchema:
        def validate(value: np.ndarray | Sequence[int | float]) -> np.ndarray:
            array = value if isinstance(value, np.ndarray) else np.array(value)

            if not np.issubdtype(array.dtype, np.number):
                raise TypeError(f"Expected a numeric array, got array with dtype {array.dtype}")

            if not array.ndim == 1:
                raise TypeError(
                    f"Expected a 1 dimensional array, got {array.ndim} dimensional array"
                )

            return array

        return core_schema.no_info_plain_validator_function(validate)


NumericArray = Annotated[NumericArrayType, NumericArrayCheck]
