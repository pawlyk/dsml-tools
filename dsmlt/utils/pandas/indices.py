"""
Helper function for indexes manipulation
"""
import operator

from typing import List, Tuple

import pandas as pd
import numpy as np


__all__ = (
    "join_indices",
    "join_indices_dataframe",
)


def join_indices(index1, index2, operation: str):
    """
    Join two indices `index1` and `index2` using operator given by `operation`.

    Parameters
    ----------
        index1 : First index
        index2 : Second index
        operation : Operator - *and* or *or*.

    Returns
    -------
        mask : A numobservations-length boolean array.
    """
    if not np.issubsctype(index1, bool):
        raise AttributeError("Invalid type of index1.")
    if not np.issubsctype(index2, bool):
        raise AttributeError("Invalid type of index2.")
    if operation not in (
        "and",
        "or",
    ):
        raise AttributeError(
            "Invalid value `{}` of operation attribute.".format(operation)
        )
    operation = operator.and_ if operation == "and" else operator.or_

    return operation(index1, index2)


def join_indices_dataframe(
    index: pd.DataFrame,
    operation: str,
    columns: (List[str], Tuple[str]) = None,
    inverse: bool = False,
):
    """
    Join two indices columns from `index` by columns `columns` using operator
    given by `operation`.

    Parameters
    ----------
        index : Index data frame that contains only True or False
        columns : List of data frame columns. If not given uses all columns
            from data frame
        operation : Operator - *and* or *or*
        inverse : Inverse columns before apply operation.\

    Returns
    -------
        mask : A numobservations-length boolean array.
    """

    def inverse_series(serie, inverse):
        return ~serie if inverse else serie

    if operation not in ("and", "or"):
        raise AttributeError(
            "Invalid value `{}` of operation attribute.".format(operation)
        )
    operation = operator.and_ if operation == "and" else operator.or_
    if not columns:
        columns = index.columns

    result = inverse_series(index.get(columns[0]), inverse)
    for column in columns[1:]:
        second = inverse_series(index.get(column), inverse)
        result = operation(result, second)

    return result
