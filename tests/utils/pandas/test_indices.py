import numpy as np
import pytest

from pandas import DataFrame, Series

from dsmlt.utils import (
    join_indices,
    join_indices_dataframe,
)


class TestMissingFunction:
    def test_indices(self):
        index1 = np.array([True, True, False, False], dtype=bool)
        index2 = np.array([True, False, True, False], dtype=bool)

        assert np.array_equal(
            np.array([True, True, True, False], dtype=bool),
            join_indices(index1, index2, operation="or"),
        )
        assert np.array_equal(
            np.array([True, False, False, False], dtype=bool),
            join_indices(index1, index2, operation="and"),
        )

    def test_indices_wrong_type_index(self):
        with pytest.raises(AttributeError) as exc:
            join_indices(
                np.array([1, 2, 3], dtype=int),
                np.array([True, False, False, False], dtype=bool),
                operation="and",
            )

        assert str(exc.value) == "Invalid type of index1."

        with pytest.raises(AttributeError) as exc:
            join_indices(
                np.array([True, False, False, False], dtype=bool),
                np.array([1, 2, 3], dtype=int),
                operation="and",
            )

        assert str(exc.value) == "Invalid type of index2."

    def test_indices_wrong_operation(self):
        with pytest.raises(AttributeError) as exc:
            join_indices(
                np.array([True, True, False, False], dtype=bool),
                np.array([True, False, False, False], dtype=bool),
                operation="xor",
            )

        assert str(exc.value) == "Invalid value `xor` of operation attribute."

        with pytest.raises(AttributeError) as exc:
            join_indices(
                np.array([True, True, False, False], dtype=bool),
                np.array([True, False, False, False], dtype=bool),
                operation=1,
            )

        assert str(exc.value) == "Invalid value `1` of operation attribute."

    def test_indices_dataframe(self):
        data_frame = DataFrame(
            [
                [
                    True,
                    False,
                ],
                [
                    False,
                    True,
                ],
                [
                    False,
                    False,
                ],
                [
                    True,
                    True,
                ],
            ]
        )

        assert Series([False, False, False, True]).equals(
            join_indices_dataframe(data_frame, operation="and")
        )
        assert Series([True, True, False, True]).equals(
            join_indices_dataframe(data_frame, operation="or")
        )

        assert Series([False, False, True, False]).equals(
            join_indices_dataframe(data_frame, operation="and", inverse=True)
        )
        assert Series([True, True, True, False]).equals(
            join_indices_dataframe(data_frame, operation="or", inverse=True)
        )

    def test_indices_dataframe_with_columns(self):
        data_frame = DataFrame(
            [
                [
                    True,
                    False,
                    True,
                ],
                [
                    False,
                    True,
                    False,
                ],
                [
                    False,
                    False,
                    True,
                ],
                [
                    True,
                    True,
                    False,
                ],
            ],
            columns=(
                "A",
                "B",
                "C",
            ),
        )

        assert Series([False, False, False, True]).equals(
            join_indices_dataframe(
                data_frame,
                operation="and",
                columns=(
                    "A",
                    "B",
                ),
            )
        )
        assert Series([True, True, False, True]).equals(
            join_indices_dataframe(
                data_frame,
                operation="or",
                columns=(
                    "A",
                    "B",
                ),
            )
        )

        assert Series([False, False, True, False]).equals(
            join_indices_dataframe(
                data_frame,
                operation="and",
                inverse=True,
                columns=(
                    "A",
                    "B",
                ),
            )
        )
        assert Series([True, True, True, False]).equals(
            join_indices_dataframe(
                data_frame,
                operation="or",
                inverse=True,
                columns=(
                    "A",
                    "B",
                ),
            )
        )

    def test_indices_dataframe_wrong_operation(self):
        data_frame = DataFrame(
            [
                [
                    True,
                    False,
                ],
                [
                    False,
                    True,
                ],
                [False, False],
                [
                    True,
                    True,
                ],
            ]
        )

        with pytest.raises(AttributeError) as exc:
            join_indices_dataframe(data_frame, operation="xor")

        assert str(exc.value) == "Invalid value `xor` of operation attribute."

        with pytest.raises(AttributeError) as exc:
            join_indices_dataframe(data_frame, operation=1)

        assert str(exc.value) == "Invalid value `1` of operation attribute."
