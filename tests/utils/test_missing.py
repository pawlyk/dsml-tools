import numpy as np
import pytest
from pandas import DataFrame, Series

from dsmlt.utils import (
    missing, missing_count,
    random_size, random_series, random_dataframe, random_narray,
)


class TestMissingFunction:

    def test_simple_numpy_ndarray(self):
        # Test on presence NAN value
        size = random_size(3, low=1, high=10)
        data_ndarray = random_narray(size, p_missing=0.2)
        result_ndarray = missing(data_ndarray)
        assert result_ndarray.dtype.type == np.bool_
        assert result_ndarray.shape == size

        data_ndarray = np.array(
            [[1, 2, 3, 4, np.NAN],
             [5, 6, 7, 8, 9]]
        )
        assert missing(data_ndarray).any()

        # Test on presence optional None value as missing
        data_ndarray = np.array(
            [[1, 2, 3, 4, None],
             [5, 6, 7, 8, 9]]
        )
        assert missing(data_ndarray).any()

        # Test on presence optional 999 value as missing
        data_ndarray = np.array(
            [[1, 2, 3, 4, 999],
             [5, 6, 7, 8, 9]]
        )
        assert missing(data_ndarray, missing_value=999).any()

    def test_simple_pandas_series(self):
        # Test on presence NAN value
        data_series = random_series(20, p_missing=0.2)
        result_series = missing(data_series)
        assert result_series.dtype.type == np.bool_
        assert result_series.shape == (20,)
        assert result_series.size == 20

        data_series = Series([1, 2, 3, 4, 5, np.NAN, '', 'None'])
        assert any(missing(data_series))

        # Test on presence optional None value as missing
        data_series = Series([1, 2, 3, 4, 5, None, '', 'None'])
        assert any(missing(data_series))

        # Test on presence optional '' value as missing
        data_series = Series([1, 2, 3, 4, 5, 6, '', 'a'])
        assert any(missing(data_series, missing_value=''))

        # Test on presence optional 'None' value as missing
        data_series = Series([1, 2, 3, 4, 5, 6, 'a', 'None'])
        assert any(missing(data_series, missing_value='None'))

        # Test on presence optional 999 value as missing
        data_series = Series([1, 2, 3, 4, 5, 999, 'a', 'a'])
        assert any(missing(data_series, missing_value=999))

        # Test on presence optional list of values as missing
        data_series = Series([1, 2, 3, 4, 5, 999, '', 'a'])
        assert any(missing(data_series, missing_value=[999, 'a', '']))
        assert any(missing(data_series, missing_value=(999, 'a', '')))

    def test_simple_pandas_dataframe(self):
        # Test on presence NAN value
        data_dataframe = random_dataframe(3, 4, p_missing=0.2)
        result_dataframe = missing(data_dataframe)
        for item_type in result_dataframe.dtypes.values:
            assert item_type == np.bool_
        assert result_dataframe.shape == (3, 4,)
        assert result_dataframe.size == 12

        data_dataframe = DataFrame({
            'one': [1, 2, 3, 4, 5, np.NAN, '', 'None'],
            'two': [1, 2, np.NAN, 4, 5, 6, '', 'None'],
        })
        assert any(missing(data_dataframe))

        # Test on presence optional None value as missing
        data_dataframe = DataFrame({
            'one': [1, 2, 3, 4, 5, None, '', 'None'],
            'two': [1, 2, None, 4, 5, 6, '', 'None'],
        })
        assert any(missing(data_dataframe))

        # Test on presence optional '' value as missing
        data_dataframe = DataFrame({
            'one': [1, 2, 3, 4, 5, 6, '', 'b'],
            'two': [1, 2, 3, 4, 5, 6, '', 'c'],
        })
        assert any(missing(data_dataframe, missing_value=''))

        # Test on presence optional 'None' value as missing
        data_dataframe = DataFrame({
            'one': [1, 2, 3, 4, 5, 6, 'b', 'None'],
            'two': [1, 2, 3, 4, 5, 6, 'c', 'None'],
        })
        assert any(missing(data_dataframe, missing_value='None'))

        # Test on presence optional 999 value as missing
        data_dataframe = DataFrame({
            'one': [1, 2, 3, 4, 5, 999, 'b', 'b'],
            'two': [1, 2, 999, 4, 5, 6, 'c', 'c'],
        })
        assert any(missing(data_dataframe, missing_value=999))

        # Test on presence optional list of values as missing
        data_dataframe = DataFrame({
            'one': [1, 2, 3, 4, 5, 999, '', 'a'],
            'two': [1, 2, 999, 4, 5, 6, '', 'c'],
        })
        assert any(missing(data_dataframe, missing_value=[999, 'a', '']))
        assert any(missing(data_dataframe, missing_value=(999, 'a', '')))

    def test_wrong_data_type(self):
        with pytest.raises(AttributeError) as exc:
            missing('some wrong parameter here')
        assert str(exc.value) == \
            "Passed value `points` with invalid type - <class 'str'>."

    def test_wrong_missing_value(self):
        with pytest.raises(AttributeError) as exc:
            missing(np.array([1, 2, 3]), missing_value={'a': 1, 'b': 2})
        assert str(exc.value) == \
            "Passed value `missing_value` with invalid type - <class 'dict'>."


class TestMissingCountFunction:

    def test_simple_numpy_ndarray(self):
        size = random_size(3, low=1, high=10)
        data_ndarray = random_narray(size, p_missing=0.5)
        assert missing_count(data_ndarray) >= 1

    def test_simple_pandas_series(self):
        data_series = random_series(20, p_missing=0.5)
        assert missing_count(data_series) >= 1

    def test_simple_pandas_dataframe(self):
        data_dataframe = random_dataframe(3, 4, p_missing=0.5)
        assert missing_count(data_dataframe) >= 1

    def test_wrong_data_type(self):
        with pytest.raises(AttributeError) as exc:
            missing_count('some wrong parameter here')
        assert str(exc.value) == \
            "Passed value `points` with invalid type - <class 'str'>."
