import numpy as np
import pytest

from dsmlt import utils


class TestMissingFunction:

    def test_simple_numpy_ndarray(self):
        size = utils.random_size(3, low=1, high=10)
        data_ndarray = utils.random_narray(size, p_missing=0.2)
        result_ndarray = utils.missing(data_ndarray)
        assert result_ndarray.dtype.type == np.bool_
        assert result_ndarray.shape == size

    def test_simple_pandas_series(self):
        data_series = utils.random_series(20, p_missing=0.2)
        result_series = utils.missing(data_series)
        assert result_series.dtype.type == np.bool_
        assert result_series.shape == (20,)
        assert result_series.size == 20

    def test_simple_pandas_dataframe(self):
        data_dataframe = utils.random_dataframe(3, 4, p_missing=0.2)
        result_dataframe = utils.missing(data_dataframe)
        for item_type in result_dataframe.dtypes.values:
            assert item_type == np.bool_
        assert result_dataframe.shape == (3, 4,)
        assert result_dataframe.size == 12

    def test_wrong_data_type(self):
        with pytest.raises(AttributeError) as exc:
            utils.missing('some wrong parameter here')
        assert str(exc.value) == \
            "Passed value `points` with invalid type - <class 'str'>."


class TestMissingCountFunction:

    def test_simple_numpy_ndarray(self):
        size = utils.random_size(3, low=1, high=10)
        data_ndarray = utils.random_narray(size, p_missing=0.5)
        assert utils.missing_count(data_ndarray) >= 1

    def test_simple_pandas_series(self):
        data_series = utils.random_series(20, p_missing=0.5)
        assert utils.missing_count(data_series) >= 1

    def test_simple_pandas_dataframe(self):
        data_dataframe = utils.random_dataframe(3, 4, p_missing=0.5)
        assert utils.missing_count(data_dataframe) >= 1

    def test_wrong_data_type(self):
        with pytest.raises(AttributeError) as exc:
            utils.missing_count('some wrong parameter here')
        assert str(exc.value) == \
            "Passed value `points` with invalid type - <class 'str'>."
