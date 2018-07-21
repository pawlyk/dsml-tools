import numpy as np
import pytest

from dsmlt import utils


def test_missing():
    size = utils.random_size(3, low=1, high=10)
    data_ndarray = utils.random_narray(size, p_missing=0.2)
    result_ndarray = utils.missing(data_ndarray)
    assert result_ndarray.dtype.type == np.bool_
    assert result_ndarray.shape == size

    data_series = utils.random_series(20, p_missing=0.2)
    result_series = utils.missing(data_series)
    assert result_series.dtype.type == np.bool_
    assert result_series.shape == (20, )
    assert result_series.size == 20

    data_dataframe = utils.random_dataframe(3, 4, p_missing=0.2)
    result_dataframe = utils.missing(data_dataframe)
    for item_type in result_dataframe.dtypes.values:
        assert item_type == np.bool_
    assert result_dataframe.shape == (3, 4, )
    assert result_dataframe.size == 12

    with pytest.raises(AttributeError) as exc:
        utils.missing('some wrong parameter here')
    assert str(exc.value) == \
        "Passed value `points` with invalid type - <class 'str'>."


def test_missing_count():
    size = utils.random_size(3, low=1, high=10)
    data_ndarray = utils.random_narray(size, p_missing=0.5)
    assert utils.missing_count(data_ndarray) >= 1

    data_series = utils.random_series(20, p_missing=0.5)
    assert utils.missing_count(data_series) >= 1

    data_dataframe = utils.random_dataframe(3, 4, p_missing=0.5)
    assert utils.missing_count(data_dataframe) >= 1

    with pytest.raises(AttributeError) as exc:
        utils.missing_count('some wrong parameter here')
    assert str(exc.value) == \
        "Passed value `points` with invalid type - <class 'str'>."
