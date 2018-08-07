import numpy as np
import pandas as pd
import pytest

from dsmlt.preprocessing import MemoryOptimiser
from dsmlt.utils.random_data import (
    random_narray, random_series, random_dataframe,
)


class TestMemoryOptimisation:

    def test_init_optimiser(self):
        optimiser = MemoryOptimiser()
        assert optimiser.mode == 'auto'
        assert optimiser.axis == 0
        assert optimiser.copy is True

        optimiser = MemoryOptimiser(mode='convert', axis=3, copy=False)
        assert optimiser.mode == 'convert'
        assert optimiser.axis == 3
        assert optimiser.copy is False

        optimiser = MemoryOptimiser(mode=int)
        assert optimiser.mode == int

        optimiser = MemoryOptimiser(mode=float)
        assert optimiser.mode == float

        optimiser = MemoryOptimiser(mode=np.int8)
        assert optimiser.mode == np.int8

        optimiser = MemoryOptimiser(mode=np.float128)
        assert optimiser.mode == np.float128

    def test_wrong_init_optimiser(self):
        with pytest.raises(AttributeError) as exc:
            MemoryOptimiser(mode='random')
        assert str(exc.value) == 'Passed invalid value of `mode` - `random`.'

        with pytest.raises(AttributeError) as exc:
            MemoryOptimiser(mode=set)
        assert str(exc.value) == \
            "Passed invalid value of `mode` - `<class 'set'>`."

        with pytest.raises(NotImplementedError) as exc:
            MemoryOptimiser(mode=[int, np.int8, np.float128, ])
        assert str(exc.value) == 'Not implemented yet.'

        with pytest.raises(NotImplementedError) as exc:
            MemoryOptimiser(mode={'a': int, 'b': np.int8, 'c': np.float16, })
        assert str(exc.value) == 'Not implemented yet.'

    def test_reset_optimiser(self):
        data = random_narray((2, 3, 4), astype=np.int64)
        optimiser = MemoryOptimiser(mode=np.int8)
        optimiser.fit(data)
        assert hasattr(optimiser, 'data_types_')

        optimiser._reset()
        assert not hasattr(optimiser, 'data_types_')

    def test_simple_numpy_ndarray(self):
        # test mode int
        data = random_narray((2, 3, 4), astype=np.int64)
        optimiser = MemoryOptimiser(mode=np.int8)
        optimiser.fit(data)
        assert optimiser.data_types_ == np.int8

        data = optimiser.transform(data)
        assert data.dtype.type == np.int8

        # test mode float
        data = random_narray((2, 3, 4), astype=np.float64)
        optimiser = MemoryOptimiser(mode=np.float16)
        optimiser.fit(data)
        assert optimiser.data_types_ == np.float16

        data = optimiser.transform(data)
        assert data.dtype.type == np.float16

    def test_simple_pandas_series(self):
        # test mode int
        data = random_series(100, astype=np.int64)
        optimiser = MemoryOptimiser(mode=np.int8)
        optimiser.fit(data)
        assert optimiser.data_types_ == np.int8

        data = optimiser.transform(data)
        assert data.dtype.type == np.int8

        # test mode float
        data = random_series(100, astype=np.float64)
        optimiser = MemoryOptimiser(mode=np.float16)
        optimiser.fit(data)
        assert optimiser.data_types_ == np.float16

        data = optimiser.transform(data)
        assert data.dtype.type == np.float16

    def test_simple_pandas_dataframe(self):
        # test mode int
        data = random_dataframe(3, 4, astype=np.int64)
        optimiser = MemoryOptimiser(mode=np.int8)
        optimiser.fit(data)
        assert len(optimiser.data_types_) == 4
        assert list(optimiser.data_types_.values())[0] == np.int8

        data = optimiser.transform(data)
        for item_type in data.dtypes.values:
            assert item_type == np.int8

        # test mode float
        data = random_dataframe(3, 4, astype=np.float64)
        optimiser = MemoryOptimiser(mode=np.float16)
        optimiser.fit(data)
        assert len(optimiser.data_types_) == 4
        assert list(optimiser.data_types_.values())[0] == np.float16

        data = optimiser.transform(data)
        for item_type in data.dtypes.values:
            assert item_type == np.float16

    def test_analyze_numpy_ndarray(self):
        # test mode auto for int
        data = random_narray(
            (2, 3, 4), low=0, high=100, dtype=np.int8, astype=np.int64
        )
        optimiser = MemoryOptimiser(mode='auto')
        optimiser.fit(data)
        assert optimiser.data_types_ == np.int8

        data = optimiser.transform(data)
        assert data.dtype.type == np.int8

        # test mode auto for float
        data = random_narray((2, 3, 4), dtype=np.float16, astype=np.float64)
        optimiser = MemoryOptimiser(mode='auto')
        optimiser.fit(data)
        assert optimiser.data_types_ == np.float16

        data = optimiser.transform(data)
        assert data.dtype.type == np.float16

        # test mode convert for int
        data = random_narray(
            (2, 3, 4), low=0, high=100, dtype=np.int8, astype=np.int64
        )
        optimiser = MemoryOptimiser(mode='convert')
        optimiser.fit(data)
        assert optimiser.data_types_ == np.int8

        data = optimiser.transform(data)
        assert data.dtype.type == np.int8

        # test mode convert for float
        data = random_narray(
            (2, 3, 4), low=0, high=100, dtype=np.int8, astype=np.int64
        )
        optimiser = MemoryOptimiser(mode='convert')
        optimiser.fit(data)
        assert optimiser.data_types_ == np.int8

        data = optimiser.transform(data)
        assert data.dtype.type == np.int8

    def test_analyze_pandas_series(self):
        # test mode auto for int
        data = random_series(
            100, low=0, high=100, dtype=np.int8, astype=np.int64
        )
        optimiser = MemoryOptimiser(mode='auto')
        optimiser.fit(data)
        assert optimiser.data_types_ == np.int8

        data = optimiser.transform(data)
        assert data.dtype.type == np.int8

        # test mode auto for float
        data = random_series(100, dtype=np.float16, astype=np.float64)
        optimiser = MemoryOptimiser(mode='auto')
        optimiser.fit(data)
        assert optimiser.data_types_ == np.float16

        data = optimiser.transform(data)
        assert data.dtype.type == np.float16

        # test mode convert for int
        data = random_series(
            100, low=0, high=100, dtype=np.int8, astype=np.int64
        )
        optimiser = MemoryOptimiser(mode='convert')
        optimiser.fit(data)

        data = optimiser.transform(data)
        assert data.dtype.type == np.int8

        # test mode convert for float
        data = random_series(
            100, low=0, high=100, dtype=np.int8, astype=np.float64
        )
        optimiser = MemoryOptimiser(mode='convert')
        optimiser.fit(data)
        assert optimiser.data_types_ == np.int8

        data = optimiser.transform(data)
        assert data.dtype.type == np.int8

    def test_analyze_pandas_dataframe(self):
        # test mode auto for int
        data = random_dataframe(
            3, 4, low=0, high=100, dtype=np.int8, astype=np.int64
        )
        optimiser = MemoryOptimiser(mode='auto')
        optimiser.fit(data)
        assert list(optimiser.data_types_.values())[0] == np.int8

        data = optimiser.transform(data)
        for item_type in data.dtypes.values:
            assert item_type == np.int8

        # test mode auto for float
        data = random_dataframe(3, 4, dtype=np.float16, astype=np.float64)
        optimiser = MemoryOptimiser(mode='auto')
        optimiser.fit(data)
        assert list(optimiser.data_types_.values())[0] == np.float16

        data = optimiser.transform(data)
        for item_type in data.dtypes.values:
            assert item_type == np.float16

        # test mode convert for int
        data = random_dataframe(
            3, 4, low=0, high=100, dtype=np.int8, astype=np.int64
        )
        optimiser = MemoryOptimiser(mode='convert')
        optimiser.fit(data)
        assert list(optimiser.data_types_.values())[0] == np.int8

        data = optimiser.transform(data)
        for item_type in data.dtypes.values:
            assert item_type == np.int8

        # test mode convert for float
        data = random_dataframe(
            3, 4, low=0, high=100, dtype=np.int8, astype=np.float64
        )
        optimiser = MemoryOptimiser(mode='convert')
        optimiser.fit(data)
        assert list(optimiser.data_types_.values())[0] == np.int8

        data = optimiser.transform(data)
        for item_type in data.dtypes.values:
            assert item_type == np.int8

        # test mode auto for int with categorical
        data = random_dataframe(
            3, 4, low=0, high=100, dtype=np.int8, astype=np.int64
        )
        data['E'] = data['A'].astype('category')
        optimiser = MemoryOptimiser(mode='auto')
        optimiser.fit(data)
        assert list(optimiser.data_types_.values())[-1] == 'category'
        assert list(optimiser.data_types_.values())[0] == np.int8

        data = optimiser.transform(data)
        assert data.dtypes.values[-1] == 'category'

        # test mode convert for int with categorical
        data = random_dataframe(
            3, 4, low=0, high=100, dtype=np.int8, astype=np.int64
        )
        data['E'] = data['A'].astype('category')
        optimiser = MemoryOptimiser(mode='convert')
        optimiser.fit(data)
        assert list(optimiser.data_types_.values())[-1] == 'category'
        assert list(optimiser.data_types_.values())[0] == np.int8

        data = optimiser.transform(data)
        assert data.dtypes.values[-1] == 'category'

        data = random_dataframe(
            3, 4, low=0, high=100, dtype=np.int8, astype=np.int64
        )
        data['E'] = pd.Series(['a', 'b', 'c']).astype('category')
        optimiser = MemoryOptimiser(mode='convert')
        optimiser.fit(data)
        assert list(optimiser.data_types_.values())[-1] == 'category'
        assert list(optimiser.data_types_.values())[0] == np.int8

        data = optimiser.transform(data)
        assert data.dtypes.values[-1] == 'category'
