import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ...constants import NUMERICS


__all__ = ('MemoryOptimiser', )


class MemoryOptimiser(BaseEstimator, TransformerMixin):
    """Transforms features by change each feature to a optimise type.

    Parameters
    ----------
    mode : 'auto', 'convert', type, array of types, dict of types
        Mode in which should perform memory optimisation.

        - 'auto' : simply determine types from training data.
        - 'convert' : additional to 'auto' perform inter type
                      optimisation e.g. between float and integer.
        - type : predefine type of data. Can be python or numpy types.
        - array of types : predefine type for every feature.
                           Feature index used as array index.
        - dict of types : predefine type for every feature.
                          Feature index/name used as keys.

    axis : int (0 by default)
        axis used to optimise along. If 0, independently optimise each
        feature, otherwise (if 1) optimise each sample.

    copy : boolean, optional, default True
        Set to False to perform inplace row optimisation and avoid a
        copy (if the input is already a numpy array).

    Attributes
    ----------
    data_types_ : ndarray, shape (n_features,)
        Per feature new data types.

    Examples
    --------
    >>> from dsmlt.preprocessing import MemoryOptimiser
    >>>
    >>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    >>> optimiser = MemoryOptimiser()
    >>> print(optimiser.fit(data))
    MemoryOptimisation(axis=0, copy=True, inter_type=False)
    >>> print(optimiser.data_max_)
    [  1.  18.]
    >>> print(optimiser.transform(data))
    [[-1. ,  2. ],
     [-0.5,  6. ],
     [ 0. , 10. ],
     [ 1. , 18. ]]
    >>> print(optimiser.transform([[2, 2]]))
    [[ 2  2 ]]

    References
    ----------
    .. [1] ` Kaggle kernel of Guillaume Martin \
        <https://www.kaggle.com/gemartin/load-data-reduce-memory-usage>`_, \
        Load data reduce memory usage.
    """

    def __init__(self, mode='auto', axis=0, copy=True):
        if isinstance(mode, str) and mode in {'auto', 'convert', }:
            self.mode = mode

        elif isinstance(mode, (list, tuple, dict)):
            raise NotImplementedError('Not implemented yet.')

        elif mode in NUMERICS:
            self.mode = mode

        else:
            raise AttributeError(
                'Passed invalid value of `mode` - `{}`.'.format(
                    mode
                )
            )

        self.axis = axis
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the optimiser, if necessary.

        __init__ parameters are not touched.
        """
        if hasattr(self, 'data_types_'):
            del self.data_types_

    def _column_analyze_series(self, data):
        pass

    def _column_analyze_np_array(self, data):
        pass

    def _data_analyze(self, data):
        """Create map of relation column/type for data.

        Parameters
        ----------
            data : narray-like, pandas Series/DataFrame
                An numobservations by numdimensions array of observations.
        """
        # todo iterate through columns
        # todo realize better data type
        # if need iter type optimization realize it too
        pass

    def _fill_data_types(self, data):
        """Fill map of relation column/type for data with mode as dtype.

        Parameters
        ----------
            data : narray-like, pandas Series/DataFrame
                An numobservations by numdimensions array of observations.
        """
        if isinstance(data, pd.Series):
            self.data_types_ = self.mode
        elif isinstance(data, pd.DataFrame):
            self.data_types_ = dict()
            for column_name in data.columns:
                self.data_types_[column_name] = self.mode
        elif isinstance(data, np.ndarray):
            self.data_types_ = self.mode

    def fit(self, data):
        """Fit a preprocessor with data.

        Compute optimized types for columns.

        Parameters
        ----------
            data : narray-like, pandas Series/DataFrame
                Input data based on which we compute parameters.

        Returns
        -------
            self : object
                Returns the instance itself.
        """
        if self.mode in {'auto', 'convert', }:
            self._data_analyze(data)

        elif self.mode in NUMERICS:
            self._fill_data_types(data)

        else:
            pass
            # TODO implement functionality with list and dict of types

    def transform(self, data):
        """Apply preprocessor to data.

        Parameters
        ----------
            data : narray-like, pandas Series/DataFrame
                Input data that will be transformed.

        Returns
        -------
            data_new : narray-like, shape (n_samples, n_components)
        """
        return data.astype(self.data_types_, copy=self.copy)
