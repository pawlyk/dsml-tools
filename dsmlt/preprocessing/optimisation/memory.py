import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


__all__ = ('MemoryOptimisation', )


"""
for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
"""


class MemoryOptimisation(BaseEstimator, TransformerMixin):
    """Transforms features by change each feature to a optimise type.

    Parameters
    ----------
    inter_type : boolean, optional, default False
        Set to True to perform inter type optimisation e.g. between
        float and integer.

    axis : int (0 by default)
        axis used to optimise along. If 0, independently optimise each
        feature, otherwise (if 1) optimise each sample.

    copy : boolean, optional, default True
        Set to False to perform inplace row optimisation and avoid a
        copy (if the input is already a numpy array).

    Attributes
    ----------
    data_types_ : ndarray, shape (n_features,)
        Per feature new data type

    Examples
    --------
    >>> from dsmlt.preprocessing import MemoryOptimisation
    >>>
    >>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    >>> optimiser = MemoryOptimisation()
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

    def __init__(self, inter_type=False, axis=0, copy=True):
        # TODO add axis here
        self.inter_type = inter_type
        self.axis = axis
        self.copy = copy

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
        pass

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
        pass
