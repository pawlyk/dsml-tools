from abc import ABC, abstractmethod
from copy import deepcopy

from numpy import nan
from sklearn.base import BaseEstimator, TransformerMixin


__all__ = (
    'SimpleHashTableImputer',
    'HierarchicalHashTableImputer',
    'CompoundHashTableImputer',
)


class AbstractHashTableImputer(ABC, BaseEstimator, TransformerMixin):
    """
    Abstract class of imputers such that use hash table for store and
    structurize data to get simple patterns in data.
    """

    def __init__(
            self, main_columns: (list, tuple), depend_columns: (list, tuple),
            missing_values: (int, float, str, list, tuple) = nan,
            select_method='auto', merge_method=None,
    ):
        """
        Init method of AbstractHashTableImputer's inherited classes.

        Parameters:
        -----------
            main_columns: List or tuple of columns that accepted as
                main columns e.i. this mean that from data in this
                columns depends data into `depend_columns`.
            depend_columns: List or tuple of columns that accepted as
                depended columns e.i. this mean that data in this columns
                depends from data from `main_columns`.
            missing_values: int, float, str, list, tuple - a single
                value or list/tuple of values that we accept as
                indicator of missing value.
            select_method: Name of callable name or callable that uses
                for select depending values from data structure.
            merge_method: Name of callable name or callable that uses
                for merge depending values from data structure. This
                method uses only in compound imputer.
        """
        # todo realize this
        pass

    @abstractmethod
    def _get(self, keys):
        """
        Get value or tuple of values from data structure by keys.

        Parameters:
        -----------
            keys : Name or tuple of names from data structure.

        Returns
        -------
            Single value or tuple of values that get from data structure.
        """
        pass

    @abstractmethod
    def _set(self, keys, values):
        """
        Set value or tuple of values into data structure by keys.

        Parameters:
        -----------
            keys : Name or tuple of names from data structure.
            values : value or tuple of values that set into data structure.
        """
        pass

    def _sort(self):
        """
        Sort data into data structures.
        """
        # todo realize this

        pass

    def copy(self):
        """
        Copy imputer with all collected data and structures.
        """
        return deepcopy(self)

    def reset(self):
        """
        Reset internal state of imputer - data storage and structures.
        """
        # todo realize this

        pass

    def keys(self):
        """
        Get keys from data structure.
        """
        # todo realize this

        pass

    def fit(self, data):
        """Fit the imputer with data.

        Parameters
        ----------
            data : narray-like.
                Training data that represents as pandas data frame.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # todo realize this

        pass

    def transform(self, data):
        """Apply imputer on data.

        Parameters
        ----------
            data : narray-like.
                Training data that represents as pandas data frame.

        Returns
        -------
            data_new : narray-like, shape (n_samples, n_components)
        """
        # todo realize this

        pass

    def partial_fit(self, data):
        """Partially fit imputer on data.

        Parameters
        ----------
            data : narray-like.
                Training data that represents as pandas data frame.

        Returns
        -------
            data_new : narray-like, shape (n_samples, n_components)
        """
        # todo realize this

        pass


class SimpleHashTableImputer(AbstractHashTableImputer):

    def _get(self, keys):
        # todo realize this

        pass

    def _set(self, keys, values):
        # todo realize this

        pass


class HierarchicalHashTableImputer(AbstractHashTableImputer):

    def _get(self, keys):
        # todo realize this

        pass

    def _set(self, keys, values):
        # todo realize this

        pass


class CompoundHashTableImputer(AbstractHashTableImputer):

    def _get(self, keys):
        # todo realize this

        pass

    def _set(self, keys, values):
        # todo realize this

        pass


# import numpy as np
# def find_nearest(array, value):
#     array = np.sort(array)
#     first = np.searchsorted(array, value)
#     if array[first] > value:
#         return array[first - 1], array[first]
#     else:
#         return array[first], array[first + 1]
#
# array = np.random.random(10)
# print(array)
#
# value = 0.5
#
# print(find_nearest(array, value))
