"""
Helper function to create maps for different types of variables
"""
import copy

from collections import OrderedDict

import pandas as pd
import numpy as np


__all__ = (
    "DataMapper",
    "from_boolean_to_integers_map",
    "from_explanatory_to_integers",
    "from_integers_to_boolean_map",
)


def from_explanatory_to_integers(series):
    return {v: i for i, v in enumerate(set(series.values))}


from_boolean_to_integers_map = {
    False: 0,
    True: 1,
}
from_integers_to_boolean_map = {
    0: False,
    1: True,
}


class DataMapper:
    """
    Data mapper that handle bulk map of data.
    """

    def __init__(self, scaling: (dict, str) = None, inplace: bool = False):
        # TODO use scaling parameter in mapper
        self.scaling = scaling
        self.inplace = inplace

    def _get_new_data(self, data, empty_column: str = None):
        """Prepare output data.

        Parameters
        ----------
            data: Pandas data frame.
            empty_column : Column name of data frame.

        Returns
        -------
            data_new : narray-like, shape (n_samples, n_components)
        """
        if self.inplace:
            data_new = data
        else:
            if isinstance(data, pd.DataFrame):
                if empty_column:
                    data_new = copy.deepcopy(data)
                    data_new[empty_column] = np.full(
                        len(data_new[empty_column]), np.nan
                    )
                else:
                    data_new = pd.DataFrame(
                        OrderedDict((_, []) for _ in data.columns)
                    )
            elif isinstance(data, pd.Series):
                data_new = data.copy()
            else:
                raise AttributeError(
                    "Invalid `data` type. It should be instance of pandas "
                    "Series or DataFrame."
                )

        return data_new

    def _construct_data_types(self, data):
        """Create map of relation column/type for data.

        Parameters
        ----------
            data : Pandas data frame.
        """
        self.types_ = dict(zip(data.columns, data.dtypes))

    def _construct_data_mappers(self, data):
        """
        Create map of column/map of data that consists in this column.

        Parameters
        ----------
            data : Pandas data frame.
        """
        # FIXME currently we use only one mapper. In future we need add more
        self.mappers_ = {}
        for column_, type_ in self.types_.items():
            if type_ == "object":
                self.mappers_.setdefault(
                    column_, from_explanatory_to_integers(data.get(column_))
                )

    def _get_mapper_for_column(self, column_name):
        """Get mapper for given column.

        Parameters
        ----------
            column_name : Column name of data frame.

        Returns
        -------
            data_mapper : dict - mapper for given column.
        """
        return self.mappers_.get(column_name, {})

    def _get_reversed_mapper_for_column(self, column_name):
        """
        Get reversed mapper (where keys and values swap) for given column.

        Parameters
        ----------
            column_name : Column name of data frame.

        Returns
        -------
            data_mapper : dict - mapper for given column.
        """
        return dict(
            (v, k) for k, v in self.mappers_.get(column_name, {}).items()
        )

    def fit(self, data):
        """Fit the model with data.

        Parameters
        ----------
            data : narray-like.
                Training data that represents as pandas data frame.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._construct_data_types(data)
        self._construct_data_mappers(data)

        return self

    def transform(self, data):
        """Apply data mapper on data.

        Parameters
        ----------
            data : narray-like.
                Training data that represents as pandas data frame.

        Returns
        -------
            data_new : narray-like, shape (n_samples, n_components)
        """
        data_new = self._get_new_data(data)

        for column in data.columns:
            mapper = self._get_mapper_for_column(column)
            if mapper:
                # apply mapper
                data_new[column] = data.get(column).map(mapper)
            else:
                # just copy data
                data_new[column] = data.get(column)

        return data_new

    def fit_transform(self, data):
        """Fit the model with data and apply data mapper on data.

        Parameters
        ----------
            data : narray-like.
                Training data that represents as pandas data frame.

        Returns
        -------
            data_new : narray-like, shape (n_samples, n_components)
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        """Transform data back to its original view.

        Parameters
        ----------
            data : narray-like.
                Training data that represents as pandas data frame.

        Returns
        -------
            data_new : narray-like, shape (n_samples, n_components)
        """
        data_new = self._get_new_data(data)

        for column in data.columns:
            mapper = self._get_reversed_mapper_for_column(column)
            if mapper:
                # apply mapper
                data_new[column] = data.get(column).map(mapper)
            else:
                # just copy data
                data_new[column] = data.get(column)

        return data_new

    def column_transform(self, data, column):
        """Apply data mapper on data.

        Parameters
        ----------
            data : narray-like.
                Training data that represents as pandas data frame.
            column : Column name of data frame.

        Returns
        -------
            data_new : narray-like, shape (n_samples, n_components)
        """
        if column not in self.mappers_.keys():
            raise AttributeError(
                "Invalid name of column. Column not exists in mappers."
            )
        if isinstance(data, pd.DataFrame) and column not in data.columns:
            raise AttributeError(
                "Invalid name of column. Column not exists in data."
            )

        data_new = self._get_new_data(data, empty_column=column)
        mapper = self._get_mapper_for_column(column)
        if isinstance(data, pd.DataFrame):
            data_new[column] = data.get(column).map(mapper)
        elif isinstance(data, pd.Series):
            data_new.update(data.map(mapper))
        else:
            raise AttributeError(
                "Invalid `data` type. It should be instance of pandas "
                "Series or DataFrame."
            )

        return data_new

    def column_inverse_transform(self, data, column):
        """Transform data back to its original view.

        Parameters
        ----------
            data : narray-like.
                Training data that represents as pandas data frame.
            column : Column name of data frame.

        Returns
        -------
            data_new : narray-like, shape (n_samples, n_components)
        """
        if column not in self.mappers_.keys():
            raise AttributeError(
                "Invalid name of column. Column not exists in mappers."
            )
        if isinstance(data, pd.DataFrame) and column not in data.columns:
            raise AttributeError(
                "Invalid name of column. Column not exists in data."
            )

        data_new = self._get_new_data(data, empty_column=column)
        mapper = self._get_reversed_mapper_for_column(column)
        if isinstance(data, pd.DataFrame):
            data_new[column] = data.get(column).map(mapper)
        elif isinstance(data, pd.Series):
            data_new.update(data.map(mapper))
        else:
            raise AttributeError(
                "Invalid `data` type. It should be instance of pandas "
                "Series or DataFrame."
            )

        return data_new
