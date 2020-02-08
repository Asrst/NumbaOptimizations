import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import sparse
import scipy as sp
import numba
import random

# NumbaGroupBy Class
class NumbaGroupBy:
    def __init__(self, ops = 'min'):
        self.ops = ops
        self.ops_dict = {'min': self.minimum, 'max': self.maximum,
                        'count':self.count, 'size': self.size}

    def fit(self, x, index_col):
        # Factorize and Store indexes
        self.idx, self.unique_idx = pd.factorize(x[index_col],
                                                 sort = True)
        return self

    def transform(self, x, value_col, fill_na = 0):
        # factorize value column
        cidx, unique_cids = pd.factorize(x[value_col],
                                                   sort = True)
        # make output array
        col_outputs = np.zeros((len(self.unique_idx),
                                len(unique_cids)))

        # if intialize
        if fill_na != 0:
            col_outputs[:] = fill_na

        # print(col_outputs.shape)
        col_outputs = self.ops_dict[self.ops](self.idx, cidx,
                                         x[value_col].values, col_outputs)

        return col_outputs

    def fit_transform(self, x, index_col, value_col):
        self.fit(x, index_col = index_col)
        return self.transform(x, value_col = value_col)

    @staticmethod
    @numba.jit(nopython=True,cache=True)
    def minimum(row_arr, col_arr, value_arr, output_arr):
        for i in range(len(value_arr)):
            nrow, ncol = row_arr[i], col_arr[i]
            nval = value_arr[i]
            if nval < output_arr[nrow, ncol]:
                 output_arr[nrow, ncol] = nval
        return output_arr

    @staticmethod
    @numba.jit(nopython=True,cache=True)
    def maximum(row_arr, col_arr, value_arr, output_arr):
        for i in range(len(value_arr)):
            nrow, ncol = row_arr[i], col_arr[i]
            nval = value_arr[i]
            if nval < output_arr[nrow, ncol]:
                 output_arr[nrow, ncol] = nval
        return output_arr

    @staticmethod
    @numba.jit(nopython=True,cache=True)
    def count(row_arr, col_arr, value_arr, output_arr):
        for i in range(len(value_arr)):
            if not np.isnan(value_arr[i]):
                output_arr[row_arr[i], col_arr[i]] += 1
        return output_arr

    @staticmethod
    @numba.jit(nopython=True,cache=True)
    def size(row_arr, col_arr, value_arr, output_arr):
        for i in range(len(value_arr)):
            output_arr[row_arr[i], col_arr[i]] += 1
        return output_arr
