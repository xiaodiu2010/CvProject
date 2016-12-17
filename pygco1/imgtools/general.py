"""
Some general purpose image tools.

Yujia Li, 08/2013
"""

import numpy as np

def mat_switch_row_col_order(x, mat_shape):
    """
    Convert a list of matrices stored in column major order to row major order,
    or the other way around.

    x: N*(H*W) matrix, each row is one element matrix of size H*W, stored in
        column major order.
    mat_shape: should be (H,W)

    Return: newx, N*(H*W) matrix, stored in row major order.
    """
    n_dims = mat_shape[0] * mat_shape[1]

    newx = np.empty(x.shape, dtype=x.dtype)

    for i in range(x.shape[0]):
        newx[i] = x[i].reshape(mat_shape).T.reshape(1, n_dims)

    return newx

