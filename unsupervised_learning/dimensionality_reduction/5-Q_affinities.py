#!/usr/bin/env python3
"""Student-t Q affinities for t-SNE."""
import numpy as np


def Q_affinities(Y):
    """Calculate the Q affinities of a low-dimensional map.

    Args:
        Y: numpy.ndarray of shape (n, ndim) with the low-dimensional
            points.

    Returns:
        Q: numpy.ndarray of shape (n, n) with the Q affinities.
        num: numpy.ndarray of shape (n, n) with the unnormalized
            affinities.
    """
    sum_Y = np.sum(np.square(Y), axis=1)
    D = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)
    num = 1.0 / (1.0 + D)
    np.fill_diagonal(num, 0.0)
    Q = num / np.sum(num)
    return Q, num
