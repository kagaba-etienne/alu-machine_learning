#!/usr/bin/env python3
"""Gradients of the t-SNE low-dimensional map."""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """Calculate the gradients of the t-SNE map.

    The factor of 4 from the paper is omitted.

    Args:
        Y: numpy.ndarray of shape (n, ndim) with the low-dimensional
            points.
        P: numpy.ndarray of shape (n, n) with the P affinities.

    Returns:
        dY: numpy.ndarray of shape (n, ndim) with the gradients.
        Q: numpy.ndarray of shape (n, n) with the Q affinities.
    """
    Q, num = Q_affinities(Y)
    affinity = (P - Q) * num
    weighted = np.sum(affinity, axis=1, keepdims=True)
    dY = weighted * Y - np.matmul(affinity, Y)
    return dY, Q
