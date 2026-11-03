#!/usr/bin/env python3
"""Kullback-Leibler cost of a t-SNE map."""
import numpy as np


def cost(P, Q):
    """Calculate the Kullback-Leibler cost of a t-SNE map.

    Args:
        P: numpy.ndarray of shape (n, n) with the P affinities.
        Q: numpy.ndarray of shape (n, n) with the Q affinities.

    Returns:
        The cost of the transformation.
    """
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    return np.sum(P * np.log(P / Q))
