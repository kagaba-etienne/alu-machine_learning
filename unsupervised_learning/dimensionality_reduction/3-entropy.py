#!/usr/bin/env python3
"""Shannon entropy and Gaussian affinities for one t-SNE point."""
import numpy as np


def HP(Di, beta):
    """Calculate Shannon entropy and P affinities for one point.

    Args:
        Di: numpy.ndarray of shape (n - 1,) of squared distances from
            one point to every other point.
        beta: numpy.ndarray of shape (1,) with the Gaussian precision.

    Returns:
        Hi: Shannon entropy of the affinities, base 2.
        Pi: numpy.ndarray of shape (n - 1,) with the P affinities.
    """
    Pi = np.exp(-Di * beta)
    sum_Pi = np.sum(Pi)
    Pi = Pi / sum_Pi
    Hi = np.log2(sum_Pi) + np.sum(Di * Pi * beta) / np.log(2)
    return Hi, Pi
