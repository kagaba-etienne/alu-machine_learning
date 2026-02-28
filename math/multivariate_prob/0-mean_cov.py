#!/usr/bin/env python3
"""Module for calculating mean and covariance of a data set."""

import numpy as np


def mean_cov(X):
    """Calculate the mean and covariance of a data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.

    Returns:
        mean: numpy.ndarray of shape (1, d) containing the mean.
        cov: numpy.ndarray of shape (d, d) containing the covariance matrix.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / (n - 1)

    return mean, cov
