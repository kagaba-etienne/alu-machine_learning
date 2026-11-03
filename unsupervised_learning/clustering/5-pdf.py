#!/usr/bin/env python3
"""Probability density function of a multivariate Gaussian."""

import numpy as np


def pdf(X, m, S):
    """Calculate the PDF of a Gaussian distribution.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data points
            whose PDF should be evaluated.
        m: numpy.ndarray of shape (d,) containing the mean of the
            distribution.
        S: numpy.ndarray of shape (d, d) containing the covariance of
            the distribution.

    Returns:
        numpy.ndarray of shape (n,) containing the PDF values for each
        data point, or None on failure. Values are at least 1e-300.
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None
    if type(m) is not np.ndarray or m.ndim != 1:
        return None
    if type(S) is not np.ndarray or S.ndim != 2:
        return None

    n, d = X.shape
    if m.shape[0] != d or S.shape != (d, d):
        return None

    det = np.linalg.det(S)
    if det <= 0:
        return None

    diff = X - m
    inv = np.linalg.inv(S)
    mahal = np.sum(np.matmul(diff, inv) * diff, axis=1)
    coef = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
    P = coef * np.exp(-0.5 * mahal)
    return np.maximum(P, 1e-300)
