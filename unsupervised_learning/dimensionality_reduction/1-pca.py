#!/usr/bin/env python3
"""Principal component analysis to a fixed dimensionality."""
import numpy as np


def pca(X, ndim):
    """Perform PCA on a data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        ndim: the new dimensionality of the transformed data.

    Returns:
        T: numpy.ndarray of shape (n, ndim), the transformed data.
    """
    X_centered = X - np.mean(X, axis=0)
    _, _, vh = np.linalg.svd(X_centered)
    return np.matmul(X_centered, vh[:ndim].T)
