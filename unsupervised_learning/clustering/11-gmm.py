#!/usr/bin/env python3
"""Gaussian Mixture Model with scikit-learn."""

import sklearn.mixture


def gmm(X, k):
    """Calculate a GMM from a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: the number of clusters.

    Returns:
        pi: numpy.ndarray of shape (k,) containing the cluster priors.
        m: numpy.ndarray of shape (k, d) containing the centroid means.
        S: numpy.ndarray of shape (k, d, d) containing the covariance
            matrices.
        clss: numpy.ndarray of shape (n,) containing the cluster indices
            for each data point.
        bic: BIC value for the fitted model.
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    clss = model.predict(X)
    return (
        model.weights_,
        model.means_,
        model.covariances_,
        clss,
        model.bic(X),
    )
