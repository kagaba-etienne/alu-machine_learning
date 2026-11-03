#!/usr/bin/env python3
"""Agglomerative clustering with Ward linkage."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Perform agglomerative clustering on a dataset.

    Clusters are formed with Ward linkage. A dendrogram is displayed
    with each cluster in a different color, using dist as the maximum
    cophenetic distance.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        dist: maximum cophenetic distance for all clusters.

    Returns:
        numpy.ndarray of shape (n,) containing the cluster indices for
        each data point.
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method="ward")
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()
    return scipy.cluster.hierarchy.fcluster(
        linkage, t=dist, criterion="distance")
