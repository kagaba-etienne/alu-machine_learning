#!/usr/bin/env python3
"""t-SNE embedding of a high-dimensional data set."""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """Perform a t-SNE transformation.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        ndims: dimension of the embedding.
        idims: dimension of the intermediate PCA representation.
        perplexity: perplexity of the Gaussian affinities.
        iterations: number of gradient steps.
        lr: learning rate.

    Returns:
        Y: numpy.ndarray of shape (n, ndims) with the embedding.
    """
    X = pca(X, idims)
    n, _ = X.shape
    P = P_affinities(X, perplexity=perplexity)
    Y = np.random.randn(n, ndims)
    iY = np.zeros((n, ndims))
    P = P * 4
    for i in range(iterations):
        dY, Q = grads(Y, P)
        if i < 20:
            momentum = 0.5
        else:
            momentum = 0.8
        iY = momentum * iY - lr * dY
        Y = Y + iY
        Y = Y - np.mean(Y, axis=0)
        if (i + 1) % 100 == 0:
            C = cost(P, Q)
            print("Cost at iteration {}: {}".format(i + 1, C))
        if (i + 1) == 100:
            P = P / 4
    return Y
