#!/usr/bin/env python3
"""Steady state of a regular Markov chain."""
import numpy as np


def regular(P):
    """Steady-state probabilities of a regular Markov chain.

    A chain is regular when some power of its transition matrix has only
    strictly positive entries. The steady state is the unique distribution
    left unchanged by one step.

    Args:
        P: square ndarray of shape (n, n). P[i, j] is the probability of
            transitioning from state i to state j.

    Returns:
        ndarray of shape (1, n), or None on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n_states, n_cols = P.shape
    if n_states == 0 or n_states != n_cols:
        return None
    if not np.all(np.isfinite(P)) or np.any(P < 0):
        return None
    if not np.all(np.isclose(np.sum(P, axis=1), 1)):
        return None

    matrix = P.astype(float)
    power = matrix.copy()
    limit = (n_states - 1) ** 2 + 1
    is_regular = False
    for _ in range(limit):
        if np.all(power > 0):
            is_regular = True
            break
        power = np.matmul(power, matrix)
    if not is_regular:
        return None

    values, vectors = np.linalg.eig(matrix.T)
    index = int(np.argmin(np.abs(values - 1.0)))
    if not np.isclose(values[index], 1.0):
        return None
    steady = np.real(vectors[:, index]).reshape(1, n_states)
    total = np.sum(steady)
    if not np.isfinite(total) or np.isclose(total, 0):
        return None
    steady = steady / total
    if np.any(steady < -1e-8) or not np.all(np.isfinite(steady)):
        return None
    steady = np.maximum(steady, 0)
    steady = steady / np.sum(steady)
    return steady
