#!/usr/bin/env python3
"""State distribution of a Markov chain after a number of steps."""
import numpy as np


def markov_chain(P, s, t=1):
    """Probability of each state after t iterations of a Markov chain.

    Args:
        P: square ndarray of shape (n, n). P[i, j] is the probability of
            transitioning from state i to state j.
        s: ndarray of shape (1, n), the starting distribution.
        t: number of iterations the chain has been through.

    Returns:
        ndarray of shape (1, n), or None on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if not isinstance(s, np.ndarray) or s.ndim != 2:
        return None
    n_states, n_cols = P.shape
    if n_states == 0 or n_states != n_cols or s.shape != (1, n_states):
        return None
    if type(t) is not int or t < 1:
        return None
    if not np.all(np.isfinite(P)) or not np.all(np.isfinite(s)):
        return None
    try:
        step = np.linalg.matrix_power(P.astype(float), t)
    except (np.linalg.LinAlgError, ValueError):
        return None
    result = np.matmul(s.astype(float), step)
    if result.shape != (1, n_states) or not np.all(np.isfinite(result)):
        return None
    return result
