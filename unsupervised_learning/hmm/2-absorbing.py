#!/usr/bin/env python3
"""Absorbing Markov chains."""
import numpy as np


def absorbing(P):
    """Determine whether a Markov chain is absorbing.

    A chain is absorbing when it has at least one absorbing state and every
    state can reach an absorbing state. A state i is absorbing when P[i, i]
    is 1.

    Args:
        P: square ndarray of shape (n, n). P[i, j] is the probability of
            transitioning from state i to state j.

    Returns:
        True if P is an absorbing chain, False otherwise or on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    n_states, n_cols = P.shape
    if n_states == 0 or n_states != n_cols:
        return False
    if not np.all(np.isfinite(P)) or np.any(P < 0):
        return False
    if not np.all(np.isclose(np.sum(P, axis=1), 1)):
        return False

    is_absorb = np.isclose(np.diag(P), 1.0)
    if not np.any(is_absorb):
        return False
    absorb_ids = set(np.nonzero(is_absorb)[0].tolist())
    edges = []
    for state in range(n_states):
        nxt = np.nonzero(P[state] > 0)[0]
        edges.append([int(node) for node in nxt])

    for start in range(n_states):
        seen = [False] * n_states
        stack = [start]
        seen[start] = True
        found = False
        while stack:
            node = stack.pop()
            if node in absorb_ids:
                found = True
                break
            for nxt in edges[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        if not found:
            return False
    return True
