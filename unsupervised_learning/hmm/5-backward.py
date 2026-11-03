#!/usr/bin/env python3
"""Backward algorithm for a hidden Markov model."""
import numpy as np


def validate(Observation, Emission, Transition, Initial):
    """Check shapes and probability constraints of HMM inputs.

    Args:
        Observation: ndarray of shape (T,) with observation indices.
        Emission: ndarray of shape (N, M). Emission[i, j] is the
            probability of observing j from hidden state i.
        Transition: ndarray of shape (N, N). Transition[i, j] is the
            probability of moving from hidden state i to j.
        Initial: ndarray of shape (N, 1), the starting distribution.

    Returns:
        Integer observation indices, or None if the inputs are invalid.
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None
    steps = Observation.shape[0]
    n_states, n_obs = Emission.shape
    if steps < 1 or n_states < 1 or n_obs < 1:
        return None
    if Transition.shape != (n_states, n_states):
        return None
    if Initial.shape != (n_states, 1):
        return None
    arrays = (Observation, Emission, Transition, Initial)
    if not all(np.all(np.isfinite(arr)) for arr in arrays):
        return None
    if np.any(Emission < 0) or np.any(Transition < 0) or np.any(Initial < 0):
        return None
    if not np.all(np.isclose(np.sum(Emission, axis=1), 1)):
        return None
    if not np.all(np.isclose(np.sum(Transition, axis=1), 1)):
        return None
    if not np.isclose(np.sum(Initial), 1):
        return None
    if np.any(Observation < 0) or np.any(Observation >= n_obs):
        return None
    indices = Observation.astype(int)
    if not np.all(np.isclose(indices, Observation)):
        return None
    return indices


def backward(Observation, Emission, Transition, Initial):
    """Run the backward algorithm.

    Args:
        Observation: ndarray of shape (T,) with observation indices.
        Emission: ndarray of shape (N, M). Emission[i, j] is the
            probability of observing j from hidden state i.
        Transition: ndarray of shape (N, N). Transition[i, j] is the
            probability of moving from hidden state i to j.
        Initial: ndarray of shape (N, 1), the starting distribution.

    Returns:
        P, B. P is the likelihood of the observations. B has shape (N, T)
        and B[i, j] is the probability of the future observations given
        hidden state i at time j. (None, None) on failure.
    """
    indices = validate(Observation, Emission, Transition, Initial)
    if indices is None:
        return None, None
    n_states, _ = Emission.shape
    steps = indices.shape[0]
    bwd = np.zeros((n_states, steps))
    bwd[:, -1] = 1
    for time in range(steps - 2, -1, -1):
        emit = Emission[:, indices[time + 1]] * bwd[:, time + 1]
        bwd[:, time] = np.matmul(Transition, emit)
    likelihood = np.sum(
        Initial[:, 0] * Emission[:, indices[0]] * bwd[:, 0])
    if not np.isfinite(likelihood):
        return None, None
    return likelihood, bwd
