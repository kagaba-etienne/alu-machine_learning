#!/usr/bin/env python3
"""Baum-Welch expectation-maximization for a hidden Markov model."""
import numpy as np


def validate(Observations, Transition, Emission, Initial, iterations):
    """Check shapes and probability constraints of Baum-Welch inputs.

    Args:
        Observations: ndarray of shape (T,) with observation indices.
        Transition: ndarray of shape (M, M), initial transition matrix.
        Emission: ndarray of shape (M, N), initial emission matrix.
        Initial: ndarray of shape (M, 1), initial starting distribution.
        iterations: number of expectation-maximization steps.

    Returns:
        Integer observation indices, or None if the inputs are invalid.
    """
    if type(iterations) is not int or iterations < 1:
        return None
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None
    steps = Observations.shape[0]
    n_states, n_obs = Emission.shape
    if steps < 2 or n_states < 1 or n_obs < 1:
        return None
    if Transition.shape != (n_states, n_states):
        return None
    if Initial.shape != (n_states, 1):
        return None
    arrays = (Observations, Emission, Transition, Initial)
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
    if np.any(Observations < 0) or np.any(Observations >= n_obs):
        return None
    indices = Observations.astype(int)
    if not np.all(np.isclose(indices, Observations)):
        return None
    return indices


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Re-estimate transition and emission probabilities with Baum-Welch.

    Forward and backward probabilities are kept in linear space. The
    starting distribution is held fixed for every iteration.

    Args:
        Observations: ndarray of shape (T,) with observation indices.
        Transition: ndarray of shape (M, M), initial transition matrix.
        Emission: ndarray of shape (M, N), initial emission matrix.
        Initial: ndarray of shape (M, 1), initial starting distribution.
        iterations: number of expectation-maximization steps.

    Returns:
        The converged Transition and Emission matrices, or (None, None)
        on failure.
    """
    indices = validate(
        Observations, Transition, Emission, Initial, iterations)
    if indices is None:
        return None, None
    n_states, n_obs = Emission.shape
    steps = indices.shape[0]
    trans = Transition.astype(float).copy()
    emit = Emission.astype(float).copy()
    start = Initial[:, 0].astype(float)

    for _ in range(iterations):
        alpha = np.zeros((n_states, steps))
        alpha[:, 0] = start * emit[:, indices[0]]
        for time in range(1, steps):
            previous = np.matmul(alpha[:, time - 1], trans)
            alpha[:, time] = previous * emit[:, indices[time]]

        beta = np.zeros((n_states, steps))
        beta[:, -1] = 1
        for time in range(steps - 2, -1, -1):
            coeff = emit[:, indices[time + 1]] * beta[:, time + 1]
            beta[:, time] = np.matmul(trans, coeff)

        xi = np.zeros((n_states, n_states, steps - 1))
        for time in range(steps - 1):
            coeff = emit[:, indices[time + 1]] * beta[:, time + 1]
            numer = (
                alpha[:, time][:, np.newaxis]
                * trans
                * coeff[np.newaxis, :]
            )
            denom = np.sum(numer)
            xi[:, :, time] = numer / denom

        gamma = np.sum(xi, axis=1)
        state_mass = np.sum(gamma, axis=1).reshape((-1, 1))
        trans = np.sum(xi, axis=2) / state_mass
        last = np.sum(xi[:, :, steps - 2], axis=0).reshape((-1, 1))
        gamma = np.hstack((gamma, last))
        emit_mass = np.sum(gamma, axis=1).reshape((-1, 1))
        for symbol in range(n_obs):
            emit[:, symbol] = np.sum(gamma[:, indices == symbol], axis=1)
        emit = emit / emit_mass

    return trans, emit
