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
    if steps < 1 or n_states < 1 or n_obs < 1:
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
    """Re-estimate a hidden Markov model with the Baum-Welch algorithm.

    Forward and backward probabilities are scaled at each time step so the
    products stay in range. The starting distribution is updated internally
    and is not returned.

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
    start = Initial.astype(float).copy()

    for _ in range(iterations):
        alpha = np.zeros((n_states, steps))
        scale = np.zeros(steps)
        alpha[:, 0] = start[:, 0] * emit[:, indices[0]]
        scale[0] = np.sum(alpha[:, 0])
        if scale[0] == 0 or not np.isfinite(scale[0]):
            return None, None
        alpha[:, 0] /= scale[0]
        for time in range(1, steps):
            previous = np.matmul(alpha[:, time - 1], trans)
            alpha[:, time] = previous * emit[:, indices[time]]
            scale[time] = np.sum(alpha[:, time])
            if scale[time] == 0 or not np.isfinite(scale[time]):
                return None, None
            alpha[:, time] /= scale[time]

        beta = np.zeros((n_states, steps))
        beta[:, -1] = 1.0 / scale[-1]
        for time in range(steps - 2, -1, -1):
            coeff = emit[:, indices[time + 1]] * beta[:, time + 1]
            beta[:, time] = np.matmul(trans, coeff) / scale[time]

        gamma = alpha * beta
        gamma_sum = np.sum(gamma, axis=0)
        if np.any(gamma_sum == 0) or not np.all(np.isfinite(gamma)):
            return None, None
        gamma /= gamma_sum

        if steps > 1:
            xi_sum = np.zeros((n_states, n_states))
            for time in range(steps - 1):
                coeff = emit[:, indices[time + 1]] * beta[:, time + 1]
                numer = (
                    alpha[:, time][:, np.newaxis]
                    * trans
                    * coeff[np.newaxis, :]
                )
                denom = np.sum(numer)
                if denom == 0 or not np.isfinite(denom):
                    return None, None
                xi_sum += numer / denom
            denom_t = np.sum(gamma[:, :-1], axis=1).reshape((-1, 1))
            if np.any(denom_t == 0):
                return None, None
            trans = xi_sum / denom_t

        denom_e = np.sum(gamma, axis=1)
        if np.any(denom_e == 0):
            return None, None
        new_emit = np.zeros_like(emit)
        for symbol in range(n_obs):
            mask = indices == symbol
            new_emit[:, symbol] = np.sum(gamma[:, mask], axis=1) / denom_e
        emit = new_emit
        start = gamma[:, 0].reshape((-1, 1))
        if not np.all(np.isfinite(trans)) or not np.all(np.isfinite(emit)):
            return None, None

    return trans, emit
