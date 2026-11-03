#!/usr/bin/env python3
"""Epsilon-greedy action selection."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Choose the next action with epsilon-greedy.

    A uniform sample decides whether to explore or exploit. Exploration
    draws an action uniformly. Exploitation takes the action with the
    largest Q-value for this state.

    Args:
        Q: numpy.ndarray containing the Q-table.
        state: current state index.
        epsilon: probability of exploring.

    Returns:
        The index of the next action.
    """
    p = np.random.uniform(0, 1)
    if p > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, Q.shape[1])
    return action
