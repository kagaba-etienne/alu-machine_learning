#!/usr/bin/env python3
"""Initialize a Q-table for a discrete environment."""

import numpy as np


def q_init(env):
    """Initialize the Q-table to zeros.

    Args:
        env: FrozenLakeEnv instance.

    Returns:
        numpy.ndarray of zeros with shape
        (env.observation_space.n, env.action_space.n).
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
