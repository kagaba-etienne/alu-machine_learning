#!/usr/bin/env python3
"""Play one episode with a trained Q-learning agent."""

import numpy as np


def play(env, Q, max_steps=100):
    """Play an episode by always exploiting the Q-table.

    The board is rendered after reset and after every action.

    Args:
        env: FrozenLakeEnv instance.
        Q: numpy.ndarray containing the Q-table.
        max_steps: maximum number of steps in the episode.

    Returns:
        The total reward for the episode.
    """
    state = env.reset()
    total_rewards = 0
    env.render()

    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        total_rewards += reward
        env.render()
        state = new_state
        if done:
            break

    return total_rewards
