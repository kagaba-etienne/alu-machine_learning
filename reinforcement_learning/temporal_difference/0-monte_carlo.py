#!/usr/bin/env python3
"""Monte Carlo prediction for state-value estimation."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """Estimate state values with constant-alpha Monte Carlo.

    Walking each episode backward, a state is updated when it does not
    appear among the first ``episode`` transitions of that episode.

    Args:
        env: OpenAI Gym environment instance.
        V: Value estimates with shape (s,).
        policy: Function mapping a state to an action.
        episodes: Number of episodes to run.
        max_steps: Maximum steps in an episode.
        alpha: Learning rate.
        gamma: Discount rate.

    Returns:
        The updated value estimate V.
    """
    for episode in range(episodes):
        state = env.reset()
        trajectory = []
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, reward))
            if done:
                break
            state = next_state
        trajectory = np.array(trajectory, dtype=int)
        ret = 0
        for state, reward in reversed(trajectory):
            ret = gamma * ret + reward
            if state not in trajectory[:episode, 0]:
                V[state] = V[state] + alpha * (ret - V[state])
    return V
