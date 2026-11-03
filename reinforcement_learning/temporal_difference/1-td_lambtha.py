#!/usr/bin/env python3
"""TD(lambda) prediction with accumulating eligibility traces."""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Estimate state values with backward-view TD(lambda).

    Args:
        env: OpenAI Gym environment instance.
        V: Value estimates with shape (s,).
        policy: Function mapping a state to an action.
        lambtha: Eligibility-trace decay factor.
        episodes: Number of episodes to run.
        max_steps: Maximum steps in an episode.
        alpha: Learning rate.
        gamma: Discount rate.

    Returns:
        The updated value estimate V.
    """
    for _ in range(episodes):
        state = env.reset()
        eligibility = np.zeros_like(V)
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            delta = reward + gamma * V[next_state] - V[state]
            eligibility[state] += 1
            V += alpha * delta * eligibility
            eligibility *= gamma * lambtha
            if done:
                break
            state = next_state
    return V
