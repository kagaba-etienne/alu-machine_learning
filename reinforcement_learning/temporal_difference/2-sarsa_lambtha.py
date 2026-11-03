#!/usr/bin/env python3
"""SARSA(lambda) control with accumulating eligibility traces."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Choose an action with an epsilon-greedy policy.

    Args:
        Q: Action-value table with shape (s, a).
        state: Current state.
        epsilon: Probability of selecting a random action.

    Returns:
        The selected action.
    """
    if np.random.uniform() < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Learn an action-value function with on-policy SARSA(lambda).

    Args:
        env: OpenAI Gym environment instance.
        Q: Action-value table with shape (s, a).
        lambtha: Eligibility-trace decay factor.
        episodes: Number of episodes to run.
        max_steps: Maximum steps in an episode.
        alpha: Learning rate.
        gamma: Discount rate.
        epsilon: Initial threshold for epsilon-greedy exploration.
        min_epsilon: Lowest value epsilon can decay to.
        epsilon_decay: Exponential decay rate applied between episodes.

    Returns:
        The updated action-value table Q.
    """
    initial_epsilon = epsilon
    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        eligibility = np.zeros_like(Q)
        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            delta = (reward + gamma * Q[next_state, next_action]
                     - Q[state, action])
            eligibility[state, action] += 1
            Q += alpha * delta * eligibility
            eligibility *= gamma * lambtha
            if done:
                break
            state = next_state
            action = next_action
        epsilon = min_epsilon + (
            (initial_epsilon - min_epsilon)
            * np.exp(-epsilon_decay * episode))
    return Q
