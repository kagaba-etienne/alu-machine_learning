#!/usr/bin/env python3
"""Q-learning for a FrozenLake environment."""

import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Perform Q-learning.

    Falling into a hole changes that step's reward to -1 before the
    Q-table update. Epsilon decays exponentially after each episode
    and never drops below min_epsilon.

    Args:
        env: FrozenLakeEnv instance.
        Q: numpy.ndarray containing the Q-table.
        episodes: total number of episodes to train over.
        max_steps: maximum number of steps per episode.
        alpha: learning rate.
        gamma: discount rate.
        epsilon: initial threshold for epsilon-greedy.
        min_epsilon: minimum value that epsilon should decay to.
        epsilon_decay: decay rate for updating epsilon between episodes.

    Returns:
        The updated Q-table and a list of rewards per episode.
    """
    total_rewards = []
    max_epsilon = epsilon

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)

            if done and reward == 0:
                reward = -1

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state, :]) -
                Q[state, action])

            episode_reward += reward
            state = new_state

            if done:
                break

        epsilon = min_epsilon + (
            (max_epsilon - min_epsilon) *
            np.exp(-epsilon_decay * episode))
        total_rewards.append(episode_reward)

    return Q, total_rewards
