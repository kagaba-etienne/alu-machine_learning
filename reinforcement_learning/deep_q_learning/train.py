#!/usr/bin/env python3
"""Train a Deep Q-Network agent to play Atari Breakout."""
import gym
import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Permute
from keras.models import Sequential
from keras.optimizers import Adam
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    """Resize Atari frames and scale them for a DQN."""

    def process_observation(self, observation):
        """Resize a frame to 84x84 grayscale.

        Args:
            observation: RGB frame with shape (height, width, 3).

        Returns:
            Grayscale frame with shape (84, 84), dtype uint8.
        """
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        processed = np.array(img)
        return processed.astype('uint8')

    def process_state_batch(self, batch):
        """Scale a batch of stacked frames to [0, 1].

        Args:
            batch: Array of stacked frames stored as uint8.

        Returns:
            The same batch as float32, divided by 255.
        """
        return batch.astype('float32') / 255.

    def process_reward(self, reward):
        """Clip a reward to the range [-1, 1].

        Args:
            reward: Scalar reward from the environment.

        Returns:
            The reward clipped to [-1, 1].
        """
        return np.clip(reward, -1.0, 1.0)


def build_model(nb_actions):
    """Build the convolutional policy network.

    SequentialMemory stacks frames as (window, height, width). A
    permute layer reorders that to channels-last for Conv2D.

    Args:
        nb_actions: Number of discrete actions.

    Returns:
        A Keras Sequential model that maps a frame stack to Q-values.
    """
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def train():
    """Train a DQN agent on Breakout and save the policy network.

    The agent follows an epsilon-greedy policy whose epsilon anneals
    from 1.0 to 0.1. The trained Keras model is written to policy.h5.
    """
    env = gym.make('BreakoutDeterministic-v4')
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    model = build_model(nb_actions)
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=1000000)
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        processor=AtariProcessor(),
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0)
    dqn.compile(Adam(lr=0.00025), metrics=['mae'])
    dqn.fit(env, nb_steps=1750000, visualize=False, verbose=2)
    model.save('policy.h5')
    env.close()


if __name__ == '__main__':
    train()
