#!/usr/bin/env python3
"""Play Atari Breakout with a trained DQN policy network."""
import gym
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy


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


def play():
    """Load policy.h5 and render Breakout with a greedy policy.

    The agent always chooses the action with the largest Q-value.
    """
    env = gym.make('BreakoutDeterministic-v4')
    np.random.seed(123)
    env.seed(123)

    model = load_model('policy.h5')
    nb_actions = model.output_shape[-1]
    memory = SequentialMemory(limit=10000, window_length=WINDOW_LENGTH)
    policy = GreedyQPolicy()
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        test_policy=policy,
        memory=memory,
        processor=AtariProcessor(),
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0)
    dqn.compile(Adam(lr=0.00025), metrics=['mae'])
    dqn.test(env, nb_episodes=5, visualize=True)
    env.close()


if __name__ == '__main__':
    play()
