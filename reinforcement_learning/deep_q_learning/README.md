# Deep Q-Learning

Train a Deep Q-Network to play Atari Breakout, then watch the trained
policy play.

## Concepts

### Deep Q-learning

Q-learning estimates the value of taking an action in a state. Deep Q-learning
represents that action-value function with a neural network and trains the
network on the temporal-difference error between the predicted value and a
bootstrapped target.

### Policy network

The policy network maps a stack of four 84×84 grayscale frames to one Q-value
per action. The agent plays by choosing the action with the largest Q-value.
During training that choice is epsilon-greedy: with probability epsilon the
agent takes a random action. Epsilon anneals from 1.0 to 0.1 over the first
1,000,000 steps. `train.py` saves this network as `policy.h5`.

### Replay memory

Each transition (state, action, reward, next state, done) is stored in a
replay buffer (`SequentialMemory`, one million transitions). Updates sample
random minibatches from that buffer, so successive gradients are not tied to
one correlated stretch of gameplay and each transition can be reused.

### Target network

A second network, a copy of the policy network, produces the Q-value inside
the temporal-difference target. Its weights are copied from the policy network
every 10,000 steps (`target_model_update`).

### Why two networks

If the same network supplied both the predicted Q-value and the target, every
update would move the target. Learning then chases a moving target and
diverges. Holding the target network fixed for many steps keeps the target
stable while the policy network is trained.

### keras-rl

[keras-rl](https://github.com/keras-rl/keras-rl) implements
reinforcement-learning agents on top of Keras. A script builds a Keras model,
wraps it in an agent with a policy and a memory, compiles the agent, and calls
`fit` or `test`.

This project uses:

- `DQNAgent` for training and playback
- `SequentialMemory` for the replay buffer and the four-frame window
- `EpsGreedyQPolicy`, annealed with `LinearAnnealedPolicy`, while training
- `GreedyQPolicy` while playing

## Files

| File | Description |
|------|-------------|
| `train.py` | Train the agent and save the policy network to `policy.h5` |
| `play.py` | Load `policy.h5` and render games with a greedy policy |

## Usage

The scripts are written for Python 3.5 with numpy 1.15, gym 0.17.2 (Atari ROMs
installed), keras 2.2.5, keras-rl 0.4.2, Pillow, and h5py.

```bash
./train.py
./play.py
```

`train.py` trains for 1,750,000 steps on `BreakoutDeterministic-v4` and writes
`policy.h5` in the working directory. A score above 10 needs that full run.
`play.py` opens a window and plays 5 episodes, so it needs a display.
