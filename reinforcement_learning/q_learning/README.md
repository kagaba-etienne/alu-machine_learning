# Q-learning

Tabular Q-learning on OpenAI Gym's FrozenLake. The agent learns an
action-value function for a frozen grid: start on `S`, walk on `F`,
avoid holes `H`, and reach the goal `G`.

## Files

| File | Description |
|------|-------------|
| `0-load_env.py` | Load `FrozenLake-v0` |
| `1-q_init.py` | Initialize a Q-table of zeros |
| `2-epsilon_greedy.py` | Epsilon-greedy action selection |
| `3-q_learning.py` | Train the Q-table |
| `4-play.py` | Play one greedy episode and render the board |

## Tasks

### 0. Load the environment

`load_frozen_lake(desc=None, map_name=None, is_slippery=False)` builds a
`FrozenLake-v0` environment. `desc` is a custom map, `map_name` selects
a premade map such as `4x4`, and `is_slippery` turns on stochastic
slips. When both `desc` and `map_name` are `None`, gym generates a
random 8x8 map.

### 1. Initialize the Q-table

`q_init(env)` returns a zero array with one row per state and one column
per action.

### 2. Epsilon-greedy

`epsilon_greedy(Q, state, epsilon)` draws `p` with
`numpy.random.uniform`. When `p` is greater than `epsilon`, the agent
takes the best action in the Q-table. Otherwise it draws an action with
`numpy.random.randint`.

### 3. Q-learning

`train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
epsilon=1, min_epsilon=0.1, epsilon_decay=0.05)` runs Q-learning. A hole
is given reward `-1` before the Bellman update. After each episode,
epsilon decays exponentially toward `min_epsilon`. The return value is
the updated Q-table and the reward collected in each episode.

### 4. Play

`play(env, Q, max_steps=100)` follows the greedy policy, renders every
board, and returns the episode reward.

## Requirements

- Python 3.5
- NumPy 1.15
- gym 0.7 (`FrozenLake-v0`)
- pycodestyle 2.4
- Module and function docstrings, executable files
