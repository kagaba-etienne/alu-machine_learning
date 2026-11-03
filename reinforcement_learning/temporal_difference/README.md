# Temporal Difference

This project implements model-free prediction and control for an OpenAI Gym environment.

## Algorithms

### Monte Carlo (`0-monte_carlo.py`)
Estimates a state-value function from complete episode returns.
- `monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)`: Runs constant-alpha Monte Carlo and returns the updated value estimate

### TD(λ) (`1-td_lambtha.py`)
Estimates a state-value function with one-step bootstrapping and eligibility traces.
- `td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99)`: Runs backward-view TD(λ) and returns the updated value estimate

### SARSA(λ) (`2-sarsa_lambtha.py`)
Learns an action-value function with an on-policy epsilon-greedy policy.
- `epsilon_greedy(Q, state, epsilon)`: Selects a random action or the greedy action
- `sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05)`: Runs SARSA(λ) and returns the updated Q table
