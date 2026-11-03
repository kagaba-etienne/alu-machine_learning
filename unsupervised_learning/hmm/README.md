# Hidden Markov Models

Markov chains and hidden Markov models: one-step distributions, regular and absorbing chains, and the Forward, Viterbi, Backward, and Baum-Welch algorithms.

Only `numpy` is used. Each module is executable and starts with `#!/usr/bin/env python3`.

## Files

| File | Description |
|------|-------------|
| `0-markov_chain.py` | Distribution after `t` steps, `s @ P^t` |
| `1-regular.py` | Steady state of a regular chain, or `None` |
| `2-absorbing.py` | Whether every state can reach an absorbing state |
| `3-forward.py` | Forward likelihood and forward lattice |
| `4-viterbi.py` | Most likely hidden-state path and its probability |
| `5-backward.py` | Backward likelihood and backward lattice |
| `6-baum_welch.py` | Baum-Welch re-estimation of transition and emission |

## Concepts

A Markov chain satisfies the Markov property: the next state depends only on the current state. `P[i, j]` is the probability of moving from state `i` to state `j`. A distribution `s` is stationary when `s @ P = s`. A chain is regular when some power of `P` is strictly positive, which gives a unique steady state. A state is absorbing when its self-transition is 1. The chain is absorbing when at least one absorbing state exists and every state can reach one.

A hidden Markov model adds hidden states and observations. `Emission[i, j]` is the probability of observation `j` from hidden state `i`. The forward algorithm sums the probability of the observations over all hidden paths. Viterbi keeps only the most likely path. The backward algorithm computes the probability of the future observations from each state. Baum-Welch alternates those two passes (expectation) with re-estimation of the transition and emission matrices (maximization). Scaling is applied inside Baum-Welch so long sequences do not underflow.

## Usage

From this directory:

```bash
./0-main.py
./1-main.py
./2-main.py
./3-main.py
./4-main.py
./5-main.py
./6-main.py
```
