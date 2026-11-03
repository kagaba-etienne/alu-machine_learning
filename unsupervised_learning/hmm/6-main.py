#!/usr/bin/env python3
"""Test the Baum-Welch algorithm."""
import numpy as np

baum_welch = __import__('6-baum_welch').baum_welch

if __name__ == '__main__':
    np.random.seed(1)
    Emission = np.array([[0.90, 0.10, 0.00],
                         [0.40, 0.50, 0.10]])
    Transition = np.array([[0.60, 0.4],
                           [0.30, 0.70]])
    Initial = np.array([0.5, 0.5])
    Hidden = [np.random.choice(2, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(2, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)
    Observations = []
    for state in Hidden:
        Observations.append(np.random.choice(3, p=Emission[state]))
    Observations = np.array(Observations)
    T_test = np.ones((2, 2)) / 2
    E_test = np.abs(np.random.randn(2, 3))
    E_test = E_test / np.sum(E_test, axis=1).reshape((-1, 1))
    learned_t, learned_e = baum_welch(
        Observations, T_test, E_test, Initial.reshape((-1, 1)))
    print(np.round(learned_t, 2))
    print(np.round(learned_e, 2))
