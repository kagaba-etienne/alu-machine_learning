#!/usr/bin/env python3
"""Load the FrozenLake environment from OpenAI Gym."""

import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load a FrozenLake environment.

    Args:
        desc: None or a list of lists describing a custom map.
        map_name: None or the name of a premade map, such as '4x4'.
        is_slippery: True when movement can slip to a perpendicular
            direction.

    Returns:
        The FrozenLakeEnv instance. When both desc and map_name are
        None, gym generates a random 8x8 map.
    """
    return gym.make('FrozenLake-v0', desc=desc, map_name=map_name,
                    is_slippery=is_slippery)
