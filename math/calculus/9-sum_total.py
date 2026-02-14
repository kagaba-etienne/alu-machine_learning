#!/usr/bin/env python3
'''
Calculates the sum of squares
'''


def summation_i_squared(n):
    """
    Calculate the sum of squares
    
    Args:
        n (integer): the stopping number
    Returns:
        int: if n is a valid integer
        None: if n is not a valid integer
    """
    if not isinstance(n, int):
        return None
    
    return (n * (n + 1) * (2 * n + 1))//6
