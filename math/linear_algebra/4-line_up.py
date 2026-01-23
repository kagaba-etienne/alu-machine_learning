#!/usr/bin/env python3
"""
Module that defines a function to line up two matrices
by adding their corresponding elements.
"""


def add_arrays(arr1, arr2):
    """Line up two matrices by adding their corresponding elements.

    Args:
        arr1 (list of int/float): The first matrix.
        arr2 (list of int/float): The second matrix.
    Returns:
        list of list of int/float: The resulting matrix after lining up.
    """
    if len(arr1) != len(arr2):
        return None

    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result
