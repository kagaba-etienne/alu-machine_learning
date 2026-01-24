#!/usr/bin/env python3
"""
Module that defines a function to
concatenate two matrices along the first axis.
"""


def cat_arrays(arr1, arr2):
    """Concatenate two matrices along the first axis.

    Args:
        arr1 (list of int/float): The first matrix.
        arr2 (list of int/float): The second matrix.
    Returns:
        list of int/float: The resulting matrix after concatenation.
    """
    return arr1 + arr2
