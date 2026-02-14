#!/usr/bin/env python3
"""
Calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial represented
    by a list of coefficients.

    Args:
        poly (list): A list of coefficients where the index
        is the power of x.
        C (int): The integration constant.

    Returns:
        list: A new list of coefficients representing the integral.
        None: If poly or C are not valid.
    """
    # 1. Validation: Ensure poly is a valid, non-empty list and C is an integer
    if type(poly) is not list or len(poly) == 0:
        return None
    if type(C) is not int:
        return None
    for coeff in poly:
        if type(coeff) not in (int, float):
            return None

    # 2. Handle the edge case of a zero polynomial
    # If the polynomial is just 0, its integral is just the constant C
    if poly == [0]:
        return [C]

    # 3. Calculation: Start the new list with the integration constant C
    # at index 0
    integral = [C]

    # 4. Apply the integration power rule
    for i, coeff in enumerate(poly):
        # The new power will be i + 1
        new_coeff = coeff / (i + 1)

        # If the coefficient is a whole number, represent it as an int
        if new_coeff % 1 == 0:
            new_coeff = int(new_coeff)

        integral.append(new_coeff)

    # 5. Trim trailing zeros to make the list as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
