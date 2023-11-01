from typing import List
import numpy as np
from numpy import ndarray


def str_to_list(s: str) -> List[str]:
    """Converts a string representation of a list into an actual list of strings.

    Args:
        s (str): The input string to be converted.

    Returns:
        List[str]: The converted list of strings.
    """

    return s.strip("][").split(", ")


def str_to_ndarray(s: str) -> ndarray:
    """Converts a string of floats separated with ', ' into a NumPy ndarray of floats.

    Args:
        s (str): The input string of floats separated by ', '.

    Returns:
        np.ndarray: The NumPy ndarray of floats.
    """
    a = str_to_list(s)
    return np.array([float(x) for x in a])


def ssv_to_ndarray(s: str) -> ndarray:
    """Converts space-separated values to a NumPy ndarray.

    Args:
        s (str): The space-separated values to be converted.

    Returns:
        ndarray: The array containing the converted values.
    """
    data = s.split(" ")
    data = map(lambda x: float(x), data)
    return np.array(list(data))
