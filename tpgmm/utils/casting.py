from typing import List
import numpy as np
from numpy import ndarray


def str_to_list(s: str) -> List[str]:
    return s.strip("][").split(", ")


def str_to_ndarray(s: str) -> ndarray:
    """converts a given string of floats separated with ', ' into a ndarray of floats

    Args:
        s (str): _description_

    Returns:
        np.ndarray: _description_
    """
    a = str_to_list(s)
    return np.array([float(x) for x in a])


def ssv_to_ndarray(s: str) -> ndarray:
    """convert space separated values to ndarray

    Args:
        s (str): space separated values

    Returns:
        ndarray: array with values
    """
    data = s.split(" ")
    data = map(lambda x: float(x), data)
    return np.array(list(data))