import logging
from typing import Iterable, Tuple
from numpy import ndarray
import numpy as np

def subscript(*args) -> Tuple[slice]:
    """builds a slicing tuple to slice a multidimensional ArrayLike structure

    Returns:
        Tuple[slice]: tuple of slicing information
    """

    # TODO: write tests
    selection = []
    for arg in args:
        if isinstance(arg, Iterable):
            selection.append(arg)
        elif isinstance(arg, int):
            selection.append(slice(arg, arg + 1))
        else:
            selection.append(slice(None))
    return tuple(selection)

def identity_like(array: ndarray) -> ndarray:
    """if the last two axis are the same create an identity matrix with the same shape

    Args:
        array (ndarray): shape: (..., a, a) with a = dimension of covariance matrix

    Returns:
        ndarray: identity matrix like with shape: (..., a, a)
    """
    array_shape = array.shape
    if array_shape[-1] != array_shape[-2]:
        logging.error(
            "Not possible to calculate a identity matrix because the last two axes are different"
        )
        return
    identity = np.zeros_like(array)
    selection = [None] * len(array_shape)
    diagonal_index = list(range(array_shape[-1]))
    selection[-2] = diagonal_index
    selection[-1] = diagonal_index
    selection = subscript(*selection)
    identity[selection] = 1
    return identity

def get_subarray(data: ndarray, axes: Iterable[int], indices: Iterable[Iterable[int]]):
    # test every element of indices is list[int]
    for ele in indices:
        assert isinstance(ele, Iterable)

    num_data_axes = len(data.shape)
    for axis, index in zip(axes, indices):
        selection = [None] * num_data_axes
        selection[axis] = index
        selection = subscript(*selection)
        data = data[selection]
    return data
