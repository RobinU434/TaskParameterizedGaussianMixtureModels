from numpy import ndarray
import numpy as np


def multivariate_gauss_cdf(data: ndarray, mean: ndarray, cov: ndarray) -> ndarray:
    """computes the probability of given data points if there are in a given gaussian distribution

    Args:
        data (ndarray): shape: (num_points, num_features)
        mean (ndarray): shape: (num_features)
        cov (ndarray): shape: (num_features, num_features)

    Returns:
        ndarray: shape: (num_points)
    """
    num_features = len(mean)
    diff = data - mean
    a = diff @ np.linalg.inv(cov)
    diag = np.einsum("ij,ji->i", a, diff.T)
    return np.exp(-0.5 * diag) / np.sqrt(
        np.power(2 * np.pi, num_features) * np.linalg.det(cov)
    )
