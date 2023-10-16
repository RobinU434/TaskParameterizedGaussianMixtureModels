from typing import Callable, List, Union

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from tpgmm.utils.plot.decorator import plot3D



@plot3D
def plot_trajectories(
    trajectories: np.ndarray, ax=None, color: str = None, alpha: float = 1
) -> Axes:
    """plot demo trajectories

    Args:
        trajectories (np.ndarray): demo data. Shape (num_demos, demo_length, 3)
        ax (_type_, optional): axis object in case you want to add the trajectory plots to an existing axes. Defaults to None.
        color (str, optional): color for the trajectories. If auto -> different colors for each trajectory. Default is set to None
    Returns:
        _type_: _description_
    """
    # plot demo trajectories
    for demo_idx, demo in enumerate(trajectories):
        ax.plot(*(demo.T), label=f"Demo {demo_idx}", color=color, alpha=alpha)

    return ax


@plot3D
def scatter(
    data: Union[List[np.ndarray], np.ndarray],
    marker: str = ".",
    color: str = None,
    alpha: float = 1,
    ax=None,
) -> Axes:
    """scatter 3d cluster data

    Args:
        data (Union[List[np.ndarray], np.ndarray]): data points to scatter. Shape (num_clusters, num_points_per_cluster, 3)
        marker (str, optional): _description_. Defaults to ".".
        ax (_type_, optional): axis object in case you want to add the trajectory plots to an existing axes. Defaults to None.
        color (str, optional): color for the trajectories. If auto -> different colors for each trajectory. Default is set to 'auto'

    Returns:
        _type_: _description_
    """
    if color is None or (len(color) == 1 and isinstance(color, str)):
        color = [color for _ in data]

    for frame_idx, (cluster, c) in enumerate(zip(data, color)):
        ax.scatter3D(
            *(cluster.T),
            s=50,
            label=f"frame: {frame_idx}",
            alpha=alpha,
            marker=marker,
            c=c,
        )

    return ax


@plot3D
def plot_ellipsoids(
    means: np.ndarray, covs: np.ndarray, ax=None, color: str = None, alpha: float = 1
) -> Axes:
    """plot gaussian ellipsoids in 3D space

    Args:
        means (np.ndarray): means of each cluster. Shape: (num_cluster, 3)
        cov (np.ndarray): covariance matrix. Shape: (num_cluster, 3, 3)
        ax (_type_, optional): axis object in case you want to add the trajectory plots to an existing axes. Defaults to None.
        color (str, optional): color for the trajectories. If auto -> different colors for each trajectory. Default is set to 'auto'

    Returns:
        _type_: _description_
    """
    if color is None or len(color) == 1:
        color = [color for _ in means]
    for idx, (mean, cov, c) in enumerate(zip(means, covs, color)):
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # generate sphere
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        # apply transformation to align with eigenvectors
        ellipsoid = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        ellipsoid = np.dot(ellipsoid, np.sqrt(np.diag(eigenvalues))) @ eigenvectors.T

        # shift ellipsoid with mean
        ellipsoid += mean

        # make it compliant with writeframes
        ellipsoid = ellipsoid.reshape((*x.shape, 3))
        ax.plot_wireframe(
            ellipsoid[:, :, 0],
            ellipsoid[:, :, 1],
            ellipsoid[:, :, 2],
            color=c,
            label=f"distr: {idx}",
            alpha=alpha,
        )

    return ax
