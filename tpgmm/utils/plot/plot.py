from typing import List, Union

import numpy as np
from matplotlib.axes import Axes
from numpy import ndarray
import matplotlib.colors as mcolors 

from tpgmm.utils.plot.decorator import plot3D


@plot3D
def plot_trajectories(
    trajectories: ndarray, ax=None, color: str = None, alpha: float = 1
) -> Axes:
    """Plot demo trajectories in 3D space.

    Args:
        trajectories (ndarray): Demo data with shape (num_demos, demo_length, 3).
        ax (Axes, optional): Axis object in case you want to add the trajectory plots to an existing axes. Defaults to None.
        color (str, optional): Color for the trajectories. If 'auto', different colors are used for each trajectory. Default is set to None.

    Returns:
        Axes: The configured axis object.
    """
    # plot demo trajectories
    for demo_idx, demo in enumerate(trajectories):
        ax.plot(*(demo.T), label=f"Demo {demo_idx}", color=color, alpha=alpha)

    return ax


@plot3D
def scatter(
    data: Union[List[ndarray], ndarray],
    marker: str = ".",
    color: str = None,
    alpha: float = 1,
    ax=None,
) -> Axes:
    """Scatter 3D cluster data.

    Args:
        data (Union[List[ndarray], ndarray]): Data points to scatter with shape (num_clusters, num_points_per_cluster, 3).
        marker (str, optional): Marker style for scatter plot. Defaults to ".".
        ax (Axes, optional): Axis object in case you want to add the trajectory plots to an existing axes. Defaults to None.
        color (str, optional): Color for the trajectories. If 'auto', different colors are used for each trajectory. Default is set to 'auto'.

    Returns:
        Axes: The configured axis object.
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
    means: ndarray, covs: ndarray, ax=None, color: str = None, alpha: float = 1
) -> Axes:
    """Plot Gaussian ellipsoids in 3D space.

    Args:
        means (ndarray): Means of each cluster with shape (num_cluster, 3).
        cov (ndarray): Covariance matrix with shape (num_cluster, 3, 3).
        ax (Axes, optional): Axis object in case you want to add the trajectory plots to an existing axes. Defaults to None.
        color (str, optional): Color for the trajectories. If 'auto', different colors are used for each trajectory. Default is set to 'auto'.

    Returns:
        Axes: The configured axis object.
    """
    if color is None:
        color = [list(mcolors.TABLEAU_COLORS.keys())[idx % len(mcolors.TABLEAU_COLORS)] for idx in range(len(means))]
    elif len(color) == 1:
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
