from typing import Callable
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from matplotlib.figure import Figure

from tpgmm.utils.plot.utils import set_axes_equal


def plot3D(plotter: Callable):
    """Wrapper for 3D plot function.

    Args:
        plotter (Callable): The plotter function.

    Returns:
        Callable: The inner function that configures and displays the 3D plot.
    """

    def inner(
        title: str = "",
        fig: Figure = None,
        ax: Axes = None,
        legend: bool = False,
        colorbar: bool = False,
        color: str = "auto",
        dpi: int = None,
        alpha: float = 1,
        show: bool = False,
        save: bool = False,
        agg: bool = False,
        **kwargs,
    ):
        """Inner function for configuring and displaying 3D plot.

        Args:
            title (str, optional): Title for the plot. Defaults to ''.
            fig (Figure, optional): Figure object to add the trajectory plots to an existing figure. Defaults to None.
            ax (Axes, optional): Axis object to add the trajectory plots to an existing axes. Defaults to None.
            legend (bool, optional): If True, a legend is plotted. Defaults to False.
            color (str, optional): Color for the trajectories. If 'auto', different colors are used for each trajectory. Default is set to 'auto'.
            show (bool, optional): If True, the figure is displayed. Defaults to False.
            save (bool, optional): If True, the figure is saved with the title (all spaces will be replaced by '_'). Defaults to False.
            agg (bool, optional): If True, sets the backend to 'agg'. Defaults to False.
            **kwargs: Additional keyword arguments for the plotter function.

        Returns:
            Figure, Axes: The configured figure and axes objects.
        """
        if isinstance(dpi, (int, float)):
            mpl.rcParams["figure.dpi"] = dpi
        if agg:
            mpl.use("agg")
        if fig is None:
            fig = plt.figure(figsize=(10, 8))
        if ax is None:
            ax = fig.add_subplot(projection="3d")
        if len(title) > 0:
            ax.set_title(title, fontsize=18)
        if color == "auto":
            color = None

        ax = plotter(ax=ax, color=color, alpha=alpha, **kwargs)

        if legend:
            ax.legend()
        if colorbar:
            Warning("color bar is not implemented yet")
        if show:
            ax = set_axes_equal(ax)
            plt.show()
        if save:
            ax = set_axes_equal(ax)
            if " " in title:
                title = title.replace(" ", "_")
            fig.savefig(title + ".png")

        return fig, ax

    return inner
