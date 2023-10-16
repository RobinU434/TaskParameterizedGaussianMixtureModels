from typing import Callable
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from matplotlib.figure import Figure

from tpgmm.utils.plot.utils import set_axes_equal


def plot3D(plotter: Callable):
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
        """wrapper3D
         for plot function

        Args:
            title (str, optional): title for plot. Defaults to ''.
            fig (_type_, optional): figure object in case you want to add the trajectory plots to an existing figure. Defaults to None.
            ax (_type_, optional): axis object in case you want to add the trajectory plots to an existing axes. Defaults to None.
            legend (bool, optional): if you want to plot a legend. Defaults to True.
            color (str, optional): color for the trajectories. If auto -> different colors for each trajectory. Default is set to 'auto'
            show (bool, optional): to show the figure. Defaults to False.
            save (bool, optional): to save the figure with title (all spaces will be replaces by: '_'). Defaults to False.

        Returns:
            _type_: _description_
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