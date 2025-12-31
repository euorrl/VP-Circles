from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_population(
    pop: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    log_scale: bool = True,
    cmap: str = "viridis",
    title: Optional[str] = "Population raster",
    show_colorbar: bool = True,
) -> plt.Axes:
    """
    Plot a 2D population raster.

    Parameters
    ----------
    pop : np.ndarray
        2D array (H, W). Values should be non-negative.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure/axes is created.
    log_scale : bool
        If True, use log1p(pop) for better visualization of skewed data.
    cmap : str
        Matplotlib colormap name.
    title : str, optional
        Plot title.
    show_colorbar : bool
        Whether to draw a colorbar.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    pop = np.asarray(pop)

    if pop.ndim != 2:
        raise ValueError(f"`pop` must be a 2D array, got shape {pop.shape}")

    if np.any(pop < 0):
        raise ValueError("`pop` must be non-negative (found values < 0).")

    data = np.log1p(pop) if log_scale else pop

    if ax is None:
        _, ax = plt.subplots()

    im = ax.imshow(data, cmap=cmap, origin="upper")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")
    if title:
        ax.set_title(title + (" (log1p)" if log_scale else ""))

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax
