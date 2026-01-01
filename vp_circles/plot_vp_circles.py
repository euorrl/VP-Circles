from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_vp_circles(
    population: np.ndarray,
    r_star: float,
    best_mask: np.ndarray,
    *,
    ax=None,
    log_scale: bool = False,
    title: str | None = None,
    show_colorbar: bool = True,
    max_show: int | None = 5,
    cmap: str = "viridis",
):
    """
    Plot the population raster with the best circle(s) overlaid.

    Notes
    -----
    Coordinates follow array indexing:
    - x = column index
    - y = row index
    with imshow(origin="upper").
    """
    if not isinstance(population, np.ndarray) or population.ndim != 2:
        raise ValueError("population must be a 2D numpy array")
    if not isinstance(best_mask, np.ndarray) or best_mask.shape != population.shape:
        raise ValueError("best_mask must be a numpy array with the same shape as population")
    if best_mask.dtype != bool:
        best_mask = best_mask.astype(bool, copy=False)
    if r_star < 0:
        raise ValueError("r_star must be >= 0")

    if ax is None:
        _, ax = plt.subplots()

    data = population
    if log_scale:
        data = np.log1p(population)

    im = ax.imshow(data, cmap=cmap, origin="upper")
    if show_colorbar:
        plt.colorbar(im, ax=ax, label="Population" + (" (log1p)" if log_scale else ""))

    ys, xs = np.nonzero(best_mask)
    if max_show is not None and len(ys) > max_show:
        ys, xs = ys[:max_show], xs[:max_show]

    for y, x in zip(ys, xs):
        ax.add_patch(plt.Circle((x, y), r_star, fill=False, linewidth=2))
        ax.plot(x, y, marker="x")  # mark center

    if title is None:
        title = f"VP Circles: r*={r_star:.2f}"
    ax.set_title(title)
    ax.set_xlabel("X (column)")
    ax.set_ylabel("Y (row)")

    return ax
