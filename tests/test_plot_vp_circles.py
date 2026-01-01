import matplotlib
matplotlib.use("Agg")  # headless backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vp_circles.plot_vp_circles import plot_vp_circles


def test_plot_vp_circles_creates_axes_when_none():
    """
    Basic smoke test when ax is None.

    Expected behavior:
    - The function returns a matplotlib Axes.
    - It draws an image on the axes.
    - It overlays at least one circle patch and one center marker when best_mask has True.
    """
    pop = np.zeros((20, 30), dtype=float)
    pop[10, 15] = 100.0

    best_mask = np.zeros_like(pop, dtype=bool)
    best_mask[10, 15] = True

    r_star = 5.0
    ax = plot_vp_circles(pop, r_star, best_mask, log_scale=False, show_colorbar=False)

    assert hasattr(ax, "imshow")
    # One image should be added
    assert len(ax.images) == 1
    # One circle patch should be added
    assert len(ax.patches) >= 1
    # One marker line (ax.plot) should be added
    assert len(ax.lines) >= 1

    plt.close(ax.figure)


def test_plot_vp_circles_uses_given_axes():
    """
    When an Axes is provided, plot should be drawn on that Axes.

    Expected behavior:
    - The returned Axes is exactly the one passed in.
    - No new figure is required for correctness.
    """
    pop = np.ones((10, 10), dtype=float)
    best_mask = np.zeros_like(pop, dtype=bool)
    best_mask[5, 5] = True

    fig, ax = plt.subplots()
    out_ax = plot_vp_circles(pop, 2.0, best_mask, ax=ax, show_colorbar=False)

    assert out_ax is ax
    assert len(ax.images) == 1
    assert len(ax.patches) >= 1
    assert len(ax.lines) >= 1

    plt.close(fig)


def test_plot_vp_circles_max_show_limits_overlays():
    """
    max_show should limit the number of circles/markers drawn.

    Expected behavior:
    - If best_mask has many True values and max_show is small,
      the number of circle patches added equals max_show.
    """
    pop = np.ones((30, 30), dtype=float)

    best_mask = np.zeros_like(pop, dtype=bool)
    # 10 best centers
    coords = [(5, 5), (5, 10), (5, 15), (10, 5), (10, 10),
              (10, 15), (15, 5), (15, 10), (15, 15), (20, 20)]
    for y, x in coords:
        best_mask[y, x] = True

    fig, ax = plt.subplots()
    max_show = 3
    plot_vp_circles(pop, 3.0, best_mask, ax=ax, max_show=max_show, show_colorbar=False)

    # Each shown center adds exactly one Circle patch
    assert len(ax.patches) == max_show
    # Each shown center adds exactly one marker line
    assert len(ax.lines) == max_show

    plt.close(fig)


def test_plot_vp_circles_log_scale_runs():
    """
    log_scale should be accepted and produce a valid plot.

    Expected behavior:
    - The function runs without error with log_scale=True.
    - It still produces one image on the axes.
    """
    pop = np.zeros((15, 15), dtype=float)
    pop[7, 7] = 1000.0

    best_mask = np.zeros_like(pop, dtype=bool)
    best_mask[7, 7] = True

    fig, ax = plt.subplots()
    plot_vp_circles(pop, 1.0, best_mask, ax=ax, log_scale=True, show_colorbar=False)

    assert len(ax.images) == 1

    plt.close(fig)


def test_plot_vp_circles_rejects_non_2d_population():
    """
    Input validation: population must be 2D.

    Expected behavior:
    - Raise ValueError for non-2D population.
    """
    pop = np.ones(10, dtype=float)
    best_mask = np.ones(10, dtype=bool)

    with pytest.raises(ValueError):
        plot_vp_circles(pop, 1.0, best_mask, show_colorbar=False)


def test_plot_vp_circles_rejects_mismatched_mask_shape():
    """
    Input validation: best_mask shape must match population.

    Expected behavior:
    - Raise ValueError if best_mask.shape != population.shape.
    """
    pop = np.ones((10, 10), dtype=float)
    best_mask = np.ones((9, 10), dtype=bool)

    with pytest.raises(ValueError):
        plot_vp_circles(pop, 1.0, best_mask, show_colorbar=False)


def test_plot_vp_circles_rejects_negative_radius():
    """
    Input validation: r_star must be non-negative.

    Expected behavior:
    - Raise ValueError if r_star < 0.
    """
    pop = np.ones((10, 10), dtype=float)
    best_mask = np.zeros_like(pop, dtype=bool)
    best_mask[5, 5] = True

    with pytest.raises(ValueError):
        plot_vp_circles(pop, -1.0, best_mask, show_colorbar=False)


def test_plot_vp_circles_casts_mask_to_bool():
    """
    best_mask dtype handling.

    Expected behavior:
    - Non-bool best_mask should be accepted and cast to bool internally.
    - Plot should include overlays for non-zero entries.
    """
    pop = np.ones((10, 10), dtype=float)
    # int mask (0/1)
    best_mask = np.zeros_like(pop, dtype=int)
    best_mask[3, 4] = 1

    fig, ax = plt.subplots()
    plot_vp_circles(pop, 2.0, best_mask, ax=ax, show_colorbar=False)

    assert len(ax.patches) == 1
    assert len(ax.lines) == 1

    plt.close(fig)
