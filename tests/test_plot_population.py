import numpy as np
import pytest

from vp_circles.plot_population import plot_population


def test_plot_population_accepts_2d_array():
    """
    A valid input should work:
    - pop must be a 2D array (H, W)
    - the function should return a Matplotlib Axes object so users can overlay more plots later
    """
    pop = np.ones((10, 12), dtype=float)

    # Disable colorbar to keep the test lightweight and avoid extra figure elements
    ax = plot_population(pop, log_scale=False, show_colorbar=False)

    # A simple, robust check: Matplotlib Axes objects have methods like `set_title`
    assert hasattr(ax, "set_title")


def test_plot_population_rejects_non_2d():
    """
    Invalid input should be rejected:
    - if pop is not 2D (e.g., 1D), the function should raise ValueError
    """
    pop = np.ones(10, dtype=float)  # 1D array, invalid

    with pytest.raises(ValueError):
        plot_population(pop, show_colorbar=False)


def test_plot_population_rejects_negative_values():
    """
    Population values should be non-negative:
    - if pop contains negative values, the function should raise ValueError
    """
    pop = np.array([[1.0, -1.0], [2.0, 3.0]])  # contains a negative value

    with pytest.raises(ValueError):
        plot_population(pop, show_colorbar=False)
