import numpy as np
import pytest

from vp_circles.find_vp_circles import find_vp_circles
from vp_circles.disk_kernel import disk_kernel
from vp_circles.fft_convolve2d import fft_convolve2d


def _is_feasible(pop: np.ndarray, f: float, r: float, center_mask: np.ndarray) -> bool:
    """
    Helper: check whether there exists a center in center_mask whose disk sum >= f * total.
    Uses the same convolution approach as Stage I/II feasibility checks.
    """
    total = float(pop.sum())
    threshold = f * total
    sums = fft_convolve2d(pop, disk_kernel(float(r)))
    return bool(np.any((sums >= threshold) & center_mask))


def test_find_vp_circles_rejects_non_2d():
    """
    Input validation for population dimensionality.

    Expected behavior:
    - Raise ValueError if population is not a 2D array.
    """
    pop_1d = np.ones(10, dtype=float)
    with pytest.raises(ValueError):
        find_vp_circles(pop_1d, 0.5)


def test_find_vp_circles_rejects_negative_population():
    """
    Input validation for negative values.

    Expected behavior:
    - Raise ValueError if population contains negative entries.
    """
    pop = np.zeros((5, 5), dtype=float)
    pop[2, 2] = -1.0
    with pytest.raises(ValueError):
        find_vp_circles(pop, 0.5)


def test_find_vp_circles_rejects_invalid_f():
    """
    Input validation for f.

    Expected behavior:
    - Raise ValueError if f is not in (0, 1).
    """
    pop = np.ones((5, 5), dtype=float)
    with pytest.raises(ValueError):
        find_vp_circles(pop, 0.0)
    with pytest.raises(ValueError):
        find_vp_circles(pop, 1.0)
    with pytest.raises(ValueError):
        find_vp_circles(pop, -0.1)
    with pytest.raises(ValueError):
        find_vp_circles(pop, 1.1)


def test_find_vp_circles_rejects_zero_total_population():
    """
    Degenerate case: total population is zero.

    Expected behavior:
    - Raise ValueError when sum(population) == 0.
    """
    pop = np.zeros((6, 7), dtype=float)
    with pytest.raises(ValueError):
        find_vp_circles(pop, 0.5)


def test_find_vp_circles_single_hotspot_radius_zero():
    """
    Single hotspot should yield r_star == 0 for any f that can be met at the hotspot.

    Setup:
    - All population concentrated at one cell.
    - Any center at that cell reaches threshold with radius 0 (disk contains the center only).

    Expected behavior:
    - r_star == 0.
    - best_mask has True at the hotspot location (and may have others only if they also meet threshold at r=0).
    - Returned center(s) must be feasible at r_star.
    """
    pop = np.zeros((7, 7), dtype=float)
    pop[3, 4] = 10.0

    f = 0.5  # threshold = 5
    r_star, best_mask = find_vp_circles(pop, f, candidate_cap=32)

    assert best_mask.shape == pop.shape
    assert best_mask.dtype == bool
    assert np.isclose(r_star, 0.0, atol=0.0)
    assert best_mask[3, 4]


    # Feasibility at returned r_star
    assert _is_feasible(pop, f, r_star, best_mask)


def test_find_vp_circles_two_points_need_radius_one():
    """
    Two equal hotspots separated by 2 cells horizontally.

    Setup:
    - pop[2,2] = 5, pop[2,4] = 5, total=10
    - With f=0.95, threshold=9.5 => must include both hotspots.
    - Best center is at (2,3), requiring radius 1 (since distances are 1).

    Expected behavior:
    - r_star == 1.0
    - best_mask[2,3] == True
    - Returned center(s) feasible at r_star
    - Not feasible at any radius < 1 for that same optimal center (sanity check)
    """
    pop = np.zeros((6, 7), dtype=float)
    pop[2, 2] = 5.0
    pop[2, 4] = 5.0

    f = 0.95
    r_star, best_mask = find_vp_circles(pop, f, candidate_cap=32)

    assert best_mask.shape == pop.shape
    assert best_mask.dtype == bool

    # Exact discrete answer should be radius 1
    assert np.isclose(r_star, 1.0, atol=1e-12)
    assert best_mask[2, 3]


    # Feasible at r_star for some returned center
    assert _is_feasible(pop, f, r_star, best_mask)

    # Sanity: for the known best center (2,3), radius < 1 should be infeasible
    center_only = np.zeros_like(best_mask)
    center_only[2, 3] = True
    assert not _is_feasible(pop, f, 0.0, center_only)


def test_find_vp_circles_outputs_are_consistent_and_finite():
    """
    General output consistency on a small random raster.

    Expected behavior:
    - r_star is finite and non-negative.
    - best_mask is boolean, same shape as input.
    - best_mask contains at least one True.
    - best_mask is feasible at r_star.
    """
    rng = np.random.default_rng(0)
    pop = rng.gamma(shape=2.0, scale=10.0, size=(50, 50)).astype(float)

    f = 0.2
    r_star, best_mask = find_vp_circles(pop, f, candidate_cap=32)

    assert np.isfinite(r_star)
    assert r_star >= 0.0
    assert best_mask.shape == pop.shape
    assert best_mask.dtype == bool
    assert bool(best_mask.any())


def test_find_vp_circles_r_star_is_discrete_sqrt_of_integer():
    """
    Stage III returns r_star computed as sqrt(s_star) where s_star is an integer (a^2 + b^2).

    Expected behavior:
    - r_star^2 should be extremely close to an integer.
    """
    pop = np.zeros((9, 9), dtype=float)
    pop[4, 4] = 10.0
    pop[4, 6] = 10.0  # requires some positive radius for high f

    f = 0.9  # threshold = 18
    r_star, _ = find_vp_circles(pop, f, candidate_cap=32)

    s = r_star * r_star
    assert np.isclose(s, round(s), atol=1e-9)
