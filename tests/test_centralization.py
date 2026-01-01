import numpy as np
import pytest

from vp_circles.centralization import centralization


def test_centralization_full_grid_single_center_r0():
    """
    Basic correctness on full grid with r_star=0.

    Expected behavior:
    - Disk area at r=0 contains exactly 1 cell.
    - Area(A) = H*W when region_mask is None.
    - C = 1 - 2*(1/(H*W)).
    """
    H, W = 10, 20
    best_mask = np.zeros((H, W), dtype=bool)
    best_mask[3, 4] = True

    C = centralization(H, W, r_star=0.0, best_mask=best_mask)

    expected = 1.0 - 2.0 * (1.0 / (H * W))
    assert np.isclose(C, expected, atol=1e-12)


def test_centralization_region_mask_changes_area_A():
    """
    Region mask should change Area(A) and thus C.

    Expected behavior:
    - With A being a smaller region, the same disk intersection area leads to a different C.
    """
    H, W = 10, 10
    best_mask = np.zeros((H, W), dtype=bool)
    best_mask[5, 5] = True

    # A is a 4x4 block => area_A = 16, includes the center
    A = np.zeros((H, W), dtype=bool)
    A[3:7, 3:7] = True

    C = centralization(H, W, r_star=0.0, best_mask=best_mask, region_mask=A)

    expected = 1.0 - 2.0 * (1.0 / 16.0)
    assert np.isclose(C, expected, atol=1e-12)


def test_centralization_rejects_empty_best_mask():
    """
    Input validation: best_mask must contain at least one center.

    Expected behavior:
    - Raise ValueError if best_mask has no True entries.
    """
    H, W = 5, 6
    best_mask = np.zeros((H, W), dtype=bool)

    with pytest.raises(ValueError):
        centralization(H, W, r_star=1.0, best_mask=best_mask)
