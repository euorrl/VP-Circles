import numpy as np
import pytest

from vp_circles.disk_kernel import disk_kernel


def test_disk_kernel_shape_r0():
    """
    Test the degenerate case r = 0.

    Expected behavior:
    - The kernel should have shape (1, 1) because ceil(0)=0 -> (2*0+1, 2*0+1).
    - The single center cell must be inside the disk, so the value should be 1.
    """
    k = disk_kernel(0.0)
    assert k.shape == (1, 1)
    assert k[0, 0] == 1.0


def test_disk_kernel_negative_r_raises():
    """
    Test input validation.

    Expected behavior:
    - A negative radius is invalid, so the function must raise ValueError.
    """
    with pytest.raises(ValueError):
        disk_kernel(-0.1)


def test_disk_kernel_r1_exact_pattern():
    """
    Test the exact discrete disk pattern for r = 1.0 (cell-center rule).

    For r = 1:
    - The center (0,0) is included.
    - The four direct neighbors (up/down/left/right) at distance 1 are included.
    - The four corners at distance sqrt(2) > 1 are NOT included.

    So the expected kernel is a 'plus' shape (cross) in a 3x3 array.
    """
    k = disk_kernel(1.0)
    expected = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=float)

    assert k.shape == (3, 3)
    assert np.array_equal(k, expected)


def test_disk_kernel_threshold_sqrt2():
    """
    Test the critical threshold around sqrt(2) ≈ 1.414 for corner inclusion.

    The corner cells in the 5x5 kernel are at distance sqrt(2) from the center.
    Therefore:
    - If r < sqrt(2), corners must be excluded -> corner value 0.
    - If r > sqrt(2), corners must be included -> corner value 1.

    We use r=1.4 (< sqrt(2)) and r=1.5 (> sqrt(2)) to check this transition.
    """
    k_small = disk_kernel(1.4)   # corners should still be outside
    k_large = disk_kernel(1.5)   # corners should become inside

    # [0,0] is the top-left corner of the kernel
    assert k_small[1, 1] == 0.0
    assert k_large[1, 1] == 1.0


def test_disk_kernel_symmetry():
    """
    Test geometric symmetry.

    A disk centered at the middle cell should be symmetric:
    - symmetric top/bottom (vertical flip)
    - symmetric left/right (horizontal flip)

    This test ensures the kernel is not shifted or cropped incorrectly.
    """
    k = disk_kernel(2.3)
    assert np.array_equal(k, np.flipud(k))   # up-down symmetry
    assert np.array_equal(k, np.fliplr(k))   # left-right symmetry
