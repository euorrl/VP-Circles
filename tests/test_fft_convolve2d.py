import numpy as np
from vp_circles.fft_convolve2d import fft_convolve2d


def test_fft_convolve_identity_1x1():
    """
    1x1 kernel should act as identity.

    Expected behavior:
    - The output must match the input exactly.
    """
    a = np.random.rand(6, 7)
    k = np.array([[1.0]])

    out = fft_convolve2d(a, k)

    assert out.shape == a.shape
    assert np.allclose(out, a)


def test_fft_convolve_delta_kernel():
    """
    Centered delta kernel should preserve input (same output).
    
    Expected behavior:
    - The output must match the input exactly.
    """
    a = np.random.rand(5, 5)
    k = np.zeros((3, 3))
    k[1, 1] = 1.0

    out = fft_convolve2d(a, k)

    assert np.allclose(out, a, atol=1e-9)
