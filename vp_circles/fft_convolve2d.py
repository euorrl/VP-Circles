import numpy as np


def fft_convolve2d(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    2D convolution via FFT with zero-padding, returning output of the same shape as `a`.

    This computes the standard discrete convolution:
        out[i, j] = sum_{u,v} a[i-u, j-v] * k[u, v]
    with `a` treated as zero outside its boundaries.

    Parameters
    ----------
    a : np.ndarray, shape (H, W)
        Input 2D array (e.g., population raster).
    k : np.ndarray, shape (Kh, Kw)
        2D kernel (e.g., disk kernel).

    Returns
    -------
    out : np.ndarray, shape (H, W)
        Convolution result cropped to the same shape as `a`.
    """
    a = np.asarray(a, dtype=float)
    k = np.asarray(k, dtype=float)

    if a.ndim != 2 or k.ndim != 2:
        raise ValueError("a and k must both be 2D arrays")

    H, W = a.shape
    Kh, Kw = k.shape

    # Full convolution size
    out_h = H + Kh - 1
    out_w = W + Kw - 1

    # FFT-based convolution (zero-padded)
    Fa = np.fft.rfftn(a, s=(out_h, out_w), axes=(0, 1))
    Fk = np.fft.rfftn(k, s=(out_h, out_w), axes=(0, 1))
    full = np.fft.irfftn(Fa * Fk, s=(out_h, out_w), axes=(0, 1))

    # Crop the central part to get "same" output
    start_i = (Kh - 1) // 2
    start_j = (Kw - 1) // 2

    return full[start_i:start_i + H, start_j:start_j + W]
