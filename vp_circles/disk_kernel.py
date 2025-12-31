import numpy as np


def disk_kernel(r: float) -> np.ndarray:
    """
    Create a binary disk kernel based on *cell centers*.

    Imagine a square grid of cells. Take the center of the middle cell as the
    circle center. Given a (possibly non-integer) radius r (in cell units),
    set kernel[dy, dx] = 1 if the center of that cell lies inside/on the circle,
    i.e., dx^2 + dy^2 <= r^2; otherwise set it to 0.

    Parameters
    ----------
    r : float
        Circle radius in grid-cell units. Can be non-integer. Must be >= 0.

    Returns
    -------
    kernel : np.ndarray
        A (2*ceil(r)+1, 2*ceil(r)+1) array of 0/1 floats.
    """
    if r < 0:
        raise ValueError("r must be >= 0")

    R = int(np.ceil(r))  # integer window radius (array size must be integer)

    y, x = np.ogrid[-R:R+1, -R:R+1]  # integer center offsets (dy, dx)
    mask = (x * x + y * y) <= (r * r)  # use real r in the circle test

    return mask.astype(float)
