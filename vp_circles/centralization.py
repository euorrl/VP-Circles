from __future__ import annotations

import math
from typing import Literal, Optional

import numpy as np


def centralization(
    H: int,
    W: int,
    r_star: float,
    best_mask: np.ndarray,
    *,
    region_mask: Optional[np.ndarray] = None,
    reduce: Literal["min", "max", "mean"] = "min",
) -> float:
    """
    Compute centralization statistic:

        C = 1 - 2 * Area(VP_A ∩ A) / Area(A)

    Here:
    - A is the region defined by region_mask (or full grid if None).
    - VP_A is the VP-circle disk(s) of radius r_star centered at the best centers (best_mask).

    Parameters
    ----------
    H, W : int
        Grid shape (height, width).
    r_star : float
        VP-circle optimal radius.
    best_mask : np.ndarray, shape (H, W)
        Boolean mask of best center locations.
    region_mask : np.ndarray, optional, shape (H, W)
        Boolean mask defining region A. If None, A is the full grid.
    reduce : {"min", "max", "mean"}
        How to aggregate Area(VP_A ∩ A) if multiple best centers exist.

    Returns
    -------
    C : float
        Centralization statistic.
    """
    if not isinstance(H, int) or not isinstance(W, int) or H <= 0 or W <= 0:
        raise ValueError("H and W must be positive integers")
    if r_star < 0:
        raise ValueError("r_star must be >= 0")
    if not isinstance(best_mask, np.ndarray) or best_mask.shape != (H, W):
        raise ValueError("best_mask must be a numpy array of shape (H, W)")
    if reduce not in ("min", "max", "mean"):
        raise ValueError("reduce must be one of {'min','max','mean'}")

    # Define region A
    if region_mask is None:
        A = np.ones((H, W), dtype=bool)
    else:
        if not isinstance(region_mask, np.ndarray) or region_mask.shape != (H, W):
            raise ValueError("region_mask must be a numpy array with the same shape (H, W)")
        A = region_mask.astype(bool, copy=False)

    area_A = int(A.sum())
    if area_A == 0:
        raise ValueError("region_mask (A) has zero area")

    # Ensure bool mask
    if best_mask.dtype != bool:
        best_mask = best_mask.astype(bool, copy=False)

    ys, xs = np.nonzero(best_mask)
    if len(ys) == 0:
        raise ValueError("best_mask contains no centers (no True entries)")

    # Discrete disk definition uses s_star = floor(r_star^2)
    s_star = int(math.floor(r_star * r_star))
    R = int(math.isqrt(s_star))

    # Precompute offsets in disk: dy^2 + dx^2 <= s_star
    grid = np.arange(-R, R + 1, dtype=np.int32)
    dy, dx = np.meshgrid(grid, grid, indexing="ij")
    dy = dy.ravel()
    dx = dx.ravel()
    d2 = dy * dy + dx * dx
    keep = d2 <= s_star
    dy, dx = dy[keep], dx[keep]

    # Compute intersection area for each best center
    inter_areas = []
    for cy, cx in zip(ys, xs):
        yy = cy + dy
        xx = cx + dx
        valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        yy = yy[valid]
        xx = xx[valid]
        inter_areas.append(int(A[yy, xx].sum()))

    if reduce == "min":
        area_inter = float(min(inter_areas))
    elif reduce == "max":
        area_inter = float(max(inter_areas))
    else:  # mean
        area_inter = float(np.mean(inter_areas))

    ratio = area_inter / float(area_A)
    C = 1.0 - 2.0 * ratio
    return C
