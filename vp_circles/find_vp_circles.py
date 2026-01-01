from __future__ import annotations

import math
from typing import Tuple, Optional
import warnings
import numpy as np

from .disk_kernel import disk_kernel
from .fft_convolve2d import fft_convolve2d
from .find_sum_of_squares import find_sum_of_squares


def find_vp_circles(
    population: np.ndarray,
    f: float,
    *,
    candidate_cap: int = 32,
) -> Tuple[float, np.ndarray]:
    """
    Three-stage VP circle search (Stage I halving -> Stage II binary+pruning -> Stage III exact),
    using a fixed candidate switch threshold (default: 32). 

    Parameters
    ----------
    population : np.ndarray, shape (H, W)
        2D population raster (non-negative).
    f : float
        Target population fraction in (0, 1).
    candidate_cap : int
        Enter Stage III when number of feasible centers <= candidate_cap.

    Returns
    -------
    r_star : float
        Minimal radius (exact once Stage III is implemented).
    best_mask : np.ndarray, shape (H, W), dtype=bool
        Mask of centers achieving r_star (may contain multiple centers).

    Notes
    -----
    - Stage III is currently a placeholder and must be implemented using the exact method
      from the referenced paper, applied to <= candidate_cap centers.
    - Stage I initial radius is half of the grid diagonal length (in cell units).
    """

    # -------------------------
    # 0) Input checks
    # -------------------------
    if not isinstance(population, np.ndarray):
        raise TypeError("population must be a numpy array")
    if population.ndim != 2:
        raise ValueError(f"population must be 2D, got shape {population.shape}")
    if np.any(population < 0):
        raise ValueError("population must be non-negative")
    if not (0.0 < float(f) < 1.0):
        raise ValueError("f must be in (0, 1)")
    if candidate_cap <= 0:
        raise ValueError("candidate_cap must be positive")

    H, W = population.shape
    total_pop = float(population.sum())
    if total_pop <= 0:
        # Degenerate case: no population anywhere -> no circle can reach any positive fraction
        raise ValueError("total population is 0; VP circle is undefined")

    threshold = f * total_pop


    # -------------------------
    # Core: feasibility checker
    # returns same-size mask
    # -------------------------
    def _feasible_mask(r: float) -> Tuple[np.ndarray, int]:
        """
        Feasibility check at radius r.
        Returns a (H, W) boolean mask and number of feasible centers.
        """
        if r < 0:
            raise ValueError("radius r must be >= 0")

        k = disk_kernel(float(r))
        sums = fft_convolve2d(population, k)  # must be (H, W)

        if sums.shape != population.shape:
            raise RuntimeError(
                f"fft_convolve2d returned shape {sums.shape}, expected {population.shape}"
            )

        mask = sums >= threshold
        # bool sum -> int count
        n = int(mask.sum())
        return mask, n


    # -------------------------
    # Stage I: exponential halving
    # initial radius = half diagonal
    # -------------------------
    r = 0.5 * math.sqrt(H * H + W * W)

    best_feasible_r: Optional[float] = None
    best_mask: Optional[np.ndarray] = None
    best_n: int = 0

    # Ensure we start from a feasible radius. With half-diagonal it should usually be feasible,
    mask_r, n_r = _feasible_mask(r)

    # Now do halving until we bracket feasible->infeasible or hit candidate_cap
    while True:
        # Early jump to Stage III
        if n_r <= candidate_cap:
            # Candidate set already small enough; go exact
            candidate_mask = mask_r
            bracket_low = None
            bracket_high = r
            break

        # Record this feasible state and shrink r
        best_feasible_r = r
        best_mask = mask_r
        best_n = n_r

        r_next = r / 2.0
        mask_next, n_next = _feasible_mask(r_next)

        if n_next == 0:
            # Bracket found: (r_next infeasible, r feasible)
            # Move to Stage II
            bracket_low = r_next
            bracket_high = r
            candidate_mask = best_mask  # last feasible mask
            break

        # continue halving
        r = r_next
        mask_r, n_r = mask_next, n_next

    # If we jumped early and candidate_mask comes from current mask_r
    if "candidate_mask" not in locals():
        # Safety (should never happen)
        raise RuntimeError("internal error: candidate_mask not set")


    # -------------------------
    # Stage II: binary search + pruning
    # termination: |C| <= cap OR interval < 1
    # -------------------------
    # If we did not bracket (i.e., Stage I jumped directly), we can skip Stage II.
    if bracket_low is not None and bracket_high is not None:
        r_low = float(bracket_low)
        r_high = float(bracket_high)

        # candidate_mask must always be "last feasible" mask (same size)
        # and we keep updating it when mid is feasible.
        while True:
            # Stop by candidate count
            n_cand = int(candidate_mask.sum())
            if n_cand <= candidate_cap:
                # Move to Stage III
                break

            # Stop by coarse interval width (< 1 cell)
            if (r_high - r_low) < 1.0:
                # Move to Stage III
                break

            r_mid = 0.5 * (r_low + r_high)
            mid_mask, mid_n = _feasible_mask(r_mid)

            if mid_n > 0:
                # feasible -> tighten upper bound and update candidates
                r_high = r_mid
                candidate_mask = mid_mask
            else:
                # infeasible -> raise lower bound; keep last feasible candidates
                r_low = r_mid

    # -------------------------
    # Stage III: exact search over candidate centers
    # exact discrete radii based on d^2 = a^2 + b^2
    # -------------------------

    # Determine (r_low, r_high) bounds for Stage III
    # - If Stage I jumped directly: bracket_low is None -> use r_low = 0
    # - If Stage II ran: we have r_low and r_high (still in scope)
    if bracket_low is None:
        r_low = 0.0
        r_high = float(bracket_high)  # type: ignore[arg-type]
    else:
        # r_low, r_high already set in Stage II
        r_low = float(r_low)
        r_high = float(r_high)

    # Upper bound on squared radius to consider (discrete boundary)
    s_high = int(math.floor(r_high * r_high))
    if s_high < 0:
        raise RuntimeError("internal error: s_high < 0")

    # Candidate centers to evaluate (y, x)
    cand_ys, cand_xs = np.nonzero(candidate_mask)
    num_cand = int(len(cand_ys))
    if num_cand == 0:
        # Should not happen if r_high is feasible, but keep safe.
        raise RuntimeError("internal error: no candidates for Stage III")

    if num_cand > candidate_cap:
        warnings.warn(
            f"Stage III running with {num_cand} candidates (> candidate_cap={candidate_cap}). "
            "This may be slow. Consider tightening Stage II or lowering candidate_cap.",
            RuntimeWarning,
        )

    # Discrete critical squared radii levels (sorted ascending)
    # We include 0 so the prefix sum is well-defined.
    levels = find_sum_of_squares(0, s_high)
    if len(levels) == 0 or levels[0] != 0:
        # Ensure 0 is present (should be, since 0 = 0^2 + 0^2)
        levels = [0] + levels

    levels_arr = np.array(levels, dtype=np.int32)

    # Build a mapping from d2 -> index in levels (for fast bucketing)
    idx_map = np.full(s_high + 1, -1, dtype=np.int32)
    idx_map[levels_arr] = np.arange(len(levels_arr), dtype=np.int32)

    # Precompute disk offsets up to s_high (cache by s_high)
    # Offsets include all (dy, dx) with dy^2 + dx^2 <= s_high
    _offset_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _get_offsets(s_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if s_max in _offset_cache:
            return _offset_cache[s_max]
        R = int(math.isqrt(s_max))
        grid = np.arange(-R, R + 1, dtype=np.int32)
        dy, dx = np.meshgrid(grid, grid, indexing="ij")
        dy = dy.ravel()
        dx = dx.ravel()
        d2 = dy * dy + dx * dx
        keep = d2 <= s_max
        dy, dx, d2 = dy[keep], dx[keep], d2[keep]
        _offset_cache[s_max] = (dy, dx, d2)
        return dy, dx, d2

    dy, dx, d2 = _get_offsets(s_high)

    # Evaluate each candidate center: compute exact minimal feasible radius
    r_candidates = np.full(num_cand, np.inf, dtype=np.float64)

    for i in range(num_cand):
        cy = int(cand_ys[i])
        cx = int(cand_xs[i])

        ys = cy + dy
        xs = cx + dx

        # clip to grid bounds
        valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        if not np.any(valid):
            continue

        ys_v = ys[valid]
        xs_v = xs[valid]
        d2_v = d2[valid]

        # Bucket population by d2 level index
        weights = np.zeros(len(levels_arr), dtype=np.float64)
        level_idx = idx_map[d2_v]
        # idx_map should always be valid for d2_v because d2_v <= s_high and
        # any integer d2 might not be a sum-of-two-squares; those would map to -1.
        # We must ignore them.
        ok = level_idx >= 0
        if np.any(ok):
            np.add.at(weights, level_idx[ok], population[ys_v[ok], xs_v[ok]].astype(np.float64, copy=False))

        # Prefix sum to find minimal d2 where cumulative >= threshold
        cum = np.cumsum(weights)
        j = int(np.searchsorted(cum, threshold, side="left"))
        if j < len(levels_arr):
            s_star = int(levels_arr[j])
            r_candidates[i] = math.sqrt(s_star)
        else:
            # Should not happen since r_high was feasible for these candidates,
            # but keep safe: assign r_high as fallback.
            r_candidates[i] = r_high

    # Global optimum
    r_star = float(np.min(r_candidates))
    # If something went wrong and all are inf
    if not np.isfinite(r_star):
        raise RuntimeError("internal error: failed to compute any finite r_star in Stage III")

    best_mask = np.zeros((H, W), dtype=bool)
    best_idx = np.isclose(r_candidates, r_star, rtol=0.0, atol=1e-12)
    best_mask[cand_ys[best_idx], cand_xs[best_idx]] = True

    return r_star, best_mask
