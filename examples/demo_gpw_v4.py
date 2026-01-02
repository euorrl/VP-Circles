"""
Demo: Compute a VP circle on a real GPW raster (ASCII grid) with f = 0.5.

Run from the project root:
    python examples/demo_gpw_v4.py

Input file expected in the project root (same folder where you run the command):
    gpw_v4_population_count_rev11_2020_1_deg.asc
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from vp_circles import find_vp_circles, centralization, plot_population, plot_vp_circles


def load_gpw_ascii(path: str) -> np.ndarray:
    """
    Load a GPW-style ESRI ASCII grid exported as .asc.

    Notes
    -----
    - We skip the 6-line header (ncols, nrows, xllcorner, yllcorner, cellsize, NODATA_value).
    - Many GPW rasters use a negative NODATA value (e.g., -9999).
      We treat any negative value as missing and set it to 0 for this demo.
    """
    pop = np.loadtxt(path, skiprows=6, dtype=float)
    pop = np.nan_to_num(pop, nan=0.0, posinf=0.0, neginf=0.0)
    pop[pop < 0] = 0.0
    return pop


def main() -> None:
    path = "examples/gpw_v4_population_count_rev11_2020_1_deg.asc"
    pop = load_gpw_ascii(path)
    H, W = pop.shape

    total = float(pop.sum())
    f = 0.5
    target = f * total

    print("=== VP-Circles demo (GPW raster) ===")
    print(f"Raster shape: H={H}, W={W} (N={H*W})")
    print(f"Total population: {total:.6e}")
    print(f"Target fraction f={f} -> target population: {target:.6e}")
    print()

    r_star, best_mask = find_vp_circles(pop, f)
    C = centralization(H, W, r_star, best_mask, region_mask=None, reduce="min")
    print(f"Optimal radius r*: {r_star:.6f}")
    print(f"Number of optimal centers: {best_mask.sum()}")
    print(f"Centralization C: {C:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    plot_population(pop, ax=axes[0], log_scale=True)
    plot_vp_circles(pop, r_star, best_mask, ax=axes[1], log_scale=True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
