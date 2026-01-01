import numpy as np
import matplotlib.pyplot as plt

from vp_circles.find_vp_circles import find_vp_circles
from vp_circles.plot_population import plot_population


def make_population_with_points(H, W, n_points, *, seed=0, background=0.0, point_mass=1000.0):
    """
    Create a population raster with n_points isolated hotspots (single cells).
    """
    rng = np.random.default_rng(seed)
    pop = np.full((H, W), background, dtype=float)

    # pick unique coordinates away from borders a bit (optional, to keep circles visible)
    margin = 5
    ys = rng.integers(margin, H - margin, size=n_points)
    xs = rng.integers(margin, W - margin, size=n_points)

    # ensure uniqueness
    coords = set()
    for y, x in zip(ys, xs):
        coords.add((int(y), int(x)))
    while len(coords) < n_points:
        coords.add((int(rng.integers(margin, H - margin)), int(rng.integers(margin, W - margin))))

    for (y, x) in coords:
        pop[y, x] += point_mass

    return pop, sorted(coords)


def overlay_best(ax, best_mask, r_star, *, max_show=3):
    ys, xs = np.nonzero(best_mask)
    for i in range(min(len(ys), max_show)):
        cy, cx = int(ys[i]), int(xs[i])
        ax.plot(cx, cy, marker="x")  # best center
        ax.add_patch(plt.Circle((cx, cy), r_star, fill=False))


# demo settings
H, W = 120, 160
f = 0.6            # target fraction (tune if you want larger/smaller circles)
candidate_cap = 32
point_mass = 1000.0

cases = [1, 2, 5, 10]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.ravel()

for ax, n_points in zip(axes, cases):
    pop, coords = make_population_with_points(
        H, W, n_points,
        seed=42 + n_points,
        background=0.0,
        point_mass=point_mass,
    )

    r_star, best_mask = find_vp_circles(pop, f, candidate_cap=candidate_cap)

    # plot population
    plot_population(pop, log_scale=False, show_colorbar=False,
                    title=f"{n_points} point(s) | f={f} | r*={r_star:.2f}",
                    ax=ax)

    # mark true hotspot locations (optional)
    for (y, x) in coords:
        ax.plot(x, y, marker="o")  # hotspot marker

    # overlay best centers and circles
    overlay_best(ax, best_mask, r_star, max_show=2)

plt.tight_layout()
plt.show()
