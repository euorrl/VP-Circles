import numpy as np
import matplotlib.pyplot as plt

from vp_circles.find_vp_circles import find_vp_circles
from vp_circles.plot_vp_circles import plot_vp_circles


def gaussian2d(H, W, cy, cx, sigma_y, sigma_x, amp=1.0):
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    gy = ((y - cy) / sigma_y) ** 2
    gx = ((x - cx) / sigma_x) ** 2
    return amp * np.exp(-0.5 * (gy + gx))


def make_case_maps(H, W):
    maps = {}

    # 1) Single peak (clean unimodal)
    pop1 = gaussian2d(H, W, cy=0.45 * H, cx=0.55 * W, sigma_y=12, sigma_x=18, amp=8000.0)
    maps["Single peak"] = pop1

    # 2) Two separated peaks (bimodal)
    pop2 = (
        gaussian2d(H, W, cy=0.35 * H, cx=0.30 * W, sigma_y=10, sigma_x=10, amp=7000.0)
        + gaussian2d(H, W, cy=0.65 * H, cx=0.75 * W, sigma_y=12, sigma_x=12, amp=6500.0)
    )
    maps["Two peaks"] = pop2

    # 3) Ridge / strip (anisotropic band)
    pop3 = gaussian2d(H, W, cy=0.50 * H, cx=0.50 * W, sigma_y=6, sigma_x=45, amp=8000.0)
    maps["Horizontal ridge"] = pop3

    # 4) Ring (donut-like)
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    cy, cx = 0.50 * H, 0.50 * W
    rr = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    pop4 = 7000.0 * np.exp(-0.5 * ((rr - 35.0) / 6.0) ** 2)  # ring radius ~35
    maps["Ring"] = pop4

    # 5) Four corners (multi-modal edges)
    pop5 = (
        gaussian2d(H, W, cy=0.15 * H, cx=0.15 * W, sigma_y=10, sigma_x=10, amp=5000.0)
        + gaussian2d(H, W, cy=0.15 * H, cx=0.85 * W, sigma_y=10, sigma_x=10, amp=5000.0)
        + gaussian2d(H, W, cy=0.85 * H, cx=0.15 * W, sigma_y=10, sigma_x=10, amp=5000.0)
        + gaussian2d(H, W, cy=0.85 * H, cx=0.85 * W, sigma_y=10, sigma_x=10, amp=5000.0)
    )
    maps["Four corners"] = pop5

    # 6) Gradient + hotspot (background trend + local spike)
    y = np.arange(H)[:, None]
    gradient = (y / (H - 1)) * 2000.0  # increasing from top to bottom
    pop6 = gradient + gaussian2d(H, W, cy=0.30 * H, cx=0.65 * W, sigma_y=8, sigma_x=8, amp=9000.0)
    maps["Gradient + hotspot"] = pop6

    # ensure non-negative float
    for k in maps:
        maps[k] = maps[k].astype(float)

    return maps


# settings
H, W = 140, 200
f = 0.25
candidate_cap = 32

maps = make_case_maps(H, W)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.ravel()

for ax, (name, pop) in zip(axes, maps.items()):
    r_star, best_mask = find_vp_circles(pop, f, candidate_cap=candidate_cap)
    plot_vp_circles(
        pop,
        r_star,
        best_mask,
        ax=ax,
        log_scale=False,
        show_colorbar=False,
        max_show=2,
        title=f"{name}\n(f={f}, r*={r_star:.2f}, #best={int(best_mask.sum())})",
    )

plt.tight_layout()
plt.show()
