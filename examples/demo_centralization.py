import numpy as np
import matplotlib.pyplot as plt

from vp_circles.find_vp_circles import find_vp_circles
from vp_circles.centralization import centralization
from vp_circles.plot_vp_circles import plot_vp_circles


# create a representative population heatmap (two peaks)
H, W = 160, 220
y = np.arange(H)[:, None]
x = np.arange(W)[None, :]

pop = (
    8000.0 * np.exp(-0.5 * (((y - 0.35 * H) / 12.0) ** 2 + ((x - 0.30 * W) / 12.0) ** 2))
    + 6500.0 * np.exp(-0.5 * (((y - 0.65 * H) / 14.0) ** 2 + ((x - 0.75 * W) / 14.0) ** 2))
)

# define a region A (a central rectangle)
A = np.zeros((H, W), dtype=bool)
A[int(0.2 * H) : int(0.8 * H), int(0.2 * W) : int(0.8 * W)] = True

# run VP-Circle on population restricted to A
f = 0.5
pop_A = np.where(A, pop, 0.0)
r_star, best_mask = find_vp_circles(pop_A, f, candidate_cap=32)

# compute centralization C using only (H, W, r_star, best_mask)
C = centralization(H, W, r_star, best_mask, region_mask=A, reduce="min")

print(f"r_star = {r_star:.6f}")
print(f"#best centers = {int(best_mask.sum())}")
print(f"Centralization C = {C:.6f}")

# plot population + VP circles + region boundary
ax = plot_vp_circles(
    pop,
    r_star,
    best_mask,
    log_scale=False,
    title=f"f={f}, r*={r_star:.2f}, best centers={int(best_mask.sum())}, C={C:.3f}",
    show_colorbar=True,
    max_show=2,
)

# draw the region A border (rectangle)
y0, y1 = int(0.2 * H), int(0.8 * H)
x0, x1 = int(0.2 * W), int(0.8 * W)
ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])

plt.show()
