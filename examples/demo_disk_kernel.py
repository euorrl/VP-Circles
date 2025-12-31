import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from vp_circles.disk_kernel import disk_kernel

# radii: small + large + threshold around sqrt(2)
radii = [0.0, 0.5, 1.0, 1.4, 1.5, 2.0, 3.0]

fig, axes = plt.subplots(1, len(radii), figsize=(3 * len(radii), 3))

for ax, r in zip(axes, radii):
    k = disk_kernel(r)
    H, W = k.shape

    # show kernel as black/white blocks (no smoothing)
    ax.imshow(k, cmap="gray", origin="upper", interpolation="nearest")

    # draw grid lines to show cell structure
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", linewidth=0.6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # circle center is the middle cell center
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

    # draw the continuous circle and its center
    ax.add_patch(Circle((cx, cy), r, fill=False, linewidth=2))
    ax.scatter([cx], [cy], marker="x", s=60)

    ax.set_title(f"r={r}")

plt.suptitle("Disk kernel (cell centers inside dx^2+dy^2<=r^2)")
plt.tight_layout()
plt.show()
