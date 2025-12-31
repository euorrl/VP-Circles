import numpy as np
import matplotlib.pyplot as plt

from vp_circles.disk_kernel import disk_kernel
from vp_circles.fft_convolve2d import fft_convolve2d

# 1) input: all ones
H, W = 9, 9
a = np.ones((H, W), dtype=float)

# 2) kernel: disk kernel with radius r=4.0
r = 4.0
k = disk_kernel(r)

# 3) convolution
out = fft_convolve2d(a, k)

# 4) visualize
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(a, cmap="gray", interpolation="nearest")
axes[0].set_title("Input a (30x30 all ones)")
axes[0].axis("off")

axes[1].imshow(k, cmap="gray", interpolation="nearest")
axes[1].set_title(f"Kernel k(r={r})")
axes[1].axis("off")

axes[2].imshow(out, cmap="gray", interpolation="nearest")
axes[2].set_title("Output = fft_convolve2d(a, k)")
axes[2].axis("off")

plt.tight_layout()
plt.show()
