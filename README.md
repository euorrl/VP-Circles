# VP-Circles

A **Valeriepieris Circle (VP circle)** is the smallest possible circle that contains a given fraction of the total population (or mass) in a spatial raster dataset.  
It provides a data-driven way to describe the characteristic spatial scale of population concentration.

**VP-Circles** is a Python library for computing Valeriepieris Circles on spatial raster data, with efficient algorithms, visualization tools, and centralization metrics.


## Features

- Compute Valeriepieris Circles (VP circles) on 2D spatial raster data.
- Efficient three-stage search algorithm (exponential halving, binary search with pruning, and exact discrete refinement).
- FFT-based convolution for fast population aggregation within circular neighborhoods.
- Exact radius refinement using discrete squared-distance levels (sum of two squares).
- Built-in centralization metric based on VP circles.
- Visualization utilities for population rasters and VP circles.
- Ready-to-run examples and unit tests included.


## Installation

This package is not yet published on PyPI. You can install it directly from the GitHub repository.

### From GitHub

Install the latest version using `pip`:

```bash
pip install git+https://github.com/euorrl/VP-Circles.git
```

All required dependencies will be installed automatically.

### Development installation

If you want to modify the code or contribute to the project, clone the repository and install it in editable mode:

```bash
git clone https://github.com/euorrl/VP-Circles.git
cd VP-Circles
pip install -e .
```

## Quickstart

This example computes Valeriepieris Circles (VP circles) on a 2D population raster, visualizes the result, and reports a centralization statistic.

```python
import numpy as np
import matplotlib.pyplot as plt

from vp_circles import find_vp_circles, centralization, plot_population, plot_vp_circles

# Create a synthetic population raster (smooth background + multiple hotspots)
H, W = 120, 160
yy, xx = np.mgrid[0:H, 0:W]

def gaussian2d(y0, x0, amp, sigma):
    return amp * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * sigma ** 2))

# Baseline + hotspots
population = 2.0 + (
    gaussian2d(30, 40, amp=120, sigma=10) +
    gaussian2d(70, 90, amp=200, sigma=14) +
    gaussian2d(85, 130, amp=160, sigma=12)
)

# Small random variation (optional)
rng = np.random.default_rng(0)
population *= (1.0 + 0.08 * rng.standard_normal(size=(H, W)))
population = np.clip(population, 0.0, None)

# Target fraction of total population inside the circle
f = 0.5

# Compute the VP circle (optimal radius and center mask)
r_star, best_mask = find_vp_circles(population, f)

# Compute centralization (region A defaults to the full grid)
C = centralization(H, W, r_star, best_mask, region_mask=None, reduce="min")

print(f"Optimal radius r*: {r_star:.2f}")
print(f"Number of optimal centers: {best_mask.sum()}")
print(f"Centralization C: {C:.4f}")

# Plot input raster and VP circle overlay
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

plot_population(population, ax=axes[0], log_scale=True, title="Population (log1p)")
plot_vp_circles(population, r_star, best_mask, ax=axes[1], log_scale=True, title="VP circle overlay")

plt.tight_layout()
plt.show()
```

## API Overview

The package exposes a small set of high-level functions for computing and analyzing Valeriepieris Circles, along with optional visualization utilities.

### Core API

- **`find_vp_circles(population, f, *, candidate_cap=32)`**  
  Compute the Valeriepieris Circle for a 2D population raster.  
  Returns the optimal radius `r_star` and a boolean mask of optimal center locations.

- **`centralization(H, W, r_star, best_mask, *, region_mask=None, reduce="min")`**  
  Compute a centralization statistic based on the VP circle result, measuring how concentrated the population is within the optimal circle(s).

### Visualization

- **`plot_population(population, *, ax=None, log_scale=True, ...)`**  
  Visualize a 2D population raster.

- **`plot_vp_circles(population, r_star, best_mask, *, ax=None, ...)`**  
  Plot the population raster with the optimal VP circle(s) overlaid.

### Low-level utilities

The following functions are primarily used internally by the algorithm, but are exposed for advanced use:

- **`disk_kernel(r)`**  
  Create a discrete circular kernel with radius `r`.

- **`fft_convolve2d(a, k)`**  
  Perform fast 2D convolution using FFT, returning an output with the same shape as the input.

- **`sum_of_two_squares_in_range(l, r)`**  
  Return all integers in `[l, r]` that can be written as `a^2 + b^2`, used for exact discrete radius refinement.


## Algorithm Overview

The algorithm for finding vp circles has the following three stages, please refer to **`Algorithm.md`** for details:
1. **Stage I: Exponential halving**  

2. **Stage II: Binary search with pruning**  

3. **Stage III: Exact discrete refinement** 

## Examples

The `examples/` directory contains concise, self-contained scripts illustrating how to use the main features of the VP-Circles library.  
Each script focuses on a specific task and can be executed independently.

---

### Overview

- **`demo_find_vp_circles.py`**  
  Core example demonstrating how to compute VP circles from a population raster for a given target fraction `f`.

- **`demo_centralization.py`**  
  Demonstrates how to compute a VP-circle-based centralization index.

- **`demo_plot_population.py`**  
  Visualizes a population raster.

- **`demo_plot_vp_circles.py`**  
  Visualizes VP circle(s) and their center(s) over a population raster.

- **Other scripts**  
  Additional examples illustrate supporting concepts such as disk kernels, FFT-based convolution, and discrete squared-distance levels used internally by the algorithm.

---

### Running the examples

Examples can be executed directly from the project root. For example, run:

```bash
python examples/demo_find_vp_circles.py
```

Some scripts will open a figure window to display the results.


## Testing

The `tests/` directory contains unit tests for the main components of the VP-Circles library.  
Tests are written using `pytest` and are intended to verify numerical correctness, algorithmic behavior, and API stability.

---

### Test coverage

The test suite covers the following aspects:

- disk kernel construction;
- FFT-based convolution;
- discrete squared-distance enumeration;
- VP circle computation;
- centralization index calculation;
- population and VP circle visualization utilities.

Each test file corresponds to a specific module in the `vp_circles/` package.

---

### Running the tests

To run **all tests**, execute the following command from the project root:

```bash
pytest
```
To run a **specific test file**, for example:

```bash
pytest tests/test_find_vp_circles.py
```

All tests should pass **without errors** when the library is correctly installed and configured.


## References

- Arthur, R. (2024), Valeriepieris Circles for Spatial Data Analysis. Geographical Analysis, 56: 514-529. https://doi.org/10.1111/gean.12383

- `valeriepieris` (reference implementation): https://github.com/rudyarthur/valeriepieris


## License

This project is released under the **MIT License**.

You are free to use, modify, and distribute this software, provided that the original copyright notice and license terms are included in all copies or substantial portions of the software.

See the `LICENSE` file in this repository for the full license text.
