import numpy as np
import matplotlib.pyplot as plt

from vp_circles.plot_population import plot_population

# create population raster
H, W = 200, 300
pop = np.random.gamma(shape=2.0, scale=50.0, size=(H, W))
pop[80:90, 120:130] += 5000  # add a hotspot

plot_population(pop, log_scale=True, title="Demo population")
plt.show()
