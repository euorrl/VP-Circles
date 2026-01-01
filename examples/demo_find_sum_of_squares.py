import numpy as np
import matplotlib.pyplot as plt

from vp_circles.find_sum_of_squares import find_sum_of_squares

# set an upper bound
r_upper = 8.0
s_upper = int(np.floor(r_upper * r_upper))

# generate all "critical squared radii" in [0, s_upper] that are a^2 + b^2
critical_squares = find_sum_of_squares(0, s_upper)

# convert squared radii to radii (float)
critical_radii = np.sqrt(np.array(critical_squares, dtype=float))

print("Critical squared radii:")
print(critical_squares)

print("\nCritical radii:")
print(critical_radii.tolist())
