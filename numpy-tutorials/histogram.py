import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 3, 0.7

# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2
v = np.random.normal(mu, sigma, 10000)

# Plot a normalized histogram with 50 bins
plt.hist(v, bins=50, normed=1)
plt.show()
