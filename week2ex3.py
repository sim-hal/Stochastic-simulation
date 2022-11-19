import numpy as np
from src.empirical_testing import kolmogorov_smirnov
from src.generate import box_muller, composite_method, acceptance_rejection
import scipy.stats as stats
import time

from src.stochastictypes.RV import RealRV

if __name__ == "__main__":
    c = np.sqrt(2 * np.e / np.pi)
    g = lambda x: np.exp(-np.abs(x)) / 2
    Y = composite_method([RealRV(lambda size: stats.expon.rvs(size=size)), RealRV(lambda size: -stats.expon.rvs(size=size))], [1/2, 1/2])
    f_tilde = lambda x: (1 / np.sqrt(2 * np.pi) * np.exp(- x ** 2 / 2))
    N = 1_000_000
    t_0 = time.time()
    X = acceptance_rejection(f_tilde, g, Y, c)
    samples = X(N)
    t = time.time() - t_0
    print(t)
    print(kolmogorov_smirnov(samples, stats.norm.cdf, 0.1))
    t_0 = time.time()
    X_Y = box_muller()
    samples2 = np.array(X_Y(500)).flatten()
    t = time.time() - t_0
    print(t)
    print(kolmogorov_smirnov(samples2, stats.norm.cdf, 0.1))