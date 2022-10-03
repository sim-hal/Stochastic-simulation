import numpy as np
import matplotlib.pyplot as plt
from src.util import RealArray, RealFunction, critical_value_KS
import scipy.stats as stats

def empirical_cdf(samples, t: RealArray) -> RealArray:
    return np.array([(samples <= t_i).mean() for t_i in t])

def qq_plot(samples: RealArray, theoretical_cdf: RealFunction):
    n = len(samples)
    plt.plot(theoretical_cdf(np.sort(samples)), np.linspace(0, 1, n))
    plt.savefig("plots/q-q.png")
    plt.clf()

def comparative_plot(samples: RealArray, theoretical_cdf: RealFunction):
    n = len(samples)
    empirical_cdf_increments = np.sort(samples)
    t = np.linspace(np.min(samples) - 1, np.max(samples) + 1, 1000)
    F_hat = empirical_cdf(samples, t)
    plt.plot(t, F_hat)
    plt.plot(t, theoretical_cdf(t))
    plt.savefig("plots/cumm_dist.png")
    plt.clf()

def kolmogorov_smirnov(samples: RealArray, theoretical_cdf: RealFunction, alpha: float):
    n = len(samples)
    t = np.linspace(np.min(samples) - 1, np.max(samples) + 1, 1000)
    empirical_F = empirical_cdf(samples, t)
    D_n = np.max(empirical_F - theoretical_cdf(t))
    return D_n <= critical_value_KS(n, alpha)

def chi_squared_test_uniform(m: int, rvs: np.ndarray, alpha: float=.1):
    """
    NOTE: Tests for uniform distribution
    """
    n = len(rvs)
    intervals = np.pad(np.sort(stats.uniform.rvs(size=m)), (1, 1), 'constant', constant_values=(0, 1))
    p = np.diff(intervals)
    Q_m = 0
    for left, right, p_i in zip(intervals[:-1], intervals[1:], p):
        N_i = np.sum(np.logical_and(left < rvs, rvs < right))
        Q_m += (N_i - n * p_i) ** 2 / (n * p_i)
    return Q_m <= stats.chi2.ppf(1 - 0.1, m)

