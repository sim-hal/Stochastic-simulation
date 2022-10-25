from statistics import mean
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from src.monte_carlo import sequential_monte_carlo_sample_mean, two_stage_monte_carlo_sample_mean

from src.generate import inverse_method

if __name__ == "__main__":
    alpha = 10 ** -1.5
    epsilon = 1 / 10
    paretto_inv_cdf = lambda s, gamma, x_m: (x_m - s) ** (-1 / gamma)
    P = inverse_method(lambda s: paretto_inv_cdf(s, 3.1, 1))
    mean_paretto = 3.1 / 2.1
    failures = 0
    K = int(20 / alpha)
    U = lambda size: stats.uniform.rvs(size=size, loc=-1, scale=2)
    for k in range(K):
        sample_mean, _, _ = two_stage_monte_carlo_sample_mean(P, alpha, 50, epsilon)
        failures += abs(sample_mean - mean_paretto) > epsilon
    print(failures / K, alpha)
