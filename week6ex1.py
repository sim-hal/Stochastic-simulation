from statistics import mean
import numpy as np
import scipy.stats as stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from src.monte_carlo import sequential_monte_carlo_sample_mean, two_stage_monte_carlo_sample_mean
from src.generate import multivariate_normal
from src.variance_reduction import importance_sampling_variable

from src.generate import inverse_method

if __name__ == "__main__":
    alpha = 0.05
    epsilon = 1 / 10
    sigma = np.array([[4, -1], [-1, 4]])
    X = multivariate_normal(np.zeros(2), sigma)
    a = 1
    phi = lambda z: np.all(z > a, axis=1)
    Z = lambda size: phi(X(size))
    crude_estimate = sequential_monte_carlo_sample_mean(Z, alpha, 100_000, next_N=lambda N: N + 1)
    print(crude_estimate)
    x_star = np.array([a, a])
    sigma_inv = np.linalg.inv(sigma)
    w = lambda x: np.exp(-np.sum((x_star.T @ sigma_inv)[:, None] * x.transpose(), axis=0) + (1 / 2) * x_star.T @ sigma_inv @ x_star)
    G = multivariate_normal(np.array([a, a]), sigma)
    IS = importance_sampling_variable(phi, G ,w)
    importance_sampling_estimate = sequential_monte_carlo_sample_mean(IS, alpha, 100_000, next_N=lambda N: N + 1)
    print(importance_sampling_estimate)