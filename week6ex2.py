from statistics import mean
import numpy as np
import scipy.stats as stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from src.monte_carlo import monte_carlo_estimate
from src.generate import brownian_process
from src.variance_reduction import antithetic_variable

# NOTE: Not correct

if __name__ == "__main__":
    alpha = 0.05
    S_0 = 5
    m = 1000
    T = 2
    t = np.arange(0, m)* T / m
    X = brownian_process()(t)
    phi = lambda x: S_0 * np.exp((0.5 - 0.3 ** 2 / 2) * t * 0.3 * x)
    S = lambda size: phi(X(size))
    S_antithetic = antithetic_variable(phi, X, 0)
    B = 4
    K =  5
    def PSI(size):
        samples = S(size)
        net = samples[:, -1] - K
        return (np.abs(net) + net) / 2 * np.all(samples > B, axis=1)
    def PSI_anti(size):
        samples = S_antithetic(size)
        net = samples[:, -1] - K
        return (np.abs(net) + net) / 2 * np.all(samples > B, axis=1)
    crude_estimate = monte_carlo_estimate(PSI, 100_000, .05,)
    antithetic_estimate = monte_carlo_estimate(PSI_anti, 50_000, .05)
    print(crude_estimate)
    print(antithetic_estimate)