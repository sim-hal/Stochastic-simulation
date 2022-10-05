from typing import Callable
import scipy.stats as stats
import numpy as np
from src.generate import acceptance_rejection


if __name__ == "__main__":
    f_tilde = lambda x: (np.sin(6 * x) ** 2 + 3 * np.cos(x) ** 2 * np.sin(4 * x) ** 2 + 1) * np.exp(-x ** 2 / 2)
    g = stats.norm.pdf
    c = 5 * np.sqrt(2 * np.pi)
    N = 100_000_000
    X = acceptance_rejection(f_tilde, g, lambda size: stats.norm.rvs(size=size), c)
    samples = X(N)
    attempts = X.counter
    print(attempts / (N * c))



