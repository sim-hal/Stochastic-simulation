from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from src.generate import inverse_method

if __name__ == "__main__":
    cdf = lambda x: 0 if x < 0 else 1 - 2/3 * np.exp(-x / 2) if x < 2 else 1
    inv_cdf = lambda y: 0 if y < 1/3 else -2 * np.log((3 / 2) * (1 - y)) if y < 1 - 2/3 * np.exp(-1) else 2
    samples = [inverse_method(inv_cdf) for _ in range(1000)]
    x = np.linspace(-1, 3, 1000)
    plt.plot(x, [cdf(x_i) for x_i in x])
    plt.plot(np.sort(samples), np.linspace(0, 1, len(samples), endpoint=True))
    plt.savefig("plots/inverse_method.png")