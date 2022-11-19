
from src.monte_carlo import monte_carlo_estimate
from src.stochastictypes.RV import RealRV
from src.generate import random_walk
import numpy as np

from src.stochastictypes.operations import rrv_sum



if __name__ == "__main__":
    mu = 1
    sigma = 4
    d = 10
    N = 100
    L = 1.
    h = L / N
    n = np.arange(d) + 1
    Y = RealRV(lambda size: np.random.uniform(low=-1, high=1, size=(size, d)))
    a1 = lambda x: mu + sigma / np.pi ** 2 * rrv_sum(Y * (np.cos(np.pi * n * x) / n ** 2))
    Z = h * sum(a1(i * h + h / 2) ** -1 for i in range(N))

    print(monte_carlo_estimate(Z, 200, .05))
    
