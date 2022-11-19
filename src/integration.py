"""
    Stochastic methods for integrating on the hyper unit cube
"""
from typing import Callable
from src.util import Real, RealFunction
from src.monte_carlo import monte_carlo_estimate, sample_estimates
from src.generate import latin_hypercube
from src.stochastictypes.RV import RealRV, RV
import numpy as np



def cmc_integrate(d: int, f: RealFunction, N: int, alpha: float, K=50):
    V = RealRV(lambda size: f(np.random.uniform(size=(size, N, d))))
    I = RealRV(lambda size: np.mean(V(size), axis=1))
    return monte_carlo_estimate(I, K, alpha)

def latin_hypercube_intergrate(d: int, f: RealFunction, N: int, alpha: float, K = 50):
    P = latin_hypercube(d, N)
    V = RealRV(lambda size: f(P(size)))
    I = RealRV(lambda size: np.mean(V(size), axis=1))
    return monte_carlo_estimate(I, K, alpha)
