from ctypes import ArgumentError
import numpy as np
from src.util import RandomVariable, RandomVariables, RealArray, RealFunction
import scipy.stats as stats
from random import choices

def inverse_method(inv_cdf: RealFunction) -> RandomVariable:
    return lambda size: inv_cdf(stats.uniform.rvs(size=size))

def acceptance_rejection(f_tilde: RealFunction, g: RealFunction, Y: RandomVariable, c: float) -> RandomVariable:
    def X(size: int=1) -> RealArray:
        X.counter += size
        u = stats.uniform.rvs(size=size)
        y = Y(size)
        accept = u <= f_tilde(y) / (c * g(y))
        x = np.zeros(size)
        x[accept] = y[accept]  # type: ignore
        failed = size - np.sum(accept)
        if failed > 0:
            x[~accept] = X(size=size - np.sum(accept))
        return x
    X.counter = 0
    return X

def composite_method(rvs: list[RandomVariable], p: list[float]) -> RandomVariable:
    def X(size: int=1) -> RealArray:
        if np.sum(p) != 1:
            raise ValueError("sum of p must be 1")
        ch = np.random.choice(rvs, p=p, size=size) # type: ignore
        x = np.zeros(size)
        for rv in rvs:
            chosen = rv == ch
            x[chosen] = rv(np.sum(chosen)) 
        return x
    return X

def box_muller() -> RandomVariables:
    U = stats.uniform.rvs
    def X_Y(size=1):
        u: RealArray = stats.uniform.rvs(size=size)
        v: RealArray = stats.uniform.rvs(size=size)
        rho: RealArray = np.sqrt(-2 * np.log(u))
        theta: RealArray = 2 * np.pi * v
        x: RealArray = np.multiply(rho , np.cos(theta))
        y = np.multiply(rho , np.sin(theta))
        return x, y
    return X_Y