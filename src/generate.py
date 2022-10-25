from ctypes import ArgumentError
from typing import Callable, Sequence, Union
import numpy as np
from src.util import RandomVariable, RandomVariables, RealArray, RealFunction, StateSpace, StochasticProcess, StochasticProcessOnUniformGrid
import scipy.stats as stats
from random import choices
import numpy.typing as npt

def inverse_method(inv_cdf: RealFunction) -> RandomVariable:
    return lambda size: inv_cdf(stats.uniform.rvs(size=size))

def acceptance_rejection(f_tilde: RealFunction, g: RealFunction, Y: RandomVariable, c: float) -> RandomVariable:
    def X(size: int=1) -> RealArray:
        x = np.zeros(size)
        def fill_x(x: RealArray):
            X.counter += len(x)
            u = stats.uniform.rvs(size=len(x))
            y = Y(len(x))
            accept = u <= f_tilde(y) / (c * g(y))
            x[accept] = y[accept]
            failed = len(x) - np.sum(accept)
            if failed > 0:
                fill_x(x[~accept])
            return x
        return fill_x(x)
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

def box_muller() -> RandomVariable:
    U = stats.uniform.rvs
    def X(size=1):
        """
        Size must be divisible by 2, otherwise size - 1 samples will be generated
        """
        u: RealArray = stats.uniform.rvs(size=size // 2)
        v: RealArray = stats.uniform.rvs(size=size // 2)
        rho: RealArray = np.sqrt(-2 * np.log(u))
        theta: RealArray = 2 * np.pi * v
        x: RealArray = np.multiply(rho , np.cos(theta))
        y = np.multiply(rho , np.sin(theta))
        return np.array(x, y).flatten()
    return X

def multivariate_normal(mu: RealArray, sigma: RealArray) -> RandomVariable:
    # TODO: Pivoted cholesky in the case of singular sigma
    """
    expectation mu in R^d
    covariance matrix sigma in R^(dxd)
    Returns a multivariate random variable in R^d, 
    """
    def X(size: int):
        n = len(sigma)
        A = np.linalg.cholesky(sigma)
        y = np.random.randn(size, n)
        print((y @ A).shape)
        return mu + y @ A
    return X

def conditional_multivariate_normal(mu_y: RealArray, mu_z: RealArray, sigma_yy: RealArray, sigma_zz: RealArray, sigma_yz: RealArray, z: RealArray) -> RandomVariable:
    """
    Returns the random variable X = Y | Z = z
    """
    inv_sigma_zz = np.linalg.inv(sigma_zz)
    #H = np.linalg.solve(sigma_zz, sigma_yz.transpose())
    #h = np.linalg.solve(sigma_zz, (z - mu_z))
    sigma_x = sigma_yy - sigma_yz @ inv_sigma_zz @ sigma_yz.transpose()
    mu_x = mu_y + (z - mu_z) @ (sigma_yz @ inv_sigma_zz).transpose()
    return multivariate_normal(mu_x, sigma_x)


def gaussian_process(expectation: RealFunction, covariance: Callable[[RealArray, RealArray], RealArray]) -> StochasticProcess:
    def P(t: RealArray):
        T, S = np.meshgrid(t, t)
        sigma = covariance(T, S)
        mu = expectation(t)
        X = multivariate_normal(mu, sigma)
        return X(1)[0, :]
    return P

def stationary_gaussian_process(expectation: RealFunction, covariance: RealFunction) -> StochasticProcessOnUniformGrid:
    """
    Constructs a stationary gassian process
    Equivalant to the function gaussian_process with a shift invariant covariance function, but faster as it uses FFT with circular embedding
    """
    def P(a: float, b: float, n: int):
        t = np.linspace(a, b, n)
        cov = covariance(t)
        alpha = np.r_[cov, cov[-1:1:-1]]
        lmbda = np.fft.fft(alpha)
        Y = np.random.randn(2 * (n - 1)) + 1j* np.random.randn(2 * (n - 1))
        X_tilde = np.fft.ifft(np.sqrt(2*n * lmbda) * Y)
        return expectation(t) + np.real(X_tilde[:n])
    return P



def brownian_process() -> StochasticProcess:
    def P(t: RealArray):
        dt = np.diff(t, prepend=t[0])
        increments = np.random.normal(scale=dt)
        return np.cumsum(increments)
    return P

#def markov_chain(pn_ij: Callable[[int, StateSpace, StateSpace], float], state_space: StateSpace, P_0: RealArray):
#    I, J = np.meshgrid(state_space, state_space)
#    def P(N: int):
#        
#        return np.array([pn_ij(n, I, J) @ ])