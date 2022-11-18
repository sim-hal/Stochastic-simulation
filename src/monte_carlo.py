from typing import Optional
import numpy as np
import numpy.typing as npt
from src.util import RandomVariable, Real, RealArray, RealFunction
import scipy.stats as stats



def sample_estimates(samples: RealArray):
    N = len(samples)
    est_mean = np.mean(samples, axis=0)
    est_var = 1 / (N - 1) * np.sum((samples - est_mean) ** 2, axis=0)
    return est_mean, est_var

def asymptotic_monte_carlo_confidence_interval(alpha: float, var: Real, N: int):
    c_alpha = stats.norm.ppf(1 - alpha/2)
    return c_alpha * np.sqrt(var / N)

def monte_carlo_estimate(Z: RandomVariable, N: int, alpha: float):
    samples = Z(N)
    est_mean, est_var = sample_estimates(samples)
    return est_mean, asymptotic_monte_carlo_confidence_interval(alpha, est_var, N)

def stratified_monte_carlo_estimate(Z_strats: list[RandomVariable], p: RealArray, N: npt.NDArray[np.int64], alpha: float):
    # TODO: untested
    samples = np.array([Z_i(N_i) for Z_i, N_i in zip(Z_strats, N)])
    mu_ests, var_ests = sample_estimates(samples)
    est_mean = np.sum(p * mu_ests)
    est_var = np.sum(p ** 2 * var_ests)
    return est_mean, asymptotic_monte_carlo_confidence_interval(alpha, est_var, int(np.sum(N)))


def two_stage_monte_carlo_sample_mean(Z: RandomVariable, alpha=.05, N_tilde: int=100, tol:float=1e-6, initial_sample:Optional[RealArray]=None) -> tuple[float, float, float]:
    """_summary_

    Args:
        Z (RandomVariable): _description_
        alpha (float, optional): _description_. Defaults to .05.
        N_tilde (int, optional): _description_. Defaults to 100.
        tol (float, optional): _description_. Defaults to 1e-6.
        initial_sample (Optional[RealArray], optional): _description_. Defaults to None.

    Returns:
        tuple[float, float, float]: estimated mean, lower end of confidence interval, higher end of confidence interval
    """
    if initial_sample is None:
        initial_sample = Z(N_tilde)
    
    initial_est_mean, initial_est_var = sample_estimates(initial_sample)
    c_alpha = stats.norm.ppf(1 - alpha/2)
    N = int(c_alpha ** 2 * initial_est_var / tol ** 2)
    print(f"N = {N}")
    samples = Z(N)
    est_mean, est_var = sample_estimates(samples)
    if est_var > initial_est_var:
        return two_stage_monte_carlo_sample_mean(Z, alpha, N, tol, samples)
    confidence = asymptotic_monte_carlo_confidence_interval(alpha, est_var, N)
    return est_mean, est_mean - confidence, est_mean + confidence

def sequential_monte_carlo_sample_mean(Z: RandomVariable, alpha=10 ** -1.5, N_k: int=100, tol:float=1/10, next_N=lambda N: N * 2):
    samples = Z(N_k)
    est_mean, est_var = sample_estimates(samples)
    c_alpha = stats.norm.ppf(1 - alpha/2)
    if np.all(c_alpha ** 2 * est_var / N_k > tol ** 2):
        return sequential_monte_carlo_sample_mean(Z, alpha, next_N(N_k), tol)
    N = N_k
    samples = Z(N)
    est_mean, est_var = sample_estimates(samples)
    print(f"final N = {N}, {est_mean} +- {c_alpha * np.sqrt(est_var / N)}")
    return est_mean, est_mean - c_alpha * np.sqrt(est_var / N), est_mean + c_alpha * np.sqrt(est_var / N)