from typing import Optional
import numpy as np
from src.util import RandomVariable, RealArray
import scipy.stats as stats


def two_stage_monte_carlo_sample_mean(Z: RandomVariable, alpha=.95, N_tilde: int=100, tol:float=1e-6, initial_sample:Optional[RealArray]=None) -> tuple[float, float, float]:
    if initial_sample is None:
        initial_sample = Z(N_tilde) 
    initial_est_mean = np.mean(initial_sample)
    initial_est_var= 1 / (N_tilde - 1) * np.sum((initial_sample - initial_est_mean) ** 2)
    c_alpha = stats.norm.ppf(1 - alpha/2)
    N = int(c_alpha ** 2 * initial_est_var / tol ** 2)
    sample = Z(N)
    est_mean = float(np.mean(sample))
    est_var = 1 / (N - 1) * np.sum((sample - est_mean) ** 2)
    if est_var > initial_est_var:
        return two_stage_monte_carlo_sample_mean(Z, alpha, N, tol, sample)
    return est_mean, est_mean - c_alpha * np.sqrt(est_var / N), est_mean + c_alpha * np.sqrt(est_var / N)

def sequential_monte_carlo_sample_mean(Z: RandomVariable, alpha=10 ** -1.5, N_k: int=100, tol:float=1/10):
    sample = Z(N_k) 
    est_mean = np.mean(sample)
    est_var= 1 / (N_k - 1) * np.sum((sample - est_mean) ** 2)
    c_alpha = stats.norm.ppf(1 - alpha/2)
    if c_alpha ** 2 * est_var / N_k > tol ** 2:
        return sequential_monte_carlo_sample_mean(Z, alpha, N_k * 2, tol)
    N = N_k
    sample = Z(N)
    est_mean = np.mean(sample)
    est_var = 1 / (N - 1) * np.sum((sample - est_mean) ** 2)
    print(N)
    return est_mean, est_mean - c_alpha * np.sqrt(est_var / N), est_mean + c_alpha * np.sqrt(est_var / N)
