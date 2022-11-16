import numpy as np
import matplotlib.pyplot as plt
from src.generate import conditional_multivariate_normal, gaussian_process
import matplotlib

if __name__ == "__main__":
    rho = 0.1
    expectation = lambda t: np.sin(2 * np.pi * t)
    covariance = lambda t, s: np.exp(-np.abs(t - s) / rho)
    P = gaussian_process(expectation, covariance)
    n = 50
    m = n - 1
    t = np.linspace(0, 1, n, endpoint=True)
    s = np.linspace((t[1] - t[0]) / 2, t[-2] + (t[-1] - t[-2]) / 2, m, endpoint=True)
    z = P(t)(1)
    sigma_yy = covariance(*np.meshgrid(s, s))
    sigma_yz = covariance(*np.meshgrid(t, s))
    sigma_zz = covariance(*np.meshgrid(t, t))
    mu_y = expectation(s)
    mu_z = expectation(t)
    X = conditional_multivariate_normal(mu_y, mu_z, sigma_yy, sigma_zz, sigma_yz, z)
    plt.plot(t, z)
    plt.savefig("plots/gaussian_process.png")
    plt.clf()
    r = np.zeros(n + m)
    r[0::2] = t
    r[1::2] = s
    samples = np.zeros((n + m))
    samples[0::2] = z
    samples[1::2] = X(1)
    plt.plot(r, samples)
    plt.savefig("plots/gaussian_process_full.png")

