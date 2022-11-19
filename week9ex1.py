from src.monte_carlo import monte_carlo_estimate
from src.stochastictypes.RV import RealRV
import numpy as np
from scipy.special import erf

from src.variance_reduction import control_variates_variable
from src.integration import cmc_integrate, latin_hypercube, latin_hypercube_intergrate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")

if __name__ == "__main__":
    d = 1
    omega_1 = 1/2
    c = np.ones(d) * 9 / d
    f = lambda x: np.cos(2 * np.pi * omega_1 + np.sum(x * c, axis=-1))

    #f = lambda x: x ** 2
    print(latin_hypercube_intergrate(d, f, 10_000, .05))
    print(cmc_integrate(d, f, 10_000, .05))
    exact = np.real(np.exp(2j * np.pi * omega_1) * np.prod(1 / (c * 1j) * (np.exp(1j * c) - 1)))
    print(exact)

    lhs = latin_hypercube(2, 1000)(1)
    plt.scatter(lhs[0, :, 0], lhs[0, :, 1])
    plt.savefig("plots/lhc.png")

    c = np.ones(d) * 7.25 / d
    omega = np.ones(d) * 1 / 2
    f2 = lambda x: np.prod(1 / (c ** -2 + (x - omega) ** 2), axis=-1)
    #f2 = lambda x: x ** 2
    exact = np.prod(c * (np.arctan(c * (1 - omega)) + np.arctan(c * omega)))
    print(latin_hypercube_intergrate(d, f2, 10_000, 0.05))
    print(cmc_integrate(d, f2, 10_000, .05))
    print(exact)

    c = np.ones(d) * 7.03 / d
    f3 = lambda x: np.exp(- np.sum(c ** 2 * (x - omega) ** 2, axis=-1))
    exact = np.prod(np.sqrt(np.pi) / (2 * c) * (erf(c * (1 - omega)) + erf(c * omega)))
    print(latin_hypercube_intergrate(d, f3, 10_00, 0.05))
    print(cmc_integrate(d, f3, 10_000, .05))
    print(exact)
