from src.generate import gaussian_process, random_walk, stationary_gaussian_process
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.util import RealArray
from src.stochastictypes.RV import RV, RealRV
from src.stochastictypes.SP import SP
from src.stochastictypes.operations import sp_exp
from src.monte_carlo import stratified_monte_carlo_estimate


if __name__ == "__main__":
    S = 12
    T = 1
    db = 1 / S
    M = 1000
    t = np.linspace(1/M, T - 1 / M, M)
    a = 0

    brownian_bridge = gaussian_process(lambda t: t * 5, lambda s, t: np.minimum(s, t) - s * t)

    print(brownian_bridge(t)(1))


    def stratified_standard_wiener(p1: float, p2: float) -> RV[list[SP]]:
        def randomProcess(size: int):
            u = np.random.uniform(p1, p2, size=size)
            b = stats.norm.ppf(u) + a
            procs = [gaussian_process(lambda t, b_i=b_i: a + t * (b_i - a), lambda s, t: np.minimum(s, t) - s * t) for b_i in b]
            return procs
        return RV(randomProcess)
    

    str_vars = [stratified_standard_wiener(db * i, db * (i + 1)) for i in range(S)]
    for str_var in str_vars:
        processes = str_var(2)
        plt.plot(t, processes[0](t)(1).flatten(), t, processes[1](t)(1).flatten())
    plt.savefig("plots/strat_brownian_motion")

    r = .05
    sigma = .3
    M = 100
    t = np.linspace(1/M, T - 1 / M, M)
    X_0 = 6

    def stratified_brownian_motion(p1: float, p2: float) -> RV[list[SP]]:
        W_stratified = stratified_standard_wiener(p1, p2)
        def randomProcess(size: int):
            W = W_stratified(size)
            return [sp_exp(W_i * sigma  + (r - sigma ** 2 / 2)) * X_0 for W_i in W]
        return RV(randomProcess)


    

    N = S * 100
    N_prop = np.ones(S) * 100
    str_vars = [stratified_brownian_motion(db * i, db * (i + 1)) for i in range(S)]
    

