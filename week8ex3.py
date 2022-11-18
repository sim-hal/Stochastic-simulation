from src.monte_carlo import monte_carlo_estimate
from src.stochastictypes.RV import RealRV
import numpy as np

from src.variance_reduction import control_variates_variable



if __name__ == "__main__":
    U = RealRV(lambda size: np.random.uniform(size=size))
    S = U + U
    Y1 = S <= 1
    Y2 = S >= np.sqrt(2)
    def temp(size: int):
        samples = S(size)
        return np.where(np.logical_and(1 < samples, samples <= np.sqrt(2)), samples, 0)
    Y3 = RealRV(temp)

    Z =  (U ** 2 + U ** 2 <= 1) * 4

    mean_Y_1 = 0.5
    var_Y_1, _ = monte_carlo_estimate((Y1 - mean_Y_1) ** 2, 10_000, .05)
    mean_Z, _ = monte_carlo_estimate(Z, 10_000, .05)
    cov_zy, _ = monte_carlo_estimate((Z - mean_Z) * (Y1 - mean_Y_1), 10_000, .05)
    Z_ai = control_variates_variable(Z, Y1, mean_Y_1, cov_zy, var_Y_1)
    print(monte_carlo_estimate(Z_ai, 10_000, .05))

    print(monte_carlo_estimate(Z, 10_000, .05))
    print(monte_carlo_estimate(Y1, 10_000, .05))
    print(monte_carlo_estimate(Y2, 10_000, .05))
    print(monte_carlo_estimate(Y3, 10_000, .05))


    
    
