from typing import Union
from src.util import RandomVariable, RealArray, RealFunction, Real, Random
import numpy as np

def importance_sampling_variable(phi: RealFunction, dominating_variable: RandomVariable, w: RealFunction) -> Random[RealArray]:
    def IS(size: int):
        samples = dominating_variable(size)
        return phi(samples) * w(samples)
    return IS

def antithetic_variable(phi: RealFunction, X: RandomVariable, mu_x: Union[float, RealArray]) -> Random[RealArray]:
    def AV(size: int):
        samples = X(size)
        return (phi(samples) + phi(2 * mu_x - samples)) / 2
    return AV

def control_variates_variable(Z: RandomVariable, Y: RandomVariable, mean_y: Real, cov_ZY: Real, var_Y: Real) -> Random[RealArray]:
    alpha = cov_ZY / var_Y
    return lambda size: Z(size) - alpha * (Y(size) - mean_y)
    
