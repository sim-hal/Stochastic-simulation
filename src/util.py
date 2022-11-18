from typing import Callable, Iterable, Sequence, Type, Union, Any, TypeVar, Collection, Generic
import numpy.typing as npt
import numpy as np
import math
from scipy import interpolate as intrp
from multipledispatch import dispatch

C = TypeVar("C", bound=Collection)
Random = Callable[[int], C]
RealArray = npt.NDArray[np.floating]
Real = RealArray | float | np.floating
RealFunction = Callable[[RealArray], RealArray]
RandomVariable = Random[RealArray]
StochasticProcess = Callable[..., RandomVariable]

T = TypeVar("T", RandomVariable, StochasticProcess)



def callable_or_not_add(a: Callable, b: Callable | float):
    return lambda s: a(s) + (b(s) if callable(b) else b) 


def critical_value_KS(n: int, alpha: float):
    alpha1 = alpha / 2
    if n <= 20:
        a1 = np.array([0.00500, 0.01000, 0.02500, 0.05000, 0.10000])
        exact = np.array([[0.99500, 0.99000, 0.97500, 0.95000, 0.90000],
        [0.92929, 0.90000, 0.84189, 0.77639, 0.68377],
        [0.82900, 0.78456, 0.70760, 0.63604, 0.56481],
        [0.73424, 0.68887, 0.62394, 0.56522, 0.49265],
        [0.66853, 0.62718, 0.56328, 0.50945, 0.44698],
        [0.61661, 0.57741, 0.51926, 0.46799, 0.41037],
        [0.57581, 0.53844, 0.48342, 0.43607, 0.38148],
        [0.54179, 0.50654, 0.45427, 0.40962, 0.35831],
        [0.51332, 0.47960, 0.43001, 0.38746, 0.33910],
        [0.48893, 0.45662, 0.40925, 0.36866, 0.32260],
        [0.46770, 0.43670, 0.39122, 0.35242, 0.30829],
        [0.44905, 0.41918, 0.37543, 0.33815, 0.29577],
        [0.43247, 0.40362, 0.36143, 0.32549, 0.28470],
        [0.41762, 0.38970, 0.34890, 0.31417, 0.27481],
        [0.40420, 0.37713, 0.33760, 0.30397, 0.26588],
        [0.39201, 0.36571, 0.32733, 0.29472, 0.25778],
        [0.38086, 0.35528, 0.31796, 0.28627, 0.25039],
        [0.37062, 0.34569, 0.30936, 0.27851, 0.24360],
        [0.36117, 0.33685, 0.30143, 0.27136, 0.23735],
        [0.35241, 0.32866, 0.29408, 0.26473, 0.23156]])
        criticalValue = intrp.interp1d(a1 , exact[n-1,:], kind = 'cubic')(alpha1)
    else: 
        A = 0.09037 * (-math.log(alpha1, 10))**1.5 + 0.01515 * math.log(alpha1,10)**2 - 0.08467 * alpha1 
        asymptoticStat = np.sqrt(-0.5*np.log(alpha1)/n)
        criticalValue = asymptoticStat - 0.16693 / n - A / n**1.5
        criticalValue = np.min([criticalValue, 1-alpha1])
    return criticalValue



