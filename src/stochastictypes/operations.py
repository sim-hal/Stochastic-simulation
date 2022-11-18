from functools import singledispatch
from src.stochastictypes.RV import RealRV
from src.stochastictypes.SP import SP
import numpy as np
from typing import Callable, TypeVar, TypeVarTuple

from src.util import Real



def rrv_exp(rrv: RealRV) -> RealRV:
    return RealRV(lambda size: np.exp(rrv(size)))

def sp_exp(sp: SP) -> SP:
    return SP(lambda *args: rrv_exp(sp(*args)))

def rrv_min(rrv: RealRV) -> RealRV:
    return RealRV(lambda size: np.min(rrv(size), axis=-1))