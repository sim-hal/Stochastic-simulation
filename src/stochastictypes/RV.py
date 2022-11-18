from functools import total_ordering
from typing import Callable, Generic, Iterable, Self, TypeVar
import numpy.typing as npt
from src.util import Real, RealArray
import numpy as np

D = TypeVar("D", bound=Iterable)

class RV(Generic[D]):
    def __init__(self, gen: Callable[[int], D]):
        self.gen = gen
    
    def __call__(self, size: int) -> D:
        return self.gen(size)

class RealRV(RV[RealArray]):
    def __init__(self, gen: Callable[[int], RealArray]):
        super().__init__(gen)
    
    def __add__(self, other: Real | Self) -> Self:
        def new(size: int):
            v = other(size) if isinstance(other, RealRV) else other 
            return self(size) + v
        return RealRV(new)

    def __mul__(self, other: Real | Self):
        def new(size: int):
            v = other(size) if isinstance(other, RealRV) else other 
            return self(size) * v
        return RealRV(new)
    def __sub__(self, other: Real | Self) -> Self:
        def new(size: int):
            v = other(size) if isinstance(other, RealRV) else other 
            return self(size) - v
        return RealRV(new)
    def __div__(self, other: Real | Self):
        def new(size: int):
            v = other(size) if isinstance(other, RealRV) else other 
            return self(size) / v
        return RealRV(new)

    def __lt__(self, other: Real | Self) ->  Self:
        def new(size: int):
            v = other(size) if isinstance(other, RealRV) else other 
            return np.array(self(size) < v, dtype=float)
        return RealRV(new)
    def __eq__(self, other: Real | Self) ->  Self:
        def new(size: int):
            v = other(size) if isinstance(other, RealRV) else other 
            return np.array(self(size) == v, dtype=float)
        return RealRV(new)
    def __le__(self, other: Real | Self) ->  Self:
        def new(size: int):
            v = other(size) if isinstance(other, RealRV) else other 
            return np.array(self(size) <= v, dtype=float)
        return RealRV(new)
    def __ge__(self, other: Real | Self) ->  Self:
        def new(size: int):
            v = other(size) if isinstance(other, RealRV) else other 
            return np.array(self(size) >= v, dtype=float)
        return RealRV(new)

    def __pow__(self, p: Real | Self) ->  Self:
        def new(size: int):
            v = p(size) if isinstance(p, RealRV) else p 
            return self(size) ** v
        return RealRV(new)


