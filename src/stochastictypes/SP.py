from src.stochastictypes.RV import RealRV
from typing import Callable, Self, TypeVarTuple, Generic
from src.util import Real, RealArray
import numpy as np

Is = TypeVarTuple("Is")

class SP(Generic[*Is]):
    def __init__(self, gen: Callable[[*Is], RealRV]):
        self.gen= gen
    
    def __call__(self, *args: *Is) -> RealRV:
        return self.gen(*args)

    def __add__(self, other: Self | Real) -> Self:
        def new(*args: *Is):
            v = other(*args) if callable(other) else other 
            return self(*args) + v
        return SP(new)

    def __mul__(self, other: Self | Real) -> Self:
        def new(*args: *Is):
            v = other(*args) if callable(other) else other 
            return self(*args) * v
        return SP(new)
    
    def __sub__(self, other: Self | Real) -> Self:
        def new(*args: *Is):
            v = other(*args) if callable(other) else other 
            return self(*args) - v
        return SP(new)

    def __div__(self, other: Self | Real) -> Self:
        def new(*args: *Is):
            v = other(*args) if callable(other) else other 
            return self(*args) + v
        return SP(new)