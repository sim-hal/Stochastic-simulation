from ex1 import chi_squared_test
import numpy as np
class LCG:
    a = 3
    b = 0
    m = 31
    def __init__(self, seed: int) -> None:
        self.state = seed % self.m
    
    def __call__(self) -> float:
        self.state = ((self.a * self.state + self.b) % self.m)
        return self.state / self.m

if __name__ == "__main__":
    lcg = LCG(1000)
    rvs = np.array([lcg() for _ in range(10)])
    print(rvs)
    print(chi_squared_test(20, rvs, 0.1))
