from week1ex1 import chi_squared_test_uniform
import numpy as np
from src.LCG import LCG

if __name__ == "__main__":
    lcg = LCG(1000)
    rvs = np.array([lcg() for _ in range(10)])
    print(rvs)
    print(chi_squared_test_uniform(20, rvs, 0.1))
