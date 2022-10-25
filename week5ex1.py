import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n = 2
    N = 100
    Z = lambda size: np.linalg.norm(stats.uniform.rvs(size=(N, size), loc=-1, scale=2) , axis=1) < 1
    print(np.mean(Z(n)))
