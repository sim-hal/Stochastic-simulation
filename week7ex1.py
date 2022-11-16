from src.generate import random_walk
import numpy as np



if __name__ == "__main__":
    p = np.array([.5, .5])
    increments = np.array([-1, 1])
    RW = random_walk(p, increments, 0)
    X = RW(100)
    
