import numpy as np
from src.generate import multivariate_normal
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mu = np.array([2, 1])
    sigma = np.array([[1, 2], [2, 5]])
    X = multivariate_normal(mu, sigma)
    samples = X(1_000_000)
    x = samples[0, :]
    y = samples[1, :]
    H, x_edges, y_edges = np.histogram2d(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(aspect="equal")
    X, Y = np.meshgrid(x_edges, y_edges)
    print(X.shape, Y.shape, H.shape)
    ax.pcolormesh(X, Y, H)
    fig.savefig("plots/bivariate_histogram.png")