import numpy as np
from src.generate import stationary_gaussian_process
import matplotlib.pyplot as plt

if __name__ == "__main__":
    a = 0
    b = 1
    n = 50
    h = (b - a) / n
    H = 0.5
    C_dBH = lambda diff: 1/2 * (  np.abs(-diff + h) ** (2 * H) + np.abs(diff + h) ** (2 * H) - 2 *  diff ** (2 * H))
    P = stationary_gaussian_process(lambda t: np.zeros(len(t)), C_dBH)
    dB = P(0, 1, 50)
    plt.plot(np.linspace(a, b, n), dB)
    plt.savefig("plots/fft_gaussian.png")
    plt.clf()
    plt.plot(np.linspace(a, b, n), dB.cumsum())
    plt.savefig("plots/fft_gaussian_cummulated.png")