import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from src.empirical_testing import kolmogorov_smirnov, chi_squared_test_uniform, comparative_plot, qq_plot


if __name__ == "__main__":
    samples = stats.uniform.rvs(size=10_00)
    comparative_plot(samples, stats.uniform.cdf)
    qq_plot(samples, stats.uniform.cdf)
    print(f"passed test: {kolmogorov_smirnov(samples, stats.uniform.cdf, .1)}")
    print(f"passed test: {chi_squared_test_uniform(20, samples, .1)}")
