import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

def kolmogorov_smirnov_test(dist, rvs: np.ndarray):
    """
    Using alpha = 0.10
    rvs is the samples, testing against the distribution dist
    n (length of rvs) has to be greater than 45, as we use K_alpha = 1.22


    """
    n = len(rvs)
    if n <= 45:
        raise ValueError("n has to be greater than 45 to ensure convergence to the alpha-quantile of K")
    empirical_cdf_increments = np.sort(rvs)
    min_sample = empirical_cdf_increments[0]
    max_sample = empirical_cdf_increments[-1]
    D_n = np.max(np.abs(np.linspace(0, 1, n) - dist.cdf(empirical_cdf_increments)))
    plt.plot(empirical_cdf_increments, np.linspace(0, 1, n))
    plt.plot(np.linspace(min_sample, max_sample, 100), dist.cdf(np.linspace(min_sample, max_sample, 100)))
    plt.savefig("plots/cumm_dist.png")
    plt.clf()
    plt.plot(dist.cdf(empirical_cdf_increments), np.linspace(0, 1, n))
    plt.savefig("plots/q-q.png")
    return np.sqrt(n) * D_n <= 1.22

def chi_squared_test(m: int, rvs: np.ndarray, alpha: float=.1):
    n = len(rvs)
    intervals = np.pad(np.sort(stats.uniform.rvs(size=m)), (1, 1), 'constant', constant_values=(0, 1))
    p = np.diff(intervals)
    Q_m = 0
    for left, right, p_i in zip(intervals[:-1], intervals[1:], p):
        N_i = np.sum(np.logical_and(left < rvs, rvs < right))
        Q_m += (N_i - n * p_i) ** 2 / (n * p_i)
    return Q_m <= stats.chi2.ppf(1 - 0.1, m)

def ChiSquareTest(X, K = 10, alpha = 0.1):
    """
    Chi Squared Test for data X and significance alpha with K degrees of freedom
    """
    n = len(X)
    p = np.ones(K) / K
    N = np.array([np.sum((float(i)/K < X) & (X <= (i+1.)/K)) for i in range(K)])
    QK = np.sum( (N-n*p)**2. / (n*p)) # test statistic
    critval = stats.chi2.ppf(1-alpha, K-1)
    true_chi = 1 * (QK > critval)
    rej = ['cannot be', 'is']
    stat = {'Statistic': QK, 'Quantile': critval, 'Significance': alpha}
    message = 'Chi2 test: the null hypothesis H0 ' + rej[true_chi] + ' rejected at level alpha = ' + str(alpha)
    return message, stat


if __name__ == "__main__":
    sample = stats.uniform.rvs(size=10_000)
    print(f"passed test: {kolmogorov_smirnov_test(stats.uniform, sample)}")
    print(f"passed test: {chi_squared_test(20, sample, .001)}")
    #print(ChiSquareTest(sample, 10, 0.1))