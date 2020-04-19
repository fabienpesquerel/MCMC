import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mcmc import ULA, MALA
from function import MODEL

########################################
#              ANNULUS                 #
########################################


def annulus_torch(radius, sigma):
    def pdf(x):
        r = torch.norm(x)
        numerator = torch.exp(- 0.5 * ((r - radius) / sigma)**2) + \
            torch.exp(- 0.5 * ((r + radius) / sigma)**2)
        denominator = sigma * r
        return numerator / denominator
    return pdf

########################################
#              GAUSSIAN                #
########################################


def log_gradient_gaussian(mu, cov):
    def gradient(x): return - cov @ (x - mu)
    return gradient


def gaussian_torch(mu, sigma):
    def pdf(x):
        return torch.exp(- (x-mu) @ sigma @ (x - mu))
    return pdf

########################################
#          AUTO-CORRELATION            #
########################################


def auto_correlation(M, kappa=100):
    kappa = len(M) // kappa
    auto_corr = np.zeros((kappa-1, 2))
    mu = np.mean(M, axis=0)
    for s in range(1, kappa-1):
        auto_corr[s] = np.mean((M[:-s]-mu) * (M[s:]-mu),
                               axis=0) / np.var(M, axis=0)
    return auto_corr


########################################
#                PLOT                  #
########################################


def plot(points, x_axis=0, y_axis=1, kind="scatter",
         n_points=400, burnin=150):
    """
    { “scatter” | “kde” | “hex” }
    """
    sns.set()
    x = points[burnin:(burnin+n_points), x_axis]
    y = points[burnin:(burnin+n_points), y_axis]
    sns.jointplot(x, y, kind=kind)
    plt.show()
    plt.figure(figsize=(9, 9), linewidth=1)
    plt.plot(x[:n_points], y[:n_points])
    plt.show()


def plot_autocorrelation(points, axis_list, kappa=100, legend=False):
    auto_corr = auto_correlation(points, kappa)
    for i in axis_list:
        plt.plot(auto_corr[1:, i], label=f"Axis : {i}")
    plt.title("Autocorrelation of the time serie (MCMC generated samples)")
    plt.show()


if __name__ == '__main__':
    """
    # - GAUSSIAN DISTRIBUTION EXAMPLE
    MEAN = torch.zeros(2)
    COV = torch.tensor([[1, .9], [.9, 1]])
    GAUSSIAN_DENSITY = gaussian_torch(MEAN, COV)
    GAUSSIAN_LOG_GRADIENT = log_gradient_gaussian(MEAN, COV)
    GAUSSIAN = MODEL(density=GAUSSIAN_DENSITY,
                     log_density_gradient=GAUSSIAN_LOG_GRADIENT)
    INIT_POINT = 2 * torch.randn_like(MEAN) + 3

    # -- ULA - GAUSSIAN
    nbr_samples = 50000
    ULA_GAUSSIAN = ULA(GAUSSIAN, torch.eye(2), 5e-3, 4e-1, INIT_POINT)
    points = ULA_GAUSSIAN.fit(nbr_samples)
    points = np.array(points)
    # plot(points, burnin=0, n_points=nbr_samples-1)
    # plot(points, burnin=200, n_points=nbr_samples-201)
    plot_autocorrelation(points, [0, 1])

    # -- MALA - GAUSSIAN
    nbr_samples = 50000
    MALA_GAUSSIAN = MALA(GAUSSIAN, torch.eye(2), 2e-2, 4e-1, INIT_POINT)
    points = MALA_GAUSSIAN.fit(nbr_samples)
    points = np.array(points)
    # plot(points, burnin=0, n_points=nbr_samples-1)
    auto_corr = auto_correlation(points, 100)
    plot(points, burnin=200, n_points=nbr_samples-201, kind='kde')
    plot_autocorrelation(points, [0, 1])
"""

    # -- MALA with adjusted acceptation ratio - GAUSSIAN
    # nbr_samples = 10000
    # MALA_GAUSSIAN = MALA(GAUSSIAN, torch.eye(2), 2e-2, 1e-2, INIT_POINT)
    # points = MALA_GAUSSIAN.adjusted_fit(nbr_samples)
    # points = np.array(points)
    # plot(points, burnin=0, n_points=nbr_samples-1)
    # plot(points, burnin=200, n_points=nbr_samples-201)

    # - GAUSSIAN DISTRIBUTION EXAMPLE
    MEAN = torch.zeros(2)
    COV = torch.tensor([[1, .9], [.9, 1]])
    GAUSSIAN_DENSITY = gaussian_torch(MEAN, COV)
    GAUSSIAN = MODEL(density=GAUSSIAN_DENSITY)
    INIT_POINT = 2 * torch.randn_like(MEAN) + 3

    # -- MALA - GAUSSIAN
    nbr_samples = 50000
    MALA_GAUSSIAN = MALA(GAUSSIAN, torch.eye(2), 2e-2, 4e-1, INIT_POINT)
    points = MALA_GAUSSIAN.fit(nbr_samples)
    points = np.array(points)
    # plot(points, burnin=0, n_points=nbr_samples-1)
    auto_corr = auto_correlation(points, 100)
    plot(points, burnin=200, n_points=nbr_samples-201, kind='kde')
    plot_autocorrelation(points, [0, 1])
