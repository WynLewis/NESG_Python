from scipy import special
from scipy import stats
from scipy import optimize
from scipy import integrate
import numpy as np

import matplotlib.pyplot as plt

p = {}
with open("Parameters.txt") as f:
    for line in f:
        (key, val) = line.split(':')
        p[eval(key)] = float(val)

version = 6.4
tenor = [1/12, 1/4, 1/2, 1, 2, 3, 5, 7, 10, 20, 30]
score = np.matrix([-4.6879, -0.4661, 0.3347])
score = np.matrix([0, 0, 0])
rate = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
rateformat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(len(tenor)):
    t = tenor[i]
    omega = p[version, 'Nominal', 'Shulman_Omega_a'] + (p[version, 'Nominal', 'Shulman_Omega_b'] +
                                                        p[version, 'Nominal', 'Shulman_Omega_c'] * t) * np.exp(
        (-1 * p[version, 'Nominal', 'Shulman_Omega_d']) * t **
        p[version, 'Nominal', 'Shulman_Omega_e'])

    beta = p[version, 'Nominal', 'Shulman_Beta_a'] + (p[version, 'Nominal', 'Shulman_Beta_b'] +
                                                      p[version, 'Nominal', 'Shulman_Beta_c'] * t) * np.exp(
        (-1 * p[version, 'Nominal', 'Shulman_Beta_d']) * t **
        p[version, 'Nominal', 'Shulman_Beta_e'])

    mu = p[version, 'Nominal', 'Shulman_Mu_a'] + (p[version, 'Nominal', 'Shulman_Mu_b'] +
                                                  p[version, 'Nominal', 'Shulman_Mu_c'] * t) * np.exp(
        (-1 * p[version, 'Nominal', 'Shulman_Mu_d']) * t **
        p[version, 'Nominal', 'Shulman_Mu_e'])

    sigma = p[version, 'Nominal', 'Shulman_Sigma_a'] + (p[version, 'Nominal', 'Shulman_Sigma_b'] +
                                                        p[version, 'Nominal', 'Shulman_Sigma_c'] * t) * np.exp(
        (-1 * p[version, 'Nominal', 'Shulman_Sigma_d']) * t **
        p[version, 'Nominal', 'Shulman_Sigma_e'])

    factor1 = p[version, 'Nominal', 'Shulman_Factor1_a'] + (p[version, 'Nominal', 'Shulman_Factor1_b'] +
                                                            p[version, 'Nominal', 'Shulman_Factor1_c'] * t) * np.exp(
        (-1 * p[version, 'Nominal', 'Shulman_Factor1_d'])
        * t ** p[version, 'Nominal', 'Shulman_Factor1_e'])

    factor2 = p[version, 'Nominal', 'Shulman_Factor2_a'] + (p[version, 'Nominal', 'Shulman_Factor2_b'] +
                                                            p[version, 'Nominal', 'Shulman_Factor2_c'] * t) * np.exp(
        (-1 * p[version, 'Nominal', 'Shulman_Factor2_d']) * t **
        p[version, 'Nominal', 'Shulman_Factor2_e'])

    factor3 = p[version, 'Nominal', 'Shulman_Factor3_a'] + (p[version, 'Nominal', 'Shulman_Factor3_b'] +
                                                            p[version, 'Nominal', 'Shulman_Factor3_c'] * t) * np.exp(
        (-1 * p[version, 'Nominal', 'Shulman_Factor3_d']) * t **
        p[version, 'Nominal', 'Shulman_Factor3_e'])

    factor = np.matrix([factor1, factor2, factor3])
    K = stats.norm.cdf(score @ factor.T)
    L = stats.lognorm.ppf(K, beta)

    schulman_mean = (2 * omega - 1) * np.exp(0.5 * beta ** 2)

    schulman_var = np.sqrt((2 * omega ** 2 - 2 * omega + 1) * np.exp(2 * beta ** 2) - (2 * omega - 1) ** 2 * np.exp(
        beta ** 2) - 2 * omega * (1 - omega))
    schulman_variate = ((omega * L) - (1 - omega) / L - schulman_mean) / schulman_var

    rate = float(mu + sigma * schulman_variate)
    rateformat[i] = "{:.4f}".format(rate)


print(rateformat)