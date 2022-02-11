from scipy import stats
import numpy as np
from scipy.optimize import minimize


def yieldcurve(scores, tenors, p):
    """

    :param p: Parameter Dictionary
    :param scores: Array of Scores
    :param tenors: Array of Tenors
    :return: Array of yield curves for given tenors
    """
    omega = (
            p['Shulman_Omega_a'] + (p['Shulman_Omega_b'] + p['Shulman_Omega_c'] * tenors)
            * np.exp((-1 * p['Shulman_Omega_d']) * tenors ** p['Shulman_Omega_e'])
        )

    beta = (
            p['Shulman_Beta_a'] + (p['Shulman_Beta_b'] + p['Shulman_Beta_c'] * tenors)
            * np.exp((-1 * p['Shulman_Beta_d']) * tenors ** p['Shulman_Beta_e'])
    )

    mu = (
            p['Shulman_Mu_a'] + (p['Shulman_Mu_b'] + p['Shulman_Mu_c'] * tenors)
            * np.exp((-1 * p['Shulman_Mu_d']) * tenors ** p['Shulman_Mu_e'])
    )

    sigma = (
            p['Shulman_Sigma_a'] + (p['Shulman_Sigma_b'] + p['Shulman_Sigma_c'] * tenors)
            * np.exp((-1 * p['Shulman_Sigma_d']) * tenors ** p['Shulman_Sigma_e'])
    )

    factor1 = (
            p['Shulman_Factor1_a'] + (p['Shulman_Factor1_b'] + p['Shulman_Factor1_c'] * tenors)
            * np.exp((-1 * p['Shulman_Factor1_d']) * tenors ** p['Shulman_Factor1_e'])
    )

    factor2 = (
            p['Shulman_Factor2_a'] + (p['Shulman_Factor2_b'] + p['Shulman_Factor2_c'] * tenors)
            * np.exp((-1 * p['Shulman_Factor2_d']) * tenors ** p['Shulman_Factor2_e'])
    )

    factor3 = (
            p['Shulman_Factor3_a'] + (p['Shulman_Factor3_b'] + p['Shulman_Factor3_c'] * tenors)
            * np.exp((-1 * p['Shulman_Factor3_d']) * tenors ** p['Shulman_Factor3_e'])
    )

    factor = np.matrix([factor1, factor2, factor3])
    k = stats.norm.cdf(scores @ factor)
    k = k.ravel()
    l = stats.lognorm.ppf(k, beta)

    schulman_mean = (2 * omega - 1) * np.exp(0.5 * beta ** 2)

    schulman_var = np.sqrt((2 * omega ** 2 - 2 * omega + 1) * np.exp(2 * beta ** 2) - (2 * omega - 1) ** 2 * np.exp(
        beta ** 2) - 2 * omega * (1 - omega))
    schulman_variate = ((omega * l) - (1 - omega) / l - schulman_mean) / schulman_var

    rates = mu + sigma * schulman_variate

    return rates


def calibrate_scores(curve, tenors, p):
    """

    :param p: parameter dictionary
    :param curve:
    :param tenors:
    :return: Array of Three Scores
    """
    def g(x):
        m = yieldcurve(x, tenors, p)
        ess = np.sum(np.square(curve - m)) * 1e6
        return ess

    guess = np.zeros([1, 3])
    scorevec = minimize(g, guess)
    return scorevec.x
