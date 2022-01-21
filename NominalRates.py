from scipy import stats
import numpy as np
from scipy import optimize

ver = '6.4'
p = {}
with open("Parameters.txt") as f:
    for line in f:
        (key, val) = line.split(':')
        keys = eval(key)
        if keys[0] == ver:
            p[keys[2]] = float(val)

t = np.array([1 / 12, 1 / 4, 1 / 2, 1, 2, 3, 5, 7, 10, 20, 30])
y = np.matrix([
    [0.0006, 0.0006, 0.0019, 0.0039, 0.0073, 0.0097, 0.0126, 0.0144, 0.0152, 0.0194, 0.019]
])
sc = np.matrix([
    [-4.6879, -0.4661, 0.3347],
    [0, 0, 0]
])


def yieldcurve(scores, tenors):
    """

    :param scores: Matrix of Scores
    :param tenors: Array of Tenors
    :return: Matrix of yield curves for given tenors
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
    l = stats.lognorm.ppf(k, beta)

    schulman_mean = (2 * omega - 1) * np.exp(0.5 * beta ** 2)

    schulman_var = np.sqrt((2 * omega ** 2 - 2 * omega + 1) * np.exp(2 * beta ** 2) - (2 * omega - 1) ** 2 * np.exp(
        beta ** 2) - 2 * omega * (1 - omega))
    schulman_variate = ((omega * l) - (1 - omega) / l - schulman_mean) / schulman_var

    rates = mu + sigma * schulman_variate

    return rates


def calibrate_scores(curve, tenors):
    """

    :param curve:
    :param tenors:
    :return: Array of Three Scores
    """
    def targetfunc(target, tenorvec, scorevec):
        ESS = np.sum((target - yieldcurve(scorevec, tenorvec))**2)
        return ESS

    guess = np.matrix([[0,0,0]])
    scorevec = optimize.minimize(lambda s: targetfunc(curve, tenors, s), guess)
    return scorevec


test = calibrate_scores(y, t)
print(test)

