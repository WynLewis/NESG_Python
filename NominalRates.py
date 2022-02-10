from scipy import stats
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import sys
from scipy.optimize import minimize

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
    [0.0007, 0.0005, 0.005, 0.0008, 0.0030, 0.0057, 0.0097, 0.0127, 0.0160, 0.0210, 0.225]
])
sc = np.matrix([
    [-3.7542, 0.1974, 0.1291]

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
    def g(x):
        z = np.matrix([
            [x[0], x[1], x[2]]
        ])
        m = yieldcurve(z, tenors)
        ess = np.sum(np.square(curve - m)) * 1e6
        return ess

    guess = np.zeros([1,3])
    scorevec = minimize(g, guess)
    return scorevec


output = yieldcurve(sc, t)
model = output[0]
actual = y[0]
plt.plot(t, model)
# plt.plot(tenors, actual)
plt.show()
test = calibrate_scores(y, t)

for data in model:
    sys.stdout.write('{:9.2%}'.format(data))

print()
print(test)



