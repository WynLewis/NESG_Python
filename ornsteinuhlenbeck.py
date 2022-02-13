from scipy import special
from scipy import stats
from scipy import optimize
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


class OrnsteinUhlenbeckProcess:

    # Load distribution parameters from class instantiation
    def __init__(self, eta, sigma):
        """

        :param eta:
        :param sigma:
        """

        self.eta = eta
        self.sigma = sigma
        self.kappa = -0.5 * (eta / sigma) ** 2

    def simulatepath(self, x0, t, dt, numscenarios=1):
        """

        :param x0:
        :param t:
        :param dt:
        :param numscenarios:
        :return:
        """

        numperiods = np.int(t / dt) + 1
        timeindex = dt * np.array(range(0, numperiods))
        xbuild = np.zeros((numperiods, numscenarios))
        xbuild[0] = x0
        for period in range(1, numperiods):
            xbuild[period] = xbuild[period-1] + self.kappa * xbuild[period-1] * dt + \
                             self.eta * stats.norm.rvs(size=numscenarios) * np.sqrt(dt)

        return timeindex, xbuild
