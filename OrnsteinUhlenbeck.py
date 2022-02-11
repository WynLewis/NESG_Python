from scipy import special
from scipy import stats
from scipy import optimize
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


class OrnsteinUhlenbeckProcess:

    # Load distribution parameters from class instantiation
    def __init__(self, eta, sigma):
        # α  [0,inf] controls the  long-term persistence of initial conditions (tail weight)
        #           <1 is infinite variance, <2 is infinite mean
        # θ  [0, inf] controls the short-term mean-reversion of initial conditions
        # σ  [0, inf] controls process volatility level
        # γ  [-0.5, inf] controls relationship between memory length and volatility

        self.eta = eta
        self.sigma = sigma
        self.kappa = -0.5 * (eta / sigma) ** 2

    def simulatepath(self, x0, t, dt, numscenarios=1):
        # You can just enter np.array([0]),np.array([X0]) if there is no history available

        # t          length of simulated path (years)
        # dt         timestep of simulated path
        # numscenarios          number of paths to generate
        # stacksize  number of elements in random stack

        # RETURN time index column vector, N simulated path column vectors as matrix

        numperiods = np.int(t / dt) + 1
        timeindex = dt * np.array(range(0, numperiods))
        xbuild = np.zeros((numperiods, numscenarios))
        xbuild[0] = x0
        for period in range(1, numperiods):
            xbuild[period] = xbuild[period-1] + self.kappa * xbuild[period-1] * dt + \
                             self.eta * stats.norm.rvs(size=numscenarios) * np.sqrt(dt)

        return timeindex, xbuild
