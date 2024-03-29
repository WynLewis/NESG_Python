from scipy import special
from scipy import stats
from scipy import optimize
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


class MultiFractalProcess:

    def __init__(self, alpha: float, theta: float, sigma: float, gamma: float):
        """

        :param alpha: [0,inf] controls the  long-term persistence of initial conditions (tail weight)
             <1 is infinite variance, <2 is infinite mean
        :param theta: [0, inf] controls the short-term mean-reversion of initial conditions
        :param sigma: [0, inf] controls process volatility level
        :param gamma: [-0.5, inf] controls relationship between memory length and volatility
        """

        self.alpha = alpha
        self.theta = theta
        self.sigma = sigma
        self.gamma = gamma

    # Definition of recovery function (percentage mean-reversion curve)
    def recoverycurve(self, s):
        """

        :param s:
        :return:
        """
        return 1 - (self.theta / (s + self.theta)) ** (self.alpha - 1)

    def randomstack(self, x0, stacksize=1500):
        """

        :param x0:
        :param stacksize:
        :return:
        """

        n = self.theta / (self.alpha - 1)

        d = stats.lomax.rvs(self.alpha - 1, scale=self.theta, size=stacksize)
        t = stats.lomax.rvs(self.alpha, scale=(d + self.theta), size=stacksize)

        v = self.sigma * (d + t) ** self.gamma

        dx = v * stats.norm.rvs(size=stacksize) / stacksize
        dx = dx + (x0 - sum(dx)) / stacksize

        return t, dx



    def conditionalexpiry(self, dx, dt):
        """

        :param dx:
        :param dt:
        :return:
        """
        c = -.5 * (dx / self.sigma) ** 2 / dt
        m = optimize.brentq(lambda t: self.theta + 2 * C * t ** (2 * self.gamma) * (t + self.theta), 0, 500)

        # use rejection sampling to simulate conditional expiry time
        t = stats.lomax.rvs(self.alpha + self.gamma, scale=self.theta)
        while np.random.uniform(size=1) > ((t + self.theta) / t) ** self.gamma * np.exp(
                c / (t ** (2 * self.gamma))) / m:
            t = stats.lomax.rvs(self.alpha + self.gamma, scale=self.theta)

        return t

    # Create a random stack of state variables consistent with historical process trajectory
    def historicalstack(self, t, X, stacksize=1500):
        """

        :param t:
        :param X:
        :param stacksize:
        :return:
        """

        t = t - t[0]
        initialt, initialdx = self.randomstack(X[0], stacksize)  # set random stack prior to experience period

        # build up the implied experience trajectory based upon random stack
        xbuild = np.zeros(np.size(t))
        for i, d in enumerate(initialt):
            xbuild = xbuild + (d >= t) * initialdx[i]

        # apply historical trajectory to create adjusted stack
        dt = np.diff(t)
        historicalt = np.zeros(np.size(dt))  # historical stack elements
        historicaldx = np.zeros(np.size(dt))  # historical stack elements

        for i, d in enumerate(dt):
            historicaldx[i] = X[i + 1] - xbuild[i + 1]  # compute the implied increment
            if np.abs(historicaldx[i] ** 2 / d) > .0001:
                historicalt[i] = t[i + 1] + self.conditionalexpiry(historicaldx[i],
                                                                   d)  # simulate the conditional duration
                xbuild = xbuild + (historicalt[i] >= t) * historicaldx[
                    i]  # update the constructed stack to follow the historical path

        # build residual stack from initial and historical stack, removing expired increments
        dx = np.concatenate((initialdx[np.nonzero(initialt > t[-1])], historicaldx[np.nonzero(historicalt > t[-1])]))
        d = np.concatenate((initialt[np.nonzero(initialt > t[-1])], historicalt[np.nonzero(historicalt > t[-1])]))
        d = d - t[-1]  # recenter time index to end of experience period

        return d, dx

    def simulatepath(self,x0, t, dt, t0=np.array([0]), numscenarios=1, stacksize=-1):
        """

        :param t0: index of historical data points, in ascending sequence, last value = X0
        :param x0: value at each historical time index
        :param t: length of simulated path (years)
        :param dt: timestep of simulated path
        :param numscenarios: number of paths to generate
        :param stacksize: number of elements in random stack
        :return: time index column vector, N simulated path column vectors as matrix
        """
        # t time
        # X state
        # You can just enter np.array([0]),np.array([X0]) if there is no history available

        # t
        # dt
        # numscenarios
        # stacksize  number of elements in random stack

        # RETURN time index column vector, N simulated path column vectors as matrix

        if stacksize == -1:
            stacksize = np.int(self.theta / (self.alpha - 1) / dt)

        numperiods = np.int(t / dt) + 1
        timeindex = dt * np.array(range(0, numperiods))
        xbuild = np.zeros((numperiods, numscenarios))

        # load each scenario with random stack
        initialt, initialdx = self.historicalstack(t0, x0, stacksize)
        for scenario in range(0, numscenarios):
            # set history-adjusted stack prior to simulation period
            # build up the implied experience trajectory based upon stack
            for i, d in enumerate(initialt):
                xbuild[:, scenario] = xbuild[:, scenario] + (d >= timeindex) * initialdx[i]

        # simulate rest of path simultaneously for all scenarios
        for period in range(1, numperiods + 1):
            d = stats.lomax.rvs(self.alpha - 1, scale=self.theta, size=numscenarios)  # set increment duration
            v = self.sigma * d ** self.gamma  # set increment volatility
            dx = v * stats.norm.rvs(size=numscenarios) * np.sqrt(dt)  # draw normal increments

            # apply the simulated increments to build the process values
            for scenario in range(0, numscenarios):
                endperiod = min(numperiods,
                                period + np.int(d[scenario] / dt))  # determine the end point for the increment
                xbuild[period:endperiod, scenario] = xbuild[period:endperiod, scenario] + dx[
                    scenario]  # apply the increment

        return timeindex, xbuild
