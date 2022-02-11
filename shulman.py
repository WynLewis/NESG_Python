from scipy import special
from scipy import stats
from scipy import optimize
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from multifractal import MultiFractalProcess
from ornsteinuhlenbeck import OrnsteinUhlenbeckProcess
from ratestranslation import yieldcurve, calibrate_scores
from dataclasses import dataclass
from typing import List


@dataclass
class ShulmanRates:
    """
    """
    scenario: int
    period: int
    scores: np.array
    tenors: np.array
    yieldcurve: np.array


class ShulmanProcess:

    # Load distribution parameters from class instantiation
    def __init__(self, tenors, p, state, time=np.array([0])):
        # α  [0,inf] controls the  long-term persistence of initial conditions (tail weight)
        #           <1 is infinite variance, <2 is infinite mean
        # θ  [0, inf] controls the short-term mean-reversion of initial conditions
        # σ  [0, inf] controls process volatility level
        # γ  [-0.5, inf] controls relationship between memory length and volatility

        self.tenors = tenors
        self.p = p
        self.time = time
        self.state = state
        self.SCORE1_Process = MultiFractalProcess(p['Score1_Alpha'], p['Score1_Theta'], p['Score1_Sigma'],
                                                  p['Score1_Gamma'])
        self.SCORE2_Process = OrnsteinUhlenbeckProcess(p['Score2_Eta'], p['Score2_Sigma'])
        self.SCORE3_Process = OrnsteinUhlenbeckProcess(p['Score3_Eta'], p['Score3_Sigma'])

    def simulatepath(self, t, dt, numscenarios=1, stacksize=-1) -> List[ShulmanRates]:
        # You can just enter np.array([0]),np.array([X0]) if there is no history available

        # t          length of simulated path (years)
        # dt         timestep of simulated path
        # numscenarios          number of paths to generate
        # stacksize  number of elements in random stack

        # RETURN time index column vector, N simulated path column vectors as matrix

        numperiods = np.int(t / dt) + 1
        timeindex = dt * np.array(range(0, numperiods))
        score1 = np.zeros((numperiods, numscenarios))
        score2 = np.zeros((numperiods, numscenarios))
        score3 = np.zeros((numperiods, numscenarios))
        output = []
        yc = yieldcurve(self.state, self.tenors, self.p)

        if stacksize == -1:
            stacksize = np.int(self.SCORE1_Process.theta / (self.SCORE1_Process.alpha - 1) / dt)

        # load each scenario with random stack
        initialt, initialdx = self.SCORE1_Process.historicalstack(self.time, self.state, stacksize)
        for scenario in range(0, numscenarios):
            # set history-adjusted stack prior to simulation period
            # build up the implied experience trajectory based upon stack
            for i, d in enumerate(initialt):
                score1[:, scenario] = score1[:, scenario] + (d >= timeindex) * initialdx[i]
            output.append(ShulmanRates(
                period=0,
                scenario=scenario + 1,
                scores=self.state,
                tenors=self.tenors,
                yieldcurve=yc
            ))
        score2[0] = self.state[1]
        score3[0] = self.state[2]

        for period in range(1, numperiods):
            d = stats.lomax.rvs(self.SCORE1_Process.alpha - 1, scale=self.SCORE1_Process.theta, size=numscenarios)
            v = self.SCORE1_Process.sigma * d ** self.SCORE1_Process.gamma  # set increment volatility
            dx = v * stats.norm.rvs(size=numscenarios) * np.sqrt(dt)  # draw normal increments
            score2[period] = score2[period - 1] + self.SCORE2_Process.kappa * score2[period - 1] * dt + \
                             self.SCORE2_Process.eta * stats.norm.rvs(size=numscenarios) * np.sqrt(dt)
            score3[period] = score3[period - 1] + self.SCORE3_Process.kappa * score3[period - 1] * dt + \
                             self.SCORE3_Process.eta * stats.norm.rvs(size=numscenarios) * np.sqrt(dt)
            # apply the simulated increments to build the process values
            for scenario in range(0, numscenarios):
                endperiod = min(numperiods,
                                period + np.int(d[scenario] / dt))  # determine the end point for the increment
                score1[period:endperiod, scenario] = score1[period:endperiod, scenario] + dx[
                    scenario]  # apply the increment
                yc = yieldcurve(np.array((score1[period, scenario], score2[period, scenario],
                                          score3[period, scenario])), self.tenors, self.p)

                output.append(ShulmanRates(
                    period=period,
                    scenario=scenario+1,
                    scores=np.array((score1[period, scenario], score2[period, scenario],
                                     score3[period, scenario])),
                    tenors = self.tenors,
                    yieldcurve=yc
                ))
        return output
