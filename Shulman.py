from scipy import special
from scipy import stats
from scipy import optimize
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from MultiFractal import MultiFractalProcess
from OrnsteinUhlenbeck import OrnsteinUhlenbeckProcess
from RatesTranslation import yieldcurve, calibrate_scores
from dataclasses import dataclass
from typing import List

@dataclass
class ShulmanPeriod:
    """
    """
    period: int
    scenario: int
    scores: np.array
    yieldcurve: np.array
    

class ShulmanProcess:

    # Load distribution parameters from class instantiation
    def __init__(self, tenors, initialcurve, p):
        # α  [0,inf] controls the  long-term persistence of initial conditions (tail weight)
        #           <1 is infinite variance, <2 is infinite mean
        # θ  [0, inf] controls the short-term mean-reversion of initial conditions
        # σ  [0, inf] controls process volatility level
        # γ  [-0.5, inf] controls relationship between memory length and volatility

        self.tenors = tenors
        self.icurve = initialcurve
        self.p = p
        self.SCORE1_Process = MultiFractalProcess(p['Score1_Alpha'], p['Score1_Theta'], p['Score1_Sigma'],
                                                  p['Score1_Gamma'])
        self.SCORE2_Process = OrnsteinUhlenbeckProcess(p['Score2_Eta'], p['Score2_Sigma'])
        self.SCORE3_Process = OrnsteinUhlenbeckProcess(p['Score3_Eta'], p['Score3_Sigma'])

    def simulatepath(self, t0, x0, t, dt, numscenarios=1, stacksize=-1) -> List[ShulmanPeriod]:
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

        if stacksize == -1:
            stacksize = np.int(self.SCORE1_Process.theta / (self.SCORE1_Process.alpha - 1) / dt)

        # load each scenario with random stack
        initialt, initialdx = self.SCORE1_Process.historicalstack(t0, x0, stacksize)
        for scenario in range(0, numscenarios):
            # set history-adjusted stack prior to simulation period
            # build up the implied experience trajectory based upon stack
            for i, d in enumerate(initialt):
                score1[:, scenario] = score1[:, scenario] + (d >= timeindex) * initialdx[i]

        score2[0] = x0[1]
        score3[0] = x0[2]
        output = []
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
                yc = yieldcurve(s1, s2, s3)
                
                output.append(ShulmanPeriod(
                    period = period,
                    scenario = scenario,
                    scores = np.array(1,2,3),
                    yieldcurve = yc
                    ))
            print()
        return output
