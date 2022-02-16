from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from multifractal import MultiFractalProcess
from ornsteinuhlenbeck import OrnsteinUhlenbeckProcess
from yieldcurve import ShulmanFactors, yieldcurve


@dataclass
class ShulmanRates:
    scenario: int
    period: int
    scores: np.array
    tenors: np.array
    yieldcurve: np.array


class ShulmanProcess:

    def __init__(self, tenors, p, state, time=np.array([0])):
        """
        Load distribution parameters from class instantiation.
        :param tenors:
        :param p:
        :param state:
        :param time:
        """

        self.tenors = tenors
        self.tenormap = {tenor: f'BEY{tenor:,g}' for tenor in self.tenors}
        self.p = p
        self.factors = ShulmanFactors(tenors, p)
        self.time = time
        self.state = state
        self.SCORE1_Process = MultiFractalProcess(p['Score1_Alpha'], p['Score1_Theta'], p['Score1_Sigma'],
                                                  p['Score1_Gamma'])
        self.SCORE2_Process = OrnsteinUhlenbeckProcess(p['Score2_Eta'], p['Score2_Sigma'])
        self.SCORE3_Process = OrnsteinUhlenbeckProcess(p['Score3_Eta'], p['Score3_Sigma'])

    def simulatepath(self, t, dt, numscenarios=1, stacksize=-1) -> List[ShulmanRates]:
        """

        :param t:
        :param dt:
        :param numscenarios:
        :param stacksize:
        :return:
        """
        numperiods = np.int(t / dt) + 1
        timeindex = dt * np.array(range(0, numperiods))
        score1 = np.zeros((numscenarios, numperiods))
        score2 = np.zeros((numscenarios, numperiods))
        score3 = np.zeros((numscenarios, numperiods))

        output = []
        yc = yieldcurve(self.state, self.factors)
        for scen in range(numscenarios):
            output.append([scen+1, 0, self.state[0], self.state[1], self.state[2], *yc])

        if stacksize == -1:
            stacksize = np.int(self.SCORE1_Process.theta / (self.SCORE1_Process.alpha - 1) / dt)

        initialt, initialdx = self.SCORE1_Process.historicalstack(self.time, self.state, stacksize)

        score1[:] =  np.sum(((np.tile(initialt.reshape(stacksize,1), numperiods)) >= 
                             (np.repeat(timeindex.reshape(numperiods,1).T,stacksize,axis=0))) * 
                            (np.tile((initialdx.reshape(stacksize,1)), numperiods)), axis=0)        
        score2[0] = self.state[1]
        score3[0] = self.state[2]
        d = stats.lomax.rvs(self.SCORE1_Process.alpha - 1, scale=self.SCORE1_Process.theta,
                            size=(numscenarios, numperiods))
        v = self.SCORE1_Process.sigma * d ** self.SCORE1_Process.gamma  # set increment volatility
        dx = v * stats.norm.rvs(size=(numscenarios, numperiods)) * np.sqrt(dt)  # draw normal increments
        for period in range(1, numperiods):
            score2[:, period] = score2[:, period - 1] + self.SCORE2_Process.kappa * score2[:, period - 1] * dt + \
                                self.SCORE2_Process.eta * stats.norm.rvs(size=numscenarios) * np.sqrt(dt)
            score3[:, period] = score3[:, period - 1] + self.SCORE3_Process.kappa * score3[:, period - 1] * dt + \
                                self.SCORE3_Process.eta * stats.norm.rvs(size=numscenarios) * np.sqrt(dt)
            endperiod = np.minimum(numperiods, period + (d[:, period] / dt)).astype(int)
            for scenario in range(0, numscenarios):
                score1[scenario, period:endperiod[scenario]] = score1[scenario, period:endperiod[scenario]] + dx[
                    scenario, period]  # apply the increment
            scores = np.vstack([
                score1[:, period],
                score2[:, period],
                score3[:, period]]).T
            ycs = yieldcurve(scores, self.factors)

            for ii, (curve, sc) in enumerate(zip(ycs, scores)):
                output.append([ii + 1, period, *sc, *curve])
                
        columns = "Scenario Period Score1 Score2 Score3".split()
        for tenor in self.tenors:
            columns.append(self.tenormap[tenor])
        df = pd.DataFrame(  
            columns=columns,
            data=output)
        return df
