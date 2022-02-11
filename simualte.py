from ratestranslation import calibrate_scores
import numpy as np
import matplotlib.pyplot as plt
from multifractal import MultiFractalProcess
from ornsteinuhlenbeck import OrnsteinUhlenbeckProcess
from time import time

from shulman import ShulmanProcess

st = time()
np.random.seed(12345)
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})
ver = '6.4'
p = {}
with open("Parameters.txt") as f:
    for line in f:
        (key, val) = line.split(':')
        keys = eval(key)
        if keys[0] == ver:
            p[keys[2]] = float(val)

t = np.array([1 / 12, 1 / 4, 1 / 2, 1, 2, 3, 5, 7, 10, 20, 30])
y = np.array([0.0007, 0.0005, 0.0005, 0.0008, 0.0030, 0.0057, 0.0097, 0.0127, 0.0160, 0.0210, 0.0225])
init_state = calibrate_scores(y, t, p)
dt = 1 / 360
num_years = 1
num_scen = 25
nominalprocess = ShulmanProcess(t, p, init_state)
nominalrates = nominalprocess.simulatepath(num_years, dt, num_scen, -1)
BEY10 = [x.yieldcurve[9] for x in nominalrates]

SCORE1_Process = MultiFractalProcess(p['Score1_Alpha'], p['Score1_Theta'], p['Score1_Sigma'], p['Score1_Gamma'])
SCORE2_Process = OrnsteinUhlenbeckProcess(p['Score2_Eta'], p['Score2_Sigma'])
SCORE3_Process = OrnsteinUhlenbeckProcess(p['Score3_Eta'], p['Score3_Sigma'])

t, Score1 = SCORE1_Process.simulatepath(np.array([0]), init_state, num_years, dt, num_scen, -1)
t, Score2 = SCORE2_Process.simulatepath(init_state[1], num_years, dt, num_scen)
t, Score3 = SCORE3_Process.simulatepath(init_state[2], num_years, dt, num_scen)

et = time()
print()
print('Elapsed time was %3.2f seconds.' % (et - st))
