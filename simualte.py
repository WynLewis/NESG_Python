from ratestranslation import calibrate_scores
import numpy as np
import matplotlib.pyplot as plt
from time import time
from shulman import ShulmanProcess

st = time()
np.random.seed(12345)
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
num_years = 40
num_scen = 25
nominalprocess = ShulmanProcess(t, p, init_state)
nominalrates = nominalprocess.simulatepath(num_years, dt, num_scen, -1)
# BEY10 = [x.yieldcurve[9] for x in nominalrates]
# BEY10 = np.reshape(BEY10,(-1,num_scen))

et = time()
print()
print('Elapsed time was %3.2f seconds.' % (et - st))
