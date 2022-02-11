from RatesTranslation import yieldcurve, calibrate_scores
import numpy as np
import matplotlib.pyplot as plt
import sys
from MultiFractal import MultiFractalProcess
from time import time
from OrnsteinUhlenbeck import OrnsteinUhlenbeckProcess
from Shulman import ShulmanProcess
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
y = np.matrix([
    [0.0007, 0.0005, 0.0005, 0.0008, 0.0030, 0.0057, 0.0097, 0.0127, 0.0160, 0.0210, 0.0225]
])
sc = np.matrix([
    [-3.7542, 0.1974, 0.1291]

])

output = yieldcurve(sc, t, p)
model = output[0]
test = calibrate_scores(y, t, p)
print(test)
output2 = yieldcurve(test, t, p)
model2 = output2[0]
actual = y.A1
plt.plot(t, model)
plt.plot(t, model2)
plt.plot(t, actual)
plt.show()

for data in model:
    sys.stdout.write('{:9.2%}'.format(data))
print()
for data2 in model2:
    sys.stdout.write('{:9.2%}'.format(data2))
print()
for data3 in actual:
    sys.stdout.write('{:9.2%}'.format(data3))

# SCORE1_Process = MultiFractalProcess(p['Score1_Alpha'], p['Score1_Theta'], p['Score1_Sigma'], p['Score1_Gamma'])
# SCORE2_Process = OrnsteinUhlenbeckProcess(p['Score2_Eta'], p['Score2_Sigma'])
# SCORE3_Process = OrnsteinUhlenbeckProcess(p['Score3_Eta'], p['Score3_Sigma'])
dt = 1 / 360
numyears = 40
numscen = 25
# t, Score1 = SCORE1_Process.simulatepath(np.array([0]), test, 40, dt, numscen, -1)
# t, Score2 = SCORE2_Process.simulatepath(test[1], 40, dt, numscen)
# t, Score3 = SCORE3_Process.simulatepath(test[2], 40, dt, numscen)

Nominal_Process = ShulmanProcess(t, y, p)
t, Score1 = Nominal_Process.simulatepath(np.array([0]), test, numyears, dt, numscen, -1)


plt.plot(t, np.percentile(Score1, 50, 1))
plt.plot(t, np.percentile(Score2, 50, 1))
plt.plot(t, np.percentile(Score3, 50, 1))
plt.show()
epochs, sims = np.shape(Score1)
SCORES = np.concatenate((Score1.reshape(epochs * sims,1), Score2.reshape(epochs * sims,1), Score3.reshape(epochs * sims,1)), axis = 1)
Rates = yieldcurve(SCORES, t, p)
et = time()
print()
print('Elapsed time was %3.2f seconds.' % (et - st))