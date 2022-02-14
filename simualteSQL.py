from ratestranslation import calibrate_scores
import numpy as np
import matplotlib.pyplot as plt
from time import time
from shulman import ShulmanProcess
import pyodbc


EffectiveDate = "'2021-09-30'"
connstr = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=actuarialdata.nwie.net,1021;DATABASE=NFActCorp;Trusted_Connection=yes;'
query = "EXEC [EconData].[Curve_NominalRates_Get] " + EffectiveDate

actual = []
with pyodbc.connect(connstr) as cnxn:

    with cnxn.cursor() as crs:
        crs.execute(query)
        for row in crs:
            print(row)
            actual.append(row)
    
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
y = np.array([0.0007, 0.0004, 0.0005, 0.0009, 0.0028, 0.0053, 0.0098, 0.0121, 0.0152, 0.0202, 0.0208])
y2 = np.array([actual[0][1]/100, actual[0][2]/100, actual[0][3]/100, actual[0][4]/100, actual[0][5]/100, 
               actual[0][6]/100, actual[0][7]/100, actual[0][8]/100,
               actual[0][9]/100, actual[0][10]/100, actual[0][11]/100])
test_state = calibrate_scores(y, t, p)
init_state = calibrate_scores(y2, t, p)
dt = 1 / 360
num_years = 40
num_scen = 25
nominalprocess = ShulmanProcess(t, p, init_state)
nominalrates = nominalprocess.simulatepath(num_years, dt, num_scen, -1)
numperiods = int(num_years / dt) + 1
timeindex = dt * np.array(range(0, numperiods))
BEY10 = [x.yieldcurve[8] for x in nominalrates]
BEY10 = np.reshape(BEY10, (-1, num_scen))
score1 = [x.scores[0] for x in nominalrates]
score1 = np.reshape(score1, (-1, num_scen))
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(timeindex, np.percentile(BEY10, 50, 1), label='50th')
# axs[0, 0].text(5, .01, f'Long Term Median is {np.percentile(BEY10[numperiods-1], 50):,.2%}')
axs[0, 0].set_title("BEY 10 Percentiles:  50th")
# axs[0, 0].legend()
axs[0, 1].plot(timeindex, np.percentile(BEY10, 1, 1), label='1st')
# axs[0, 1].text(5, .01, f'Long Term 1st is {np.percentile(BEY10[numperiods-1], 1):,.2%}')
axs[0, 1].set_title("BEY 10 Percentiles:  1st")
# axs[0, 1].legend()
axs[1, 0].hist(BEY10[numperiods-1], density=True)
axs[1, 1].hist(score1[numperiods-1], density=True)
plt.show()
et = time()
print()
print('Elapsed time was %3.2f seconds.' % (et - st))
