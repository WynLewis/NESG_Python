SCORE1_Process = MultiFractalProcess(p['Score1_Alpha'], p['Score1_Theta'], p['Score1_Sigma'], p['Score1_Gamma'])
SCORE2_Process = OrnsteinUhlenbeckProcess(p['Score2_Eta'], p['Score2_Sigma'])
SCORE3_Process = OrnsteinUhlenbeckProcess(p['Score3_Eta'], p['Score3_Sigma'])

t, Score1 = SCORE1_Process.simulatepath(np.array([0]), init_state, num_years, dt, num_scen, -1)
t, Score2 = SCORE2_Process.simulatepath(init_state[1], num_years, dt, num_scen)
t, Score3 = SCORE3_Process.simulatepath(init_state[2], num_years, dt, num_scen)

from multifractal import MultiFractalProcess
from ornsteinuhlenbeck import OrnsteinUhlenbeckProcess