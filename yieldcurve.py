from typing import Dict
from scipy import stats
import numpy as np
from scipy.optimize import minimize

class ShulmanFactors:

	def __init__(self, tenors, p):
		self.tenors = tenors
		self.p = p

		self.omega:np.array = (
				self.p['Shulman_Omega_a'] + (self.p['Shulman_Omega_b'] + self.p['Shulman_Omega_c'] * self.tenors)
				* np.exp((-1 * self.p['Shulman_Omega_d']) * self.tenors ** self.p['Shulman_Omega_e'])
			)

		self.beta:np.array = (
				self.p['Shulman_Beta_a'] + (self.p['Shulman_Beta_b'] + self.p['Shulman_Beta_c'] * self.tenors)
				* np.exp((-1 * self.p['Shulman_Beta_d']) * self.tenors ** self.p['Shulman_Beta_e'])
		)

		self.mu:np.array = (
				self.p['Shulman_Mu_a'] + (self.p['Shulman_Mu_b'] + self.p['Shulman_Mu_c'] * self.tenors)
				* np.exp((-1 * self.p['Shulman_Mu_d']) * self.tenors ** self.p['Shulman_Mu_e'])
		)

		self.sigma:np.array = (
				self.p['Shulman_Sigma_a'] + (self.p['Shulman_Sigma_b'] + self.p['Shulman_Sigma_c'] * self.tenors)
				* np.exp((-1 * self.p['Shulman_Sigma_d']) * self.tenors ** self.p['Shulman_Sigma_e'])
		)

		factor1:np.array = (
				self.p['Shulman_Factor1_a'] + (self.p['Shulman_Factor1_b'] + self.p['Shulman_Factor1_c'] * self.tenors)
				* np.exp((-1 * self.p['Shulman_Factor1_d']) * self.tenors ** self.p['Shulman_Factor1_e'])
		)

		factor2:np.array = (
				self.p['Shulman_Factor2_a'] + (self.p['Shulman_Factor2_b'] + self.p['Shulman_Factor2_c'] * self.tenors)
				* np.exp((-1 * self.p['Shulman_Factor2_d']) * self.tenors ** self.p['Shulman_Factor2_e'])
		)

		factor3:np.array = (
				self.p['Shulman_Factor3_a'] + (self.p['Shulman_Factor3_b'] + self.p['Shulman_Factor3_c'] * self.tenors)
				* np.exp((-1 * self.p['Shulman_Factor3_d']) * self.tenors ** self.p['Shulman_Factor3_e'])
		)

		self.factors:np.array = np.array([factor1, factor2, factor3])


def yieldcurve(scores: np.array, factors: ShulmanFactors) -> np.array:
	"""
	:param p: Parameter Dictionary
	:param scores: Array of Scores
	:param tenors: Array of Tenors
	:return: Array of yield curves for given tenors
	"""
		
	k = stats.norm.cdf(scores @ factors.factors)
	#k = k.ravel()
	l = stats.lognorm.ppf(k, factors.beta)

	schulman_mean = (2 * factors.omega - 1) * np.exp(0.5 * factors.beta ** 2)

	schulman_var = np.sqrt(
			(2 * factors.omega ** 2 - 2 * factors.omega + 1) 
			* np.exp(2 * factors.beta ** 2) 
			- (2 * factors.omega - 1) ** 2 
			* np.exp(factors.beta ** 2) - 2 * factors.omega * (1 - factors.omega))

	schulman_variate = ((factors.omega * l) - (1 - factors.omega) / l - schulman_mean) / schulman_var

	rates = factors.mu + factors.sigma * schulman_variate

	return rates


def calibrate_scores(curve:np.array, factors: ShulmanFactors):
	"""
	:param p: parameter dictionary
	:param curve:
	:param tenors:
	:return: Array of Three Scores
	"""
	def g(x):
		m = yieldcurve(x, factors)
		ess = np.sum(np.square(curve - m)) * 1e6
		return ess

	guess = np.zeros([1, 3])
	scorevec = minimize(g, guess)
	return scorevec.x
