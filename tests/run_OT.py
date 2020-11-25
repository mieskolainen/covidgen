# Optimal Transport unit test comparison with classic (weighted) mean
#
# m.mieskolainen@imperial.ac.uk, 2020


import os
import numpy as np
import pickle
import pandas as pd
import copy
import matplotlib.pyplot as plt
from scipy.integrate import simps

# Import local path
import sys
sys.path.append('./covidgen')

import covidgen
import tools
import estimators
from aux import *

# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)

# Wasserstain barycenter
import ot

n = 500  # discretization

def gausspdf(x,mu,sigma):
	return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2 / (2*sigma**2))


# Number of individuals
K = 10

# True values
theta_values  = np.array([0, 3])
tau_values    = np.array([1, 2])

# Repetitions
R = 10


# Loop over scenarios
for theta in theta_values:
	for tau in tau_values:

		for rep in range(R):

			print(f'\n<< theta = {theta}, tau = {tau} >>')

			# Generate synthetic dataset
			sigma_j = 0.5 + tau*np.random.randn(K)**2
			theta_j = theta + sigma_j


			print(f'theta_j: {theta_j}')
			print(f'sigma_j: {sigma_j}')
			print('')

			# Weights
			w    = 1/sigma_j**2;
			w   /= np.sum(w)


			### Classic estimates
			mu   = np.mean(theta_j)
			mu_w = np.sum(w * theta_j)


			# Compute pdfs
			xval    = np.linspace(-30,30,n) # Need to span the full support of pdf!
			pdf     = np.zeros((len(xval),K))
			for i in range(K):
				pdf[:,i] = gausspdf(xval, theta_j[i], sigma_j[i])

			### Optimal Transport estimates
			wpdf    = estimators.frechet_mean(xval=xval, PDF=pdf)
			wpdf_W  = estimators.frechet_mean(xval=xval, PDF=pdf, w=w)
			OT_mu   = simps(x=xval, y=wpdf * xval)
			OT_mu_W = simps(x=xval, y=wpdf_W * xval)
			
			OT_std   = np.sqrt(simps(x=xval, y=(xval - OT_mu)**2 * wpdf))
			OT_W_std = np.sqrt(simps(x=xval, y=(xval - OT_mu_W)**2 * wpdf_W))			


			# Results
			print(f'classic mean      = {mu:0.6f}     ')
			print(f'OT mean:          = {OT_mu:0.6f}  ')

			print(f'classic weighted  = {mu_w:0.6f}   ')
			print(f'OT weighted       = {OT_mu_W:0.6f}')
		
		"""
		fig,ax  = plt.subplots()
		plt.plot(xval, pdf)
		plt.plot(xval, wpdf,   color='black', label='OT')
		plt.plot(xval, wpdf_W, color='black', ls='--', lw=2, label='$1/\\sigma_i^2$ OT')
		plt.legend()
		plt.show()
		"""

