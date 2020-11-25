# Plot Gamma pdfs versus Gaussian
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import matplotlib.pyplot as plt
import numba
import os
import sys
import copy

from scipy.integrate import simps

sys.path.append('./covidgen')
import aux
import tools
import functions
import estimators

# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)

mu    = 1.0
sigma = [0.1,0.3,0.5,0.8]

fig,ax = plt.subplots()
for s in sigma:

	# Estimate gamma pdf parameters
	gamma_k, gamma_theta = estimators.gamma_param_estimate(mu=mu, sigma=s)
	print(f'Gamma pdf param k={gamma_k:0.5f}, theta={gamma_theta:0.5f}')

	x          = np.linspace(np.max([0.0, mu-s*4]), mu+s*5, 50)
	pdf_gamma  = functions.gamma_pdf(x=x, k=gamma_k, theta=gamma_theta)
	pdf_normal = functions.normpdf(x=x, mu=mu, std=s)

	print(f'Gamma pdf integral = {simps(x=x, y=pdf_gamma)}')

	mu_real    = simps(x=x, y=x*pdf_gamma)
	std_real   = np.sqrt(simps(x=x, y=(x-mu_real)**2*pdf_gamma))

	plt.plot(x, pdf_gamma,  label=f'$\\Gamma(x)$: $\\mu={mu:0.2f}$, $\\sigma={s:0.2f} \\; [{mu_real:0.2f}, {std_real:0.2f}]$')
	plt.plot(x, pdf_normal, label=f'$N(x)$: $\\mu={mu:0.2f}$, $\\sigma={s:0.2f}$', ls='--')


plt.legend()
plt.xlabel('$x$')
plt.ylabel('$pdf(x)$')

os.makedirs('./figs/bayesian/', exist_ok = True)
plt.savefig('./figs/bayesian/gamma_vs_normal_prior.pdf', bbox_inches='tight')

print('done!')