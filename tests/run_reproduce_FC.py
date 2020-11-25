# Confidence interval Neyman belt construction.
#
# Reproduce here Feldman-Cousins paper Poisson scenario
# https://arxiv.org/pdf/physics/9711021.pdf
#
"""
\\documentclass[12pt]{article}
\\usepackage[margin=0.5in]{geometry}
\\begin{document}
...
\\end{document}
"""
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

from scipy.stats import chi2

# Import local path
import sys
sys.path.append('./covidgen')


from aux import *
import estimators as est
percent = 100

# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)

def print_table(tables, k, bval, alpha):
	"""
	Print tables
	"""
	print('')
	for name in tables.keys():

		# ----------------------------------
		print('\\begin{table}[ht!]')
		print('\\tiny')
		print('\\begin{center}')
		print('\\begin{tabular}{|c||',end="")
		for j in range(len(bval)):
			print('c|',end="")
		print('}')
		# ----------------------------------

		print('\\hline')
		print('$k \\backslash b$', end="")
		for j in range(len(bval)):
			print(f' & ${bval[j]:0.1f}$', end="")
		print(' \\\\')
		print('\\hline')

		for i in range(len(k)):
			print(f'${k[i]}$ ', end="")
			for j in range(len(bval)):
				print(f'& ${tables[name][i,0,j]:0.2f},{tables[name][i,1,j]:0.2f}$ ', end="")
			print(' \\\\')

		# ----------------------------------
		print('\\hline')
		print('\\end{tabular}')
		print('\\end{center}')
		print(f'\\caption{{{name} ordered, {(1-alpha)*100:0.0f}\\% CL}}')
		print('\\end{table}')
		print('')
		# ----------------------------------

# Measured values
k = np.arange(0,21)

# Known background mean
bval1 = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5])
#bval2 = np.arange(6,16)

alpha = 0.10 # CL 90%

# Number of MC events
MC = 100000

# Parameter discretization
mu0     = np.linspace(1e-5, np.max(k)+10, 2000)
stat    = 'poisson'
plot_on = True

for bval in [bval1]:#, bval2):

	tables = {
		'exact-LLR' : np.zeros((len(k), 2, len(bval))),
		'asymp-LLR' : np.zeros((len(k), 2, len(bval))),
		'cPDF'      : np.zeros((len(k), 2, len(bval))),	
	}

	j = 0
	for b in bval:

		## MC scan with exact quantile thresholds
		exact_prange, MCminval, MCmaxval, mu0, delta   = beltscan_binom_err(k=k,b=b, theta0=mu0,alpha=alpha, MC=MC, return_full=True, mode='exact-LLR', stat=stat)

		## MC scan, but use asymptotic chi2 threshold
		chi2_prange_A, MCminval_A, MCmaxval_A, mu0, _  = beltscan_binom_err(k=k,b=b, theta0=mu0,alpha=alpha, MC=MC, return_full=True, mode='asymp-LLR', stat=stat)

		## MC scan with central pdf ordering
		pdf_prange, MCminval_pdf, MCmaxval_pdf, mu0, _ = beltscan_binom_err(k=k,b=b, theta0=mu0,alpha=alpha, MC=MC, return_full=True, mode='cPDF',      stat=stat)

		tables['exact-LLR'][:,:,j] = exact_prange
		tables['asymp-LLR'][:,:,j] = chi2_prange_A
		tables['cPDF'][:,:,j]      = pdf_prange

		j += 1

		### Plotting
		if plot_on:

			fig,ax = plt.subplots(figsize=(5,5))

			## MC LLR [plot individual horizontal lines]
			for i in range(len(MCminval)):
			    plt.plot(np.array([MCminval[i], MCmaxval[i]]), np.array([mu0[i], mu0[i]]), color=(0.9,0.9,0.9))

			## exact LLR
			plt.plot(MCminval,     mu0, linestyle='solid', color=(0.9,0.9,0.9), alpha=1, label='LLR (MC)')
			plt.plot(MCmaxval,     mu0, linestyle='solid', color=(0.9,0.9,0.9), alpha=1)

			## chi2 LLR
			plt.plot(MCminval_A,   mu0, linestyle='solid', color=(0,0,0), alpha=1, label='LLR ($\\chi^2$)')
			plt.plot(MCmaxval_A,   mu0, linestyle='solid', color=(0,0,0), alpha=1)

			## central pdf
			plt.plot(MCminval_pdf, mu0, linestyle='solid', color='tab:blue', alpha=0.5, label='Central PDF')
			plt.plot(MCmaxval_pdf, mu0, linestyle='solid', color='tab:blue', alpha=0.5)


			### Confidence interval end points solid dots

			for i in range(len(k)):
				k_ = k[i]
				for p in [0,1]:
					mark  = 10 if p == 0 else 11      # Triangle arrows
					shift = -0.04 if p == 0 else 0.04 # Visualization correction

					plt.plot(k_, exact_prange[i,p]+shift,  marker=mark, color=(0.5,0.5,0.5))
					# chi2 LLR
					plt.plot(k_, chi2_prange_A[i,p]+shift, marker=mark, color=(0.2,0,0))
					# central pdf
					plt.plot(k_, pdf_prange[i,p]+shift,    marker=mark, color='tab:blue')

			### Observed vertical line
			#plt.plot(np.ones(2) * k, np.linspace(0,15,2), linestyle='dotted', color=(0,0,0))

			plt.plot()
			plt.xlim([0, 15])
			plt.ylim([0, 15])
			plt.grid()

			plt.xticks(np.linspace(0,15,16))
			plt.yticks(np.linspace(0,15,16))

			plt.ylabel('Poisson parameter $\\mu$')
			plt.xlabel('Observable $k$ [counts]')

			#ax.set_aspect('equal', 'box')
			plt.legend(loc=2)
			plt.title(f'background $b={b}$, {(1-alpha)*100:0.0f}% CL')

			#plt.show()
			os.makedirs('./figs/profile/FC_paper/', exist_ok = True)
			plt.savefig(f'./figs/profile/FC_paper/neyman_belt_poisson_FC_paper_b_{b:0.1f}.pdf', bbox_inches='tight')

	print_table(tables, k, bval, alpha)

print(__name__ + ' done!')
