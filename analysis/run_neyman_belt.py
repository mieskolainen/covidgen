# Confidence interval Neyman belt construction
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
sys.path.append('./analysis')
sys.path.append('./covidgen')


from aux import *
import estimators as est
percent = 100

# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)

# Measurements
k1 = 7
n1 = 12597
k2 = 138
n2 = 919


# Number of MC events
MC = 1000000

# Parameter discretization
p0   = np.linspace(1e-5, (k1/n1)*3, 2000)


scale = 1 / (k2/n2)

for alpha in [0.32, 0.05]:

	print(f'alpha = {alpha:0.4f}')
	
	chi2_prange  = llr_binom_err(k=k1, n=n1, alpha=alpha)
	exact_prange = beltscan_binom_err(k=k1,n=n1, alpha=alpha, MC=MC, mode='exact-LLR', stat='binom')
	CP_prange    = clopper_pearson_err(k=k1, n=n1, CL=np.array([alpha/2, 1-alpha/2]))

	# Asymptotic (direct vertical scan)
	print(f'Asympt.  LLR:    {chi2_prange[0]*scale*percent:.3f} {chi2_prange[1]*scale*percent:.3f}')
	print(f'Exact MC LLR:    {exact_prange[0]*scale*percent:.3f} {exact_prange[1]*scale*percent:.3f}')
	print(f'Clopper-Pearson: {CP_prange[0]*scale*percent:.3f} {CP_prange[1]*scale*percent:.3f}')
	print('\n')
	

# ------------------------------------------------------------------------
### Neyman belt scans

alpha = 0.05

for stat in ['binom']:

	# MC scan with exact quantile thresholds
	exact_prange, MCminval, MCmaxval, p0, delta   = beltscan_binom_err(k=k1,n=n1, theta0=p0,alpha=alpha, MC=MC, return_full=True, mode='exact-LLR', stat=stat)
	print(f'Exact MC LLR:   {exact_prange[0]*scale*percent:.3f} {exact_prange[1]*scale*percent:.3f}')

	# MC scan, but use asymptotic chi2 threshold instead of exact quantile thresholds
	chi2_prange_A, MCminval_A, MCmaxval_A, p0, _  = beltscan_binom_err(k=k1,n=n1, theta0=p0,alpha=alpha, MC=MC, return_full=True, mode='asymp-LLR', stat=stat)
	print(f'chi2  MC LLR:   {chi2_prange_A[0]*scale*percent:.3f} {chi2_prange_A[1]*scale*percent:.3f}')

	# MC scan with central pdf ordering
	pdf_prange, MCminval_pdf, MCmaxval_pdf, p0, _ = beltscan_binom_err(k=k1,n=n1, theta0=p0,alpha=alpha, MC=MC, return_full=True, mode='cPDF',      stat=stat)
	print(f'cPDF  MC:       {pdf_prange[0]*scale*percent:.3f} {pdf_prange[1]*scale*percent:.3f}')


	# ------------------------------------------------------------------------
	### Plotting

	fig,ax = plt.subplots(figsize=(5,5))
	scale  = 1 / (k2/n2)
	hat    = (k1/n1) / (k2/n2) * percent # ML value


	## MC LLR [plot individual horizontal lines]
	for i in range(len(MCminval)):
	    plt.plot(np.array([MCminval[i], MCmaxval[i]]), np.array([p0[i], p0[i]]) * scale * percent, color=(0.9,0.9,0.9))

	## exact LLR
	plt.plot(MCminval,     p0 * scale * percent, linestyle='solid', color=(0.9,0.9,0.9), alpha=1, label='Single LLR (MC)')
	plt.plot(MCmaxval,     p0 * scale * percent, linestyle='solid', color=(0.9,0.9,0.9), alpha=1)

	## chi2 LLR
	plt.plot(MCminval_A,   p0 * scale * percent, linestyle='solid', color=(0,0,0), alpha=1, label='Single LLR ($\\chi^2$)')
	plt.plot(MCmaxval_A,   p0 * scale * percent, linestyle='solid', color=(0,0,0), alpha=1)

	## central pdf
	plt.plot(MCminval_pdf, p0 * scale * percent, linestyle='solid', color='tab:blue', alpha=0.5, label='Central PDF (CP)')
	plt.plot(MCmaxval_pdf, p0 * scale * percent, linestyle='solid', color='tab:blue', alpha=0.5)


	### Confidence interval with dashed lines
	# exact LLR
	plt.plot(np.array([0, k1]), np.ones(2)*exact_prange[0]* scale * percent, linestyle='dashed', color=(0.5,0.5,0.5))
	plt.plot(np.array([0, k1]), np.ones(2)*exact_prange[1]* scale * percent, linestyle='dashed', color=(0.5,0.5,0.5))

	# chi2 LLR
	plt.plot(np.array([0, k1]), np.ones(2)*chi2_prange[0] * scale * percent, linestyle='dashed', color=(0.2,0,0))
	plt.plot(np.array([0, k1]), np.ones(2)*chi2_prange[1] * scale * percent, linestyle='dashed', color=(0.2,0,0))

	# central pdf
	plt.plot(np.array([0, k1]), np.ones(2)*pdf_prange[0]  * scale * percent, linestyle='dashed', color='tab:blue')
	plt.plot(np.array([0, k1]), np.ones(2)*pdf_prange[1]  * scale * percent, linestyle='dashed', color='tab:blue')


	### Observed vertical line
	plt.plot(np.ones(2) * k1, np.linspace(0,1,2)*percent, linestyle='dotted', color=(0,0,0))


	plt.plot()
	plt.xlim([0, 15])
	plt.ylim([0, 1.0])

	plt.xticks(np.linspace(0,15,16))
	plt.yticks(np.linspace(0,1,11))

	plt.ylabel('Parameter $p_0 \\times (n_2/k_2) \\times 100$ [%]')
	plt.xlabel('Observable $k_1$ [counts]')

	#ax.set_aspect('equal', 'box')
	plt.legend(loc=4)
	
	#plt.show()
	os.makedirs('./figs/profile/', exist_ok = True)
	plt.savefig(f'./figs/profile/neyman_belt_{stat}.pdf', bbox_inches='tight')


print(__name__ + ' done!')
