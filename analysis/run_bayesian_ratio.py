# Bayesian ratio distribution analysis
# 
# m.mieskolainen@imperial.ac.uk, 2020


import numpy as np
import matplotlib.pyplot as plt
import numba
import sys
import os
from scipy.integrate import simps


sys.path.append('./analysis')
sys.path.append('./covidgen')


plotfolder = './figs/bayesian'


import estimators
import aux

# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)

percent = 100

def get_title(pdf_r, rval, CR_val):
	"""
	Latex output wrapper function.
	"""
	CR_val *= percent

	# Mean value integral
	IFR  = simps(y=pdf_r*rval, x=rval) * percent

	# Maximum a Posteriori estimate
	maxi = np.argmax(pdf_r)
	IFR_MAP = rval[maxi]*percent

	# Create title
	title = f'$\\mathrm{{IFR}}_{{MAP}} = {IFR_MAP:.2f}$, $\\langle \\mathrm{{IFR}} \\rangle = {IFR:.2f}$ % CR95: $[{CR_val[0]:.2f}, {CR_val[3]:.2f}]$ %'	
	#title = f'$\\mathrm{{IFR}}_{{MAP}} = {IFR_MAP:.2f}$, $\\langle \\mathrm{{IFR}} \\rangle = {IFR:.2f}$ % | CR68: $[{CR_val[1]:.2f}, {CR_val[2]:.2f}]$  CR95: $[{CR_val[0]:.2f}, {CR_val[3]:.2f}]$ %'
	return title

# ------------------------------------------------------------------------
# Input counts

k1 = 7
n1 = 12597
k2 = 138
n2 = 919

# ------------------------------------------------------------------------
### Posterior PDFs

CL_val         = estimators.q68_q95


# PRIOR 1
output_1 = estimators.bayes_binomial_ratio_err(k1=k1,n1=n1, k2=k2,n2=n2, \
	prior1='Jeffrey', prior2='Jeffrey', CL=CL_val)

rval           = output_1['val']
pdf_r_1        = output_1['pdf']
discrete_pdf_1 = output_1['discrete_pdf']
discrete_cdf_1 = output_1['discrete_cdf']
CR_val_1       = output_1['CR_value']
CR_ind_1       = output_1['CR_index']


# PRIOR 2
output_2 = estimators.bayes_binomial_ratio_err(k1=k1,n1=n1, k2=k2,n2=n2, \
	prior1='Flat',prior2='Flat', CL=CL_val)

rval           = output_2['val']
pdf_r_2        = output_2['pdf']
discrete_pdf_2 = output_2['discrete_pdf']
discrete_cdf_2 = output_2['discrete_cdf']
CR_val_2       = output_2['CR_value']
CR_ind_2       = output_2['CR_index']


### Plot posterior PDF and CDF
fig,ax1 = plt.subplots(1,1, figsize=aux.set_fig_size(width=450, aspect=0.8))
color = 'tab:red'


title1 = get_title(pdf_r_1, rval, CR_val_1)
title2 = get_title(pdf_r_2, rval, CR_val_2)

# ------------------------------------------------------------------------
ax1.plot(rval*percent, pdf_r_1, color=color, linestyle='solid',  label=f'$\\;J$-prior: {title1}')
ax1.plot(rval*percent, pdf_r_2, color=color, linestyle='dotted', label=f'$f$-prior: {title2}')
# ------------------------------------------------------------------------


ax1.set_ylabel('Posterior density $P(r \\, | \\, \\{k,n, \\alpha,\\beta\\}_{{i}})$', color=color)
ax1.set_xlabel('Infection Fatality Rate $(r \\times 100)$ [%]')
ax1.tick_params(axis='y', labelcolor=color)
#ax1.set_yscale('log')
#ax1.set_xscale('log')
ax1.set_xlim([0, None])
ax1.set_ylim([0, np.max([pdf_r_1, pdf_r_2]) * 1.25])
ax1.set_xlim([0, np.max(rval)*percent])

plt.legend()

ax2   = ax1.twinx()
color = 'tab:blue'

# ------------------------------------------------------------------------
ax2.plot(rval*percent, discrete_cdf_1, color=color, linestyle='solid')
ax2.plot(rval*percent, discrete_cdf_2, color=color, linestyle='dotted')
# ------------------------------------------------------------------------

ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0, 1.25])
ax2.set_xlim([0, 1.2])
ax2.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax2.set_ylabel('Cumulative posterior $F(r \\, | \\, \\{k,n, \\alpha,\\beta\\}_{{i}})$', color=color, rotation=270, labelpad=19)

os.makedirs(f'{plotfolder}', exist_ok = True)
plt.savefig(f'{plotfolder}/bayesian_ratio.pdf', bbox_inches='tight')

print(__name__ + f' plots done under <{plotfolder}>')
