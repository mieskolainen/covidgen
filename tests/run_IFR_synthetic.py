# Synthetic IFR delay visualization
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import os
import copy

sys.path.append('./covidgen')

import tools
import functions
import cstats
import aux

# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)
figsize = (10,3.8) # two plots side-by-side


plotfolder = './figs/epidemic'

# ========================================================================
### Load kernel pdfs
filename = './output/kernels_fine.pkl'
with open(filename, 'rb') as f:
	param = pickle.load(f)

t   = param['t']
K   = param['K']

# ========================================================================
# Capture ratios

# True IFR value
IFR_true = 0.5 * 1e-2

# Synthetic input
I = 1 / (1 + np.exp(-0.25*(t-20)))

# ** Unit normalized kernel for a discrete convolution **
KF_kernel = copy.deepcopy(K['F']);
KF_kernel /= np.sum(KF_kernel)

# Discrete convolution
F = IFR_true * tools.conv_(I, KF_kernel)

# -----------------------------------------

fig,ax = plt.subplots()
plt.plot(t, I, label='$I$')
plt.plot(t, F / IFR_true, label='$(I \\ast K_F)(t) / \\langle IFR \\rangle$')
plt.xlabel('$t$ [days]')
plt.title('diagnostics')
plt.ylim([0, 1.2])
plt.xlim([0, None])
plt.legend()

os.makedirs(f'{plotfolder}', exist_ok = True)
plt.savefig(f'{plotfolder}/conv_diagnostic.pdf', bbox_inches='tight')

# ----------------------------------------

fig,ax = plt.subplots(1,2,figsize=figsize)
EPS    = 1E-9


# Integer delta steps for delay between reading the seroprevalence and death counts
k = 0
for delta in [7, 28]:
	
	# Delayed read out
	F_delta_t = np.ones(len(F))
	F_delta_t[0:len(F[delta:])] = F[delta:]

	# ** Normalize to sum=1, we use discrete convolution **
	KS_kernel  = copy.deepcopy(K['S']);
	KS_kernel /= np.sum(KS_kernel)
	
	# Discrete convolution to obtain seroprevalence
	I_S   = tools.conv_(I, KS_kernel)

	# Get delay functions
	psi   = cstats.covid_psi(I=I, K_F=K['F'], K_S=K['S'])
	gamma = cstats.covid_gamma(I=I, K_F=K['F'], delta=delta)

	# Compute different functions
	IFR1  = F_delta_t / (I_S + EPS)
	IFR2  = F_delta_t / (psi * I_S + EPS)
	IFR3  = F_delta_t / (gamma * I_S + EPS)
	IFR4  = F_delta_t / (psi * gamma * I_S + EPS)

	percent = 100

	ax[k].plot(t, IFR1 / IFR_true, color=(0,0,0), ls=':',  label='$F(t+\\Delta t)  \\, / \\, I_S(t)$')
	ax[k].plot(t, IFR2 / IFR_true, color=(0,0,0), ls='--', label='$F(t+\\Delta t) \\, / \\, [\\psi(t) I_S(t)]$')
	ax[k].plot(t, IFR3 / IFR_true, color=(0,0,0), ls='-.', label='$F(t+\\Delta t) \\, / \\, [\\gamma(t,\\Delta t) I_S(t)]$')
	ax[k].plot(t, IFR4 / IFR_true, color=(0,0,0), ls='-',  label='$F(t+\\Delta t)  \\, / \\, [\\psi(t) \\gamma(t,\\Delta t) I_S(t)]$')


	ax[k].set_xlabel('$t$ [days]')
	if k == 0:
		ax[k].set_ylabel('Estimate / True (IFR)')
	ax[k].set_xlim([0, 95])
	ax[k].set_ylim([0.6, 2])
	ax[k].set_xticks(np.arange(0,100,10))
	ax[k].set_title(f'$\\Delta t = {delta}$')

	if k == 0:
		ax[k].legend(loc=1)

	k += 1

#plt.show()

#fig.tight_layout(pad=0.9)
os.makedirs(f'{plotfolder}', exist_ok = True)
plt.savefig(f'{plotfolder}/kernel_effects.pdf', bbox_inches='tight')
print(__name__ + f' done plots under {plotfolder}')
