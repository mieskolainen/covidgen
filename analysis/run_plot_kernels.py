# Plot time delay kernels
# and test ratio (non-linear) responses with different functions
#
# m.mieskolainen@imperial.ac.uk, 2020


import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import os
import copy

from matplotlib import cm
from tqdm import tqdm

sys.path.append('./analysis')
sys.path.append('./covidgen')

import tools
import functions
import aux
import cstats

# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)
figsize = (10,3.8) # two plots side-by-side


plotfolder = './figs/kernels'
os.makedirs(plotfolder, exist_ok = True)

# ========================================================================
### Load kernel pdfs

import run_generate_kernels

filename = './output/kernels_fine_max_100.pkl'
with open(filename, 'rb') as f:
	param = pickle.load(f)

t   = param['t']
K   = param['K']
W_a = param['W_a']
W_k = param['W_k']


# ========================================================================
# Plot functional responses

fig,ax = plt.subplots(1,2,figsize=figsize)


# Some synthetic test input functions
I = {}
I['I(t)=1']                     = np.ones(len(t))
I['I(t)=t^{1/2}']               = (t)**0.5
I['I(t)=t^2']                   = (t)**2
I['I(t)=1/(1+e^{-0.25(t-20)})'] = 1 / (1 + np.exp(-0.25*(t-20)))


# Loop over kernels
for key in I.keys():
	psi = cstats.covid_psi(I=I[key], K_F=K['F'], K_S=K['S'], delta=0)
	ax[0].plot(t, psi, label=f'${key}$')

ax[0].plot(t, np.ones(len(t)), color=(0.5,0.5,0.5), linestyle='dashed')
ax[0].set_ylabel('$\\psi(t)$')
ax[0].set_xlabel('$t$ [days]')

ax[0].set_xlim([0,100])
ax[0].set_ylim([0.4, 1.6])
ax[0].set_xticks(np.arange(0,110,10))
ax[0].legend(loc=1)


# Gamma function
k = 0
for key in ['I(t)=1', 'I(t)=1/(1+e^{-0.25(t-20)})']:
	if k == 0:
		color = 'tab:blue'
	if k == 1:
		color = 'tab:red'
	k += 1

	for delta in [7,14,28]:

		gamma = cstats.covid_gamma(I=I[key], K_F=K['F'], delta=delta)

		if delta == 7:
			ls = '-'
		if delta == 14:
			ls = '--'
		if delta == 28:
			ls = ':'

		ax[1].plot(t, gamma,  label=f'${key} \\; | \\; \\Delta t = {delta}$', linestyle=ls, color=color)

ax[1].set_ylabel('$\\gamma(t,\\Delta t)$')
ax[1].set_xlabel('$t$ [days]')

ax[1].set_xlim([0,60])
ax[1].set_ylim([1.0, 2.4])
ax[1].set_xticks(np.arange(0,110,10))
ax[1].legend(loc=1)


os.makedirs(f'{plotfolder}', exist_ok = True)
plt.savefig(f'{plotfolder}/psi_gamma_function.pdf', bbox_inches='tight')
#plt.show()


# ========================================================================
# Kernels in continuum time-domain

fig,ax = plt.subplots(1,2,figsize=figsize)

for key in K.keys():
    ls = '-' if len(key) > 1 else '--'
    symbol = ['\\lambda', 'k'] if len(key) > 1 else ['\\tilde{\\lambda}', '\\tilde{k}']
    ax[0].plot(t, K[key], linestyle=ls)
    label = str(key)
    label = label.replace('2', ' \\rightarrow ', 1)

    # Plot
    ax[1].plot(t, tools.cdfint(t, K[key]), linestyle=ls, label=f'$K_{{{label}}}$: ${symbol[0]} = {W_a[key]:0.1f}, {symbol[1]} = {W_k[key]:0.1f}$')

ax[0].set_xlabel('$t$ [days]')
ax[0].set_ylabel('$K(t)$')
ax[0].set_xlim([0,60])
ax[0].set_ylim([0,None])

ax[1].legend(loc=4)
ax[1].set_ylim([0,None])
ax[1].set_xlim([0,60])
ax[1].set_xlabel('$t$ [days]')


# Save
#fig.tight_layout(pad=0.9)
os.makedirs(f'{plotfolder}', exist_ok = True)
plt.savefig(f'{plotfolder}/kernels.pdf', bbox_inches='tight')
#plt.show()


# ========================================================================
# "Causal domains" visualized

qval = [1.35, 1.15, 0.95, 0.75, 0.5, 0.25]
k    = 0

for key in tqdm(I.keys()):

	fig,ax = plt.subplots()

	for q in qval:
		
		# Unit normalization of the kernel for discrete convolution
		kernelfunc = copy.deepcopy(K['F']);
		kernelfunc /= np.sum(kernelfunc)

		# Discrete convolution
		I_del      = tools.conv_(I[key], kernelfunc)

		print(f'max(I) = {np.max(I[key])}, max(I_del) = {np.max(I_del)}')

		y = tools.find_delay(t=t, F=I[key], Fd=I_del, rho=q)
		plt.plot(t, y, label=f'$\\epsilon = {q:0.2f}$', color=cm.RdBu(1-q/np.max(qval)))

	plt.legend(loc=1)
	plt.ylim([0,None])
	plt.xlim([0,100])
	plt.xticks(np.arange(0,110,10))
	plt.ylabel('$\\Delta t$ [days] | $\\frac{(K \\ast I)(t + \\Delta t)}{I(t)} = \\epsilon$')
	plt.xlabel('$t$ [days]')
	plt.title(f'${key}$')


	plt.savefig(f'{plotfolder}/cone_{k}.pdf', bbox_inches='tight')

	#plt.show()
	k += 1

print(__name__ + f' plotting done under <{plotfolder}>')
