# Convolution sum and integral unit tests
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import numba
import matplotlib.pyplot as plt
import scipy
import matplotlib
import os
import copy

matplotlib.rc('xtick', labelsize=6) 
matplotlib.rc('ytick', labelsize=6)

# Import local path
import sys
sys.path.append('./covidgen') 

import functions
import tools

# Time domain
t = np.linspace(0, 30, 1000)

# Delay kernel
exp_param = {'a' : 2.5}
kernel_C = functions.h_exp(t=t, **exp_param)

# ** Normalize discretized kernel to sum to one
# => count conservation with discrete convolutions **
kernel  = copy.deepcopy(kernel_C);
kernel /= np.sum(kernel)


# ------------------------------------------------------------------------
# Create synthetic cumulative input data
i0   = 1e-3
beta = 1
L    = 1000
Y    = functions.I_log(t, i0, beta, L)
# ------------------------------------------------------------------------

# Daily counts by difference
dy   = np.diff(Y, append=Y[-1])

# ------------------------------------------------------------------------
# 1. Discrete convolution
dy_conv    = tools.conv_(dy, kernel)

# 2. Continuum convolution via numerical integration
dy_conv_C  = tools.convint_(t, dy, kernel_C)

# 3. Continuum convolution with kernel function handle
dy_conv_CD = tools.convint(t, dy, functions.h_exp, exp_param)


# Cumulative sum
Y_conv  = np.cumsum(dy_conv)

# ------------------------------------------------------------------------
# Plots

fig,ax = plt.subplots(2,1)

# Daily
ax[0].plot(t, dy, label='dI(t)/dt')
ax[0].plot(t, dy_conv,    label='conv_')
ax[0].plot(t, dy_conv_C,  label='convint_', ls=':', lw=3)
ax[0].plot(t, dy_conv_CD, label='convint',  ls='--')


ax[0].set_ylim([0,None])
ax[0].set_ylabel('daily counts')
ax[0].set_title('discrete convolutions')
ax[0].legend()


# Cumulative
ax[1].plot(t, Y,      label='I(t)')
ax[1].plot(t, Y_conv, label='delayed')
ax[1].set_ylim([0,None])
ax[1].set_xlabel('$t$')
ax[1].set_ylabel('cumulative counts')
ax[1].legend()

#plt.show()

# Save
plotfolder = './figs/epidemic'
os.makedirs(f'{plotfolder}', exist_ok = True)
plt.savefig(f'{plotfolder}/conv_unit_tests.pdf', bbox_inches='tight')

print(__name__ + f' plotting done under: {plotfolder}')


