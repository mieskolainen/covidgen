# Seroreversion effect unit tests
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

# Original time domain
t0 = 0
t1 = 10000
t  = np.linspace(t0,t1, (t1-t0)*1)

# ------------------------------------------------------------------------
# Create synthetic cumulative input data
i0   = 1e-3
beta = 1
L    = 100

I    = functions.I_log(t, i0, beta, L)

# Daily counts by difference
dI   = np.diff(I, append=I[-1])

# ------------------------------------------------------------------------

# Seroconversion delay kernel
exp_param = {'a' : 1}
kernel_C = functions.h_exp(t=t, **exp_param)

# ** Normalize discretized kernel to sum to one
# => count conservation with discrete convolutions **
kernel  = copy.deepcopy(kernel_C);
kernel /= np.sum(kernel)

# Seroreversion delay kernel
exp_param_REV = {'a' : 130}

title_str = f'Half-life = {exp_param_REV["a"] * np.log(2):0.1f}'
print(title_str)
kernel_C_REV = functions.h_exp(t=t, **exp_param_REV)


# ** Normalize discretized kernel to sum to one
# => count conservation with discrete convolutions **
kernel_REV  = copy.deepcopy(kernel_C_REV);
kernel_REV /= np.sum(kernel_REV)


# Plot kernels
fig,ax = plt.subplots()
plt.plot(t, kernel_C)
plt.plot(t, kernel_C_REV)
#plt.show()

# ------------------------------------------------------------------------
# Discrete convolution


# Seroconversion counts
dI_conv     = tools.conv_(dI, kernel)

# Seroreversion decay of converted counts
dI_conv_REV = tools.conv_(dI_conv, kernel_REV)


# Cumulative sum
I_S      = tools.conv_(I, kernel)
I_RS     = tools.conv_(I_S, kernel_REV)

# Observed counts
I_tilde  = I_S - I_RS


# ------------------------------------------------------------------------
# Plots

XLIM   = 300
fig,ax = plt.subplots(3,1,figsize=(8,7))

# Daily
ax[0].plot(t, dI,          label='$dI(t)/dt$')
ax[0].plot(t, dI_conv,     label='$(K_S\\ast dI/dt)(t)$')
ax[0].plot(t, dI_conv_REV, label='$(K_R \\ast K_S\\ast dI/dt)(t)$')


ax[0].set_ylim([0,None])
ax[0].set_xlim([0,XLIM])

ax[0].set_ylabel('daily counts')
ax[0].set_title('$K_S$: seroconversion kernel, $K_R$: seroreversion kernel')
ax[0].legend()


# Cumulative
ax[1].plot(t, I,       label='$I(t)$')
ax[1].plot(t, I_S,     label='$I_S(t) = (K_S \\ast I)(t)$')
ax[1].plot(t, I_RS,    label='$I_{RS}(t) = (K_R \\ast K_S \\ast I)(t)$')
ax[1].plot(t, I_tilde, label='$\\tilde{I}_S = I_S-I_{RS}$', ls='--')

ax[1].set_ylim([0,None])
ax[1].set_xlim([0,XLIM])

ax[1].set_title(title_str, fontsize=10)
ax[1].set_ylabel('cumulative counts')
ax[1].legend(loc=1)


# Ratio
EPS = 1e-9
ax[2].plot(t, I_tilde / (I_S + EPS),       label='$\\tilde{I}_{S} / I_{S}$')

ax[2].set_ylim([0,None])
ax[2].set_xlim([0,XLIM])

ax[2].set_xlabel('$t$')
ax[2].set_ylabel('Ratio')
ax[2].legend()


#plt.show()

# Save
plotfolder = './figs/epidemic'
os.makedirs(f'{plotfolder}', exist_ok = True)
plt.savefig(f'{plotfolder}/seroreverse.pdf', bbox_inches='tight')

print(__name__ + f' plotting done under: {plotfolder}')


