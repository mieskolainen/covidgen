# Toy test visualizations of epidemic time evolution & convolutions
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


# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)


# ------------------------------------------------------------------------
# Parameters

# Deconvolution demonstration
plot_deconv = True

# Logistic ODE parameters
i0         = 0.01
L          = 2000
beta       = 0.3
fixed_beta = False

# Running beta-function
beta_0      = 0.5
beta_D      = 14
beta_lambda = 15


# Infection fatality rate
IFR  = 0.4e-2


# Fast exponential impulse response parameters
a_exp = 15      # Exponential median is ln(2)/a

# Weibull parameters
a_wei = 16      # Weibull median is a*ln(2)^(1/k)
k_wei = 1.75

# Log-normal response parameter
LN_mu    = 2.8
LN_sigma = 0.5


# Group parameters
exp_param = {'a' : a_exp}
wei_param = {'a' : a_wei, 'k': k_wei}
lgn_param = {'mu': LN_mu, 'sigma': LN_sigma}


# ------------------------------------------------------------------------
### Time domain

# Time axis [days]
t     = np.arange(0,100)

# Fixed beta modelling
if fixed_beta:
    Idiff = functions.dIdt_log(t=t, i0=i0, beta=beta, L=L)
    Icum  = functions.I_log(t=t, i0=i0, beta=beta, L=L)

# Running beta modelling
else:
    Icum  = functions.I_log_running(t=t, i0=i0, L=L, beta=functions.betafunc, beta_param={'beta_0': beta_0, 'beta_D': beta_D, 'beta_lambda': beta_lambda})
    Idiff = functions.dIdt_log_running(t=t, i0=i0, L=L, beta=functions.betafunc, beta_param={'beta_0': beta_0, 'beta_D': beta_D, 'beta_lambda': beta_lambda})


exp_kernel = functions.h_exp(t=t, **exp_param)
C_exp = tools.convmatrix(exp_kernel)

wei_kernel = functions.h_wei(t=t, **wei_param)
C_wei = tools.convmatrix(wei_kernel)

lgn_kernel = functions.h_lgn(t=t, **lgn_param)
C_lgn = tools.convmatrix(lgn_kernel)


print(f'exp condition number: {np.linalg.cond(C_exp)}')
print(f'wei condition number: {np.linalg.cond(C_wei)}')
print(f'lgn condition number: {np.linalg.cond(C_lgn)}')


# Apply the continuum convolution equation
dFdt_exp = IFR * tools.convint_(t, Idiff, exp_kernel)
dFdt_wei = IFR * tools.convint_(t, Idiff, wei_kernel)
dFdt_lgn = IFR * tools.convint_(t, Idiff, lgn_kernel)


# Turn into Poisson sampled
'''
dFdt_exp = np.random.poisson(dFdt_exp)
dFdt_wei = np.random.poisson(dFdt_wei)
dFdt_lgn = np.random.poisson(dFdt_lgn)
'''

# Integrate to get the cumulative functions
F_exp = tools.cdfint(t, dFdt_exp)
F_lgn = tools.cdfint(t, dFdt_lgn)
F_wei = tools.cdfint(t, dFdt_wei)


# Deconvolution estimate
if plot_deconv:

    # Measurement
    y = copy.deepcopy(dFdt_wei)

    # Use Weibull measurement and the exponential kernel, for more realism
    #x_hat = tools.RL_deconv(y=y, kernel=wei_kernel, iterations=4, reg=1e-9)
    #x_hat = tools.FFT_deconv_naive(y=y, kernel=wei_kernel, alpha=1e-6)
    alpha = 0.015
    x_hat, rec_err, reg_err = tools.nneg_tikhonov_deconv(y=y, kernel=wei_kernel, alpha=alpha)
    
    '''
    fig,ax = plt.subplots(1,3)
    ax[0].imshow(C_exp)
    ax[1].imshow(C_wei)
    ax[2].imshow(C_lgn)
    '''
    #plt.show()
    
    # Integrate to get the cumulative functions
    X_hat = tools.cdfint(t, x_hat)


# ------------------------------------------------------------------------
### Plotting

fig,ax = plt.subplots(6,1,figsize=(6,10))

# 
ax[0].plot(t, Idiff, color=(0,0,0.5))
ax[0].set_ylim([0,np.max(Idiff)*1.2])

#
ax[1].plot(t, Icum,  color=(0,0,0.5))
ax[1].set_ylim([0,np.max(Icum)*1.2])

#
ax[2].plot(t, functions.h_exp(t=t, **exp_param), linestyle='dotted', color=(0,0.5,0), label=f'Exp $\\lambda={a_exp:0.1f}$')
ax[2].plot(t, functions.h_lgn(t=t, **lgn_param), linestyle='solid',  color=(0,0.5,0), label=f'Log-normal $\\mu={LN_mu:0.1f}, \\sigma={LN_sigma:0.1f}$')
ax[2].plot(t, functions.h_wei(t=t, **wei_param), linestyle='dashed', color=(0,0.5,0), label=f'Weibull $\\lambda={a_wei:0.1f}, k={k_wei:0.1f}$')
ax[2].set_ylim([0, 0.1])

#
ax[3].plot(t, dFdt_exp, linestyle='dotted', color=(0,0,0))
ax[3].plot(t, dFdt_lgn, linestyle='solid',  color=(0,0,0))
ax[3].plot(t, dFdt_wei, linestyle='dashed', color=(0,0,0))

if plot_deconv:
    ax[3].plot(t, x_hat, linestyle='dashed',  color='tab:blue')


#
ax[4].plot(t, F_exp, linestyle='dotted', color=(0.5,0.5,0.5))
ax[4].plot(t, F_lgn, linestyle='solid',  color=(0,0,0))
ax[4].plot(t, F_wei, linestyle='dashed', color=(0,0,0))
ax[4].set_ylim([0,np.max(F_exp)*1.2])

if plot_deconv:
    ax[4].plot(t, X_hat, linestyle='dashed', color='tab:blue')


#
ax[5].plot(t, F_exp / Icum * 100, linestyle='dotted', color=(1,0,0))
ax[5].plot(t, F_lgn / Icum * 100, linestyle='solid',  color=(1,0,0))
ax[5].plot(t, F_wei / Icum * 100, linestyle='dashed', color=(1,0,0))

if plot_deconv:
    ax[5].plot(t, X_hat / Icum * 100, linestyle='dashed', color='tab:blue')


# Legend
title_color = (0.3,0.3,0.3)

ax[0].set_title('mean daily infections', fontsize=11, color=title_color)
ax[1].set_title('mean cumulative infections', fontsize=11, color=title_color)

ax[2].set_title('effective time-delay kernels', fontsize=11, color=title_color)

ax[3].set_title('mean daily deaths', fontsize=11, color=title_color)
ax[4].set_title('mean cumulative deaths', fontsize=11, color=title_color)

ax[5].set_title('fatality rate', fontsize=11, color=title_color)

fig.tight_layout(pad=0.9)

ax[2].legend(fontsize=9)

#ax[3].legend(fontsize=9, frameon=False)
#ax[4].legend(fontsize=9, frameon=False)

# Set axis label
for i in range(len(ax)):
    ax[i].set_xlim([0, t[-1]])

for i in range(len(ax)):
    ax[i].set_xticks(np.arange(t[0], t[-1]+1, 7))

#for i in range(1,len(ax)-1):
#    ax[i].set_xticklabels([])


ax[0].set_yticks(np.arange(0,125,25))
ax[1].set_yticks(np.arange(0,2500,500))
ax[2].set_yticks(np.arange(0,0.15,0.05))
ax[3].set_yticks(np.arange(0,0.5,0.1))
ax[4].set_yticks(np.arange(0,10,2))
ax[5].set_yticks(np.arange(0,0.6,0.1))

ax[0].set_ylabel('$dI/dt$')
ax[1].set_ylabel('$I(t)$')
ax[2].set_ylabel('$h(t)$')
ax[3].set_ylabel('$dF/dt$')
ax[4].set_ylabel('$F(t)$')
ax[5].set_ylabel('$F/I$ $\\times 100$ [%]')

for i in range(len(ax)):
    ax[i].set_ylim([0,None])

ax[0].set_ylim([0,100])
ax[1].set_ylim([0,2000])
ax[3].set_ylim([0,0.5])
ax[4].set_ylim([0,8])
ax[5].set_ylim([0,0.5])


plt.xlabel('$t$ [days]')
plt.plot()
#plt.show()

os.makedirs('./figs/epidemic', exist_ok = True)
plt.savefig('./figs/epidemic/time_domain_visuals.pdf', bbox_inches='tight')
plt.close()

print(__name__ + ' done!')
