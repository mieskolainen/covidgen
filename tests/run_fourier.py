# Fourier transforms of kernels test
#
# m.mieskolainen@imperial.ac.uk, 2020


import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import os
from matplotlib import cm
from tqdm import tqdm

sys.path.append('./covidgen')

import tools
import functions
import aux

# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)
figsize = (10, 3.8)	# Two plots side-by-side


t = np.linspace(0,100,101)


# Fast exponential impulse response parameters
a_exp = 0.1      # Exponential median is ln(2)/a

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

# -----------------------------------------------------------------

fig,ax = plt.subplots(1,3,figsize=(10,3.8))

kernel = {}

kernel['exp'] = functions.h_exp(t=t, **exp_param)
kernel['wei'] = functions.h_wei(t=t, **wei_param)
kernel['lgn'] = functions.h_lgn(t=t, **lgn_param)

ls = {}
ls['exp'] = '-'
ls['wei'] = '--'
ls['lgn'] = ':'


for key in kernel.keys():

	FFT = np.fft.fft(kernel[key])
	#FFT = 1 / FFT

	ax[0].plot(np.log10(np.abs(FFT)), label=key, color=(0,0,0), linestyle=ls[key])
	ax[1].plot(np.angle(FFT), label=key, color=(0,0,0), linestyle=ls[key])
	ax[2].plot(np.diff(np.angle(FFT)), label=key, color=(0,0,0), linestyle=ls[key])


ax[0].legend()	
ax[2].set_ylim([-0.01,0.01])
plt.show()

