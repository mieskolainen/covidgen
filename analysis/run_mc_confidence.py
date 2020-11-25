# Bernoulli MC-simulation statistical confidence interval brute force
# construction based on "exact Neyman" belt scan by varying the
# number of deaths of the simulation input. Faster approximations
# could be based on other simulation driven inference techniques.
#
# Run first:
# >> python ./analysis/run_simloop.py
#
# m.mieskolainen@imperial.ac.uk, 2020


import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import os
from tqdm import tqdm

# Import local path
import sys
sys.path.append('./analysis')
sys.path.append('./covidgen')


from estimators import *
from aux import *
import observables

# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)


# Observed values

k1 = 7
n1 = 12597
k2 = 138
n2 = 919


# ------------------------------------------------------------------------
# Read in simulations

N        = 375 # Maximum number of simulations to be read in


CI68     = np.zeros((N,2))
CI95     = np.zeros((N,2))
percent  = 100
binedges = np.linspace(1e-6, 1.0, 300)
IFR      = np.zeros((N, len(binedges)-1))

yval = np.zeros(N)


### Load simulation
for i in tqdm(range(N)):
    try:
        B,B3,args = pickle.load(open(f'./output/F_step_{i}.covmc', 'rb'))
        
        # Read input parameter value
        yval[i]   = args.F_N

        ## Compute observables
        OBS = observables.get_observables(B=B, B3=B3)

        ## Get percentiles
        CI68[i,:]  = np.array([np.percentile(OBS['IFR_hat'], Q68[0]), np.percentile(OBS['IFR_hat'], Q68[1])]) * percent
        CI95[i,:]  = np.array([np.percentile(OBS['IFR_hat'], Q95[0]), np.percentile(OBS['IFR_hat'], Q95[1])]) * percent

        # Histogram slices
        y,bins     = np.histogram(OBS['IFR_hat'] * percent, bins=binedges)
        IFR[i,:]   = copy.deepcopy(y)

    except:
        print(f'Could not find simulation with index i = {i}')

# -------------------------------------------------------------
# Find confidence interval

# Observed
IFR_obs = (k1/n1)/(k2/n2)

# Confidence interval
ind_U = np.argmin(np.abs(CI95[:,0] - IFR_obs*percent))
ind_L = np.argmin(np.abs(CI95[:,1] - IFR_obs*percent))

Y = yval/(n1)/(k2/n2)
L = Y[ind_L]*percent
U = Y[ind_U]*percent


# -------------------------------------------------------------
# Plot

import matplotlib.cm as cmx
import matplotlib.colors as colors

fig,ax = plt.subplots()

# rasterized=True to remove white lines
plt.pcolormesh(bins, yval/(n1)/(k2/n2) * percent, IFR, cmap='PuBu_r', linewidth=0, rasterized=True)

# Neyman belt
plt.plot(CI95[:,0], Y*percent, color=(0,0,0))
plt.plot(CI95[:,1], Y*percent, color=(0,0,0))

labelstr = f'CI95: [{L:0.2f}, {U:0.2f}] %'

# Neyman belt crossing
plt.plot(np.array([0, IFR_obs])*percent, np.ones(2)*U, linestyle='dashed', color=(0.8,0,0))
plt.plot(np.array([0, IFR_obs])*percent, np.ones(2)*L, linestyle='dashed', color=(0.8,0,0))
plt.plot(IFR_obs*percent*np.ones(2), np.array([0,1]),  linestyle='dashed', color=(0.9,0.9,0.9), label=labelstr)

plt.legend()
plt.xlim([1e-2,0.8])
plt.ylim([1e-2,0.8])
plt.ylabel('Parameter ${r}_0$')
plt.xlabel('Observed $r$')

#plt.show()
plotfolder = './figs/sim/'
os.makedirs(f'{plotfolder}', exist_ok = True)
plt.savefig(f'{plotfolder}/mc_confidence.pdf', bbox_inches='tight')

print(__name__ + f' plots done under <{plotfolder}>')
