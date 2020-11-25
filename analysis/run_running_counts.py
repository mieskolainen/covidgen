# Binomial confidence interval estimator (single and double ratio)
# comparison with running counts
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os

# Import local path
import sys
sys.path.append('./analysis')
sys.path.append('./covidgen')

from estimators import *
from aux import *

# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)

# Set print precision
set_arr_format(3)


# Latex output
texfile = './tex/run_running_counts.tex'
OF      = open(texfile, 'w')

def dprint(text, end='\n'):
    """ Dual print to tex and stdout """
    print(text, end=end)
    OF.write(text + end)

# ------------------------------------------------------------------------
# INPUT data

# Definition:
# F/P ~ binomial pair (k1,n1)
# I/T ~ binomial pair (k2,n2)

P = 12597  # Number of people in the city
T = 919    # Number of people in the test sample
I = 138    # Number of infected in the test sample
F = 7      # Number of deaths in the city

scale = 1 / (I/T)

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Evaluate methods

N_bootstrap = 100000

names = ['Normal (Wald)',
         'Wilson score',
         'Likelihood Ratio',
         'Clopper-Pearson',
         'Katz log',
         'Newcombe sinh$^{-1}$',
         'Conditional-mid-$P$',
         'Conditional-CP',
         'Profile Likelihood Ratio',
         'Ratio Percentile Bootstrap',
         'Ratio BCA Bootstrap',
         '2D-Bayesian & $J$-prior']

# Plotting styles
linestyle = ['dashed', 'dashed', 'dashed', 'dashed', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid', 'solid']

if len(names) != len(linestyle):
    raise Exception(__name__ + ': len(names) != len(linestyle)')

percent   = 100
ALL       = []

# Loop over range of F values
FVAL      = np.arange(2,12)
IFR       = np.zeros(FVAL.size)

N_METHODS = len(names)
CL68      = np.zeros((N_METHODS, len(FVAL), 2))
CL95      = np.zeros((N_METHODS, len(FVAL), 2))

BINS      = 1000
binedges  = np.linspace(0, 0.05, BINS)
IFR_Z     = np.zeros((len(FVAL), BINS-1))


for k in tqdm(range(len(FVAL))):

    F      = FVAL[k]
    IFR[k] = (F/P) * scale * percent
    print(f'<< F = {F} >>')
    z = 0

    # --------------------------------------------------------------------
    # Single binomial methods

    # Wald test
    CL68[z,k,:] = binom_err(k=F, n=P, z=1.0) * scale * percent
    CL95[z,k,:] = binom_err(k=F, n=P, z=z95) * scale * percent
    z+=1

    # Wilson score
    CL68[z,k,:] = wilson_err(k=F, n=P, z=1.0) * scale * percent
    CL95[z,k,:] = wilson_err(k=F, n=P, z=z95) * scale * percent
    z+=1
    
    # Likelihood ratio test
    CL68[z,k,:] = llr_binom_err(k=F, n=P, alpha=1-0.68) * scale * percent
    CL95[z,k,:] = llr_binom_err(k=F, n=P, alpha=1-0.95) * scale * percent
    z+=1

    # Clopper-Pearson
    CL68[z,k,:] = clopper_pearson_err(k=F, n=P, CL=q68) * scale * percent
    CL95[z,k,:] = clopper_pearson_err(k=F, n=P, CL=q95) * scale * percent
    z+=1

    # --------------------------------------------------------------------
    # Double binomial methods
    
    # Katz et al.
    CL68[z,k,:] = katz_binomial_ratio_err(k1=F,n1=P, k2=I,n2=T, z=1.0) * percent
    CL95[z,k,:] = katz_binomial_ratio_err(k1=F,n1=P, k2=I,n2=T, z=z95) * percent
    z+=1
    
    # Newcombe et al.
    CL68[z,k,:] = newcombe_binomial_ratio_err(k1=F,n1=P, k2=I,n2=T, z=1.0) * percent
    CL95[z,k,:] = newcombe_binomial_ratio_err(k1=F,n1=P, k2=I,n2=T, z=z95) * percent
    z+=1
    
    # Conditional ratio with Lancaster mid-P
    CL68[z,k,:] = binom_ratio_cond_err(k1=F,n1=P, k2=I,n2=T, CL=q68, method='mid-P') * percent
    CL95[z,k,:] = binom_ratio_cond_err(k1=F,n1=P, k2=I,n2=T, CL=q95, method='mid-P') * percent
    z+=1
    
    # Conditional ratio with Clopper-Pearson
    CL68[z,k,:] = binom_ratio_cond_err(k1=F,n1=P, k2=I,n2=T, CL=q68, method='CP') * percent
    CL95[z,k,:] = binom_ratio_cond_err(k1=F,n1=P, k2=I,n2=T, CL=q95, method='CP') * percent
    z+=1

    # Profile likelihood ratio
    CL68[z,k,:] = profile_LLR_binom_ratio_err(k1=F,n1=P, k2=I,n2=T, alpha=1-0.68) * percent
    CL95[z,k,:] = profile_LLR_binom_ratio_err(k1=F,n1=P, k2=I,n2=T, alpha=1-0.95) * percent
    z+=1

    # Percentile bootstrap
    CI          = bootstrap_binom_ratio_err(k1=F,n1=P, k2=I,n2=T, B=N_bootstrap, CL=[0.025, 0.16, 0.84, 0.975], type='percentile') * percent
    CL68[z,k,:] = [CI[1], CI[2]]
    CL95[z,k,:] = [CI[0], CI[3]]
    z+=1

    # BCA bootstrap
    CI          = binom_ratio_bca_bootstrap_err(k1=F,n1=P, k2=I,n2=T, B=N_bootstrap, CL=[0.025, 0.16, 0.84, 0.975]) * percent
    CL68[z,k,:] = [CI[1], CI[2]]
    CL95[z,k,:] = [CI[0], CI[3]]
    z+=1

    # Bayesian % Jeffreys-prior
    output      = bayes_binomial_ratio_err(k1=F,n1=P, k2=I,n2=T, prior1='Jeffrey', prior2='Jeffrey', CL=q68)
    CL68[z,k,:] = output['CR_value'] * percent
    output      = bayes_binomial_ratio_err(k1=F,n1=P, k2=I,n2=T, prior1='Jeffrey', prior2='Jeffrey', CL=q95)
    CL95[z,k,:] = output['CR_value'] * percent
    z+=1


os.makedirs('./figs/comparison/', exist_ok = True)


# ------------------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------------------

def setstyle():
    plt.xlabel('Fatal $F$ [counts]')
    plt.xticks(np.arange(0,25))
    #plt.grid()
    plt.xlim([0, np.max(FVAL)])
    plt.legend(names)


# ------------------------------------------------------------------------
### RELATIVE

fig = plt.figure()
for m in range(N_METHODS):
    plt.plot(FVAL, (CL95[m,:,1] - CL95[m,:,0]) / IFR, lw=2, linestyle=linestyle[m])

plt.ylabel('$\\Delta CI_{95}$ / $\\langle IFR \\rangle$')
plt.ylim([0, None])
setstyle()
plt.savefig('./figs/comparison/x_analytic_1.pdf', bbox_inches='tight')


# ------------------------------------------------------------------------
fig = plt.figure()
for m in range(N_METHODS):
    plt.plot(FVAL, (CL68[m,:,1] - CL68[m,:,0]) / IFR, lw=2, linestyle=linestyle[m])

plt.ylabel('$\\Delta CI_{68}$ / $\\langle IFR \\rangle$')
plt.ylim([0, None])
setstyle()
plt.savefig('./figs/comparison/x_analytic_2.pdf', bbox_inches='tight')


# ------------------------------------------------------------------------
### RELATIVE

fig = plt.figure()
for m in range(N_METHODS):
    plt.plot(FVAL, CL68[m,:,0] / IFR, lw=1.5, linestyle=linestyle[m])

# Reset color cycles
plt.gca().set_prop_cycle(None)

for m in range(N_METHODS):
    plt.plot(FVAL, CL68[m,:,1] / IFR, lw=1.5, linestyle=linestyle[m])

# Plot ones
plt.plot(FVAL, np.ones(len(FVAL)), lw=1.5, color=(0,0,0))

plt.ylabel('$[CL_{16}, CL_{84}]$ / $\\langle IFR \\rangle$')
plt.ylim([None, None])
setstyle()
plt.savefig('./figs/comparison/analytic_3.pdf', bbox_inches='tight')


# ------------------------------------------------------------------------
fig = plt.figure()
for m in range(N_METHODS):
    plt.plot(FVAL, CL95[m,:,0] / IFR, lw=1.5, linestyle=linestyle[m])

# Reset color cycles
plt.gca().set_prop_cycle(None)

for m in range(N_METHODS):
    plt.plot(FVAL, CL95[m,:,1] / IFR, lw=1.5, linestyle=linestyle[m])

# Plot ones
plt.plot(FVAL, np.ones(len(FVAL)), lw=1.5, color=(0,0,0))

plt.ylabel('$[CL_{2.5}, CL_{97.5}]$ / $\\langle IFR \\rangle$')
plt.ylim([None, None])
setstyle()
plt.savefig('./figs/comparison/analytic_4.pdf', bbox_inches='tight')


# ------------------------------------------------------------------------
### ABSOLUTE

fig = plt.figure()
for m in range(N_METHODS):
    plt.plot(FVAL, CL68[m,:,0], lw=1.5, linestyle=linestyle[m])

# Reset color cycles
plt.gca().set_prop_cycle(None)

for m in range(N_METHODS):
    plt.plot(FVAL, CL68[m,:,1], lw=1.5, linestyle=linestyle[m])

# Plot <IFR>
plt.plot(FVAL, IFR, lw=1.5, color=(0,0,0))

plt.ylabel('$[CL_{16}, \\langle IFR \\rangle, CL_{84}]$ [%]')
plt.ylim([None, None])
setstyle()
plt.savefig('./figs/comparison/analytic_5.pdf', bbox_inches='tight')

# ------------------------------------------------------------------------
fig = plt.figure()
for m in range(N_METHODS):
    plt.plot(FVAL, CL95[m,:,0], lw=1.5, linestyle=linestyle[m])

# Reset color cycles
plt.gca().set_prop_cycle(None)

for m in range(N_METHODS):
    plt.plot(FVAL, CL95[m,:,1], lw=1.5, linestyle=linestyle[m])

# Plot <IFR>
plt.plot(FVAL, IFR, lw=1.5, color=(0,0,0))

plt.ylabel('$[CL_{2.5}, \\langle IFR \\rangle, CL_{97.5}]$ [%]')
plt.ylim([None, None])
setstyle()
plt.savefig('./figs/comparison/analytic_6.pdf', bbox_inches='tight')


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# PRINT ALL

dprint(f'\n<< RESULTS >> \n')
dprint(f'')

for k in range(len(FVAL)):
    F = FVAL[k]
    dprint(f'F = {F}')
    dprint(f'Method & CI68 [%] & CI95 [%] \\\\')
    for m in range(N_METHODS):
        dprint(f'{names[m]:<30} & [{CL68[m,k,0]:.2f} {CL68[m,k,1]:.2f}] & [{CL95[m,k,0]:.2f} {CL95[m,k,1]:.2f}] \\\\')

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#print('\n\nERROR PROPAGATION TEST')

'''
# Evaluate the binomial fluctuations on I, and its effect on err[IFR], i.e. error on error
fhat  = I / T
err_f = np.sqrt(fhat*(1 - fhat) / T)
I_err = err_f * I

# Error propagation term: d [err_p] / dI =
dErr_pdI = ((3*F**2)/(2*I**4) - F/I**3)/np.sqrt(-(F*(F-I))/I**3)

# Error on Error via error propagation (1st order Taylor expansion)
err_err_p = np.sqrt(dErr_pdI**2 * I_err**2)

# Apply scale and 100 % to get error on FRI
print(f'Error propagation from dI to dIFR: err[err(IFR)] = {err_err_p * scale * percent:.4f} %')
print('\n')
'''

print(__name__ + ' done!')