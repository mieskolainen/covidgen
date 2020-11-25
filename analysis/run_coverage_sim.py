# Binomial proportion uncertainty estimator
# coverage & interval width simulation
#
# m.mieskolainen@imperial.ac.uk, 2020

import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import multiprocessing
import matplotlib.ticker as ticker

# Import local path
import sys
sys.path.append('./analysis')
sys.path.append('./covidgen')


from estimators import *
from aux import *

# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)


def coversim(n, pval, MC=10000):
    """ Coverage simulation loop.
    """

    print(f'coversim: n = {n}')
    percent = 100
    
    C  = {
        'S'  : np.zeros(len(pval)),
        'W'  : np.zeros(len(pval)),
        'CP' : np.zeros(len(pval)),
        'BJ' : np.zeros(len(pval)),
        'LR' : np.zeros(len(pval)),
    }
    D  = {
        'S'  : np.zeros((len(pval),MC)),
        'W'  : np.zeros((len(pval),MC)),
        'CP' : np.zeros((len(pval),MC)),
        'BJ' : np.zeros((len(pval),MC)),
        'LR' : np.zeros((len(pval),MC)),
    }
    C_L = copy.deepcopy(D)
    C_U = copy.deepcopy(D)


    # Loop over the binomial parameter p values
    for i in tqdm(range(len(pval))):

        # Draw a random binomial numbers
        kval = np.random.binomial(n=n, p=pval[i], size=MC)

        # Get interval estimates
        for j in range(len(kval)):

            k = kval[j]

            # Standard normal approximation
            C95 = binom_err(k=k, n=n, z=z95)
            C_L['S'][i,j] = C95[0]
            C_U['S'][i,j] = C95[1]
            if (C95[0] < pval[i]) & (pval[i] < C95[1]):
                C['S'][i] += 1

            # Wilson score
            C95 = wilson_err(k=k, n=n, z=z95)
            C_L['W'][i,j] = C95[0]
            C_U['W'][i,j] = C95[1]
            if (C95[0] < pval[i]) & (pval[i] < C95[1]):
                C['W'][i] += 1

            # Clopper-Pearson
            C95 = clopper_pearson_err(k=k, n=n, CL=q95)
            C_L['CP'][i,j] = C95[0]
            C_U['CP'][i,j] = C95[1]
            if (C95[0] < pval[i]) & (pval[i] < C95[1]):
                C['CP'][i] += 1

            # Bayesian
            C95 = bayes_binom_err(k=k, n=n, prior='Jeffrey', CL=q95)
            C_L['BJ'][i,j] = C95[0]
            C_U['BJ'][i,j] = C95[1]
            if (C95[0] < pval[i]) & (pval[i] < C95[1]):
                C['BJ'][i] += 1

            # Likelihood ratio
            C95 = llr_binom_err(k=k, n=n, alpha=0.05)
            C_L['LR'][i,j] = C95[0]
            C_U['LR'][i,j] = C95[1]
            if (C95[0] < pval[i]) & (pval[i] < C95[1]):
                C['LR'][i] += 1
            

    os.makedirs('./figs/coverage/', exist_ok = True)

    # --------------------------------------------------------------------
    # Coverage

    fig,ax = plt.subplots(1,1,figsize=aux.set_fig_size())
    
    plt.plot(pval*percent, C['S']  / MC, label='Normal (Wald)')
    plt.plot(pval*percent, C['W']  / MC, label='Wilson')
    plt.plot(pval*percent, C['CP'] / MC, label='Clopper-Pearson')
    plt.plot(pval*percent, C['BJ'] / MC, label='Bayesian & $J$-prior')
    plt.plot(pval*percent, C['LR'] / MC, label='Likelihood Ratio')

    # Dotted line
    plt.plot(np.array([0, 1])*percent, np.ones(2)*0.95, linestyle='dotted', color=(0,0,0))

    plt.legend()
    plt.title(f'$n = {n}$')
    plt.xlabel(f'$p \\times 100$ [%]')
    plt.ylabel(f'Coverage probability $C(p,n)$')
    plt.xscale('log')
    ax.set(xlim=(np.min(pval)*percent, np.max(pval)*percent), ylim=(0.75, 1.0))

    # Set ticking
    #tick_spacing = 0.5
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    plt.savefig(f'./figs/coverage/sim_coverage_n_{n}.pdf', bbox_inches='tight')

    # --------------------------------------------------------------------
    # Interval

    fig,ax = plt.subplots(1,1,figsize=aux.set_fig_size())

    # Solid unit line
    plt.plot(np.array([0, 1])*percent, np.ones(2), linestyle='solid', color=(0.2,0.2,0.2))

    plt.plot(pval*percent, np.mean(C_L['S'],  axis=1)/pval, label='Normal (Wald)')
    plt.plot(pval*percent, np.mean(C_L['W'],  axis=1)/pval, label='Wilson')
    plt.plot(pval*percent, np.mean(C_L['CP'], axis=1)/pval, label='Clopper-Pearson')
    plt.plot(pval*percent, np.mean(C_L['BJ'], axis=1)/pval, label='Bayesian & $J$-prior')
    plt.plot(pval*percent, np.mean(C_L['LR'], axis=1)/pval, label='Likelihood Ratio')


    # Reset color cycles
    plt.gca().set_prop_cycle(None)

    plt.plot(pval*percent, np.mean(C_U['S'],  axis=1)/pval)
    plt.plot(pval*percent, np.mean(C_U['W'],  axis=1)/pval)
    plt.plot(pval*percent, np.mean(C_U['CP'], axis=1)/pval)
    plt.plot(pval*percent, np.mean(C_U['BJ'], axis=1)/pval)
    plt.plot(pval*percent, np.mean(C_U['LR'], axis=1)/pval)

    plt.legend()
    plt.title(f'$n = {n}$')
    plt.xlabel(f'$p \\times 100$ [%]')
    plt.ylabel(f'Relative interval $[C_{{2.5}}, C_{{97.5}}](p,n) / p$')
    plt.xscale('log')
    #plt.yscale('log')  
    
    ax.set(xlim=(np.min(pval)*percent, np.max(pval)*percent), ylim=(None, None))

    # Set ticking
    #tick_spacing = 0.5
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    plt.savefig(f'./figs/coverage/sim_width_n_{n}.pdf', bbox_inches='tight')

    return True


def my_wrapper(n):
    """ Wrapper for parallel (multiprocess) processing.
    """
    # Binomial p parameter values
    pval = np.logspace(np.log10(1E-4), np.log10(1E-1), 200)
    
    # Number of MC runs
    MC   = 10000
    
    return coversim(n=n, pval=pval, MC=MC)


def main():

    # Binomial sample sizes
    nval = [100, 1000, 10000]

    pool   = multiprocessing.Pool(multiprocessing.cpu_count())
    result = pool.map(my_wrapper, nval)
    print(result)


if __name__ == "__main__":
    main()
    print(__name__ + ' done!')


