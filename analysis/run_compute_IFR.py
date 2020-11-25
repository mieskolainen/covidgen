# Read data, compute IFR probability distributions and produce pickle files.
#
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import matplotlib.pyplot as plt
import numba
import os
import pickle
import sys

sys.path.append('./analysis')
sys.path.append('./covidgen')
sys.path.append('./dataconfig')

import aux
import cstats
import estimators

# ** Generate delay kernels **
import run_generate_kernels

# ** Load datasets **
import load_datasets
CSETS,CSETS_sero = load_datasets.default()


# Font style
#import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)

# =================================================================
### Parameters

analysis_param = {

    # Time interval considered from files. Make sure is large
    # enough for all the datasets.
    'FIRST_DATE' : '2020-03-01',
    'LAST_DATE'  : '2020-07-15',

    # How many days later the death count is read out
    # -1 is the optimal decision (convolution inverse estimate)
    'deltas'     : [0, 7, 14, 21, -1],

    # Bayesian estimator quantiles
    'qbayes'     : estimators.q68_q95,

    # Maximum IFR tail value to consider
    'RMAX' : 0.04 # between [0,1], where 1 is 100%
}


# =================================================================
### Deconvolution/time-delay inverse extraction parameters

unfold_param = {
    'ZEROPAD' : 31,    # Convolution domain pre-extension, also the maximum $\Delta t$ considered
    'BS'      : 300,   # Number of bootstrap/toy MC samples
    'TSPAN'   : 200,   # Convolution kernel span (days), long tails
    'alpha'   : 0.15,  # Regularization strength
    'mode'    : 'C',   # Inversion input ('C' for reported cases, 'F' for fatalities)
    
    'q'       : estimators.q95,   # bootstrap custom quantiles [0,1]

    'data_poisson' : True, # Statistical uncertainty on
    'kernel_syst'  : True  # Systematic uncertainty on
}


# =================================================================

# Analyze data
obj = cstats.covid_analyze(CSETS=CSETS, CSETS_sero=CSETS_sero, \
                analysis_param=analysis_param, unfold_param=unfold_param)

## Save output
os.makedirs('./output/', exist_ok=True)
filename = './output/computed_counts.pkl'

with open(filename, 'wb') as f:
    pickle.dump([obj, analysis_param, unfold_param], f)
print(f'Output data saved to {filename}')


print(__name__ + ' done!')
