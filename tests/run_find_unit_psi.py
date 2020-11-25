# Find \Delta t (read-out delays) which give
# unit correction function (psi(t,\Delta t) = 1)
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import matplotlib.pyplot as plt
import numba
import os
import copy
import pickle
import sys
import scipy
from datetime import datetime, timedelta

from   matplotlib import cm
from   tqdm import tqdm


sys.path.append('./analysis')
sys.path.append('./covidgen')

# Generate delay kernels
import run_generate_kernels

import aux
import tools
import functions
import estimators
import cstats
import cio

# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)


# =================================================================
### Parameters

# Time interval
FIRST_DATE = '2020-03-01'
LAST_DATE  = '2020-06-15'

unfold_param = {
    'ZEROPAD' : 31,   # Convolution domain pre-extension, also the maximum $\Delta t$ considered
    'BS'      : 30,  # Number of bootstrap/toy MC samples
    'TSPAN'   : 200,  # Convolution kernel span (days), long tails
    'alpha'   : 0.15, # Regularization strength
    'mode'    : 'C',  # Inversion input ('C' for reported cases, 'F' for fatalities)
    
    'q'       : estimators.q95,   # bootstrap custom quantiles [0,1]

    'data_poisson' : True, # Statistical uncertainty on
    'kernel_syst'  : True  # Systematic uncertainty on
}


# ** Load datasets **
sys.path.append('./dataconfig')
import load_datasets
CSETS,CSETS_sero = load_datasets.default()

'''
# Choose datasets from datasets.py
CSETS = {
    'CHE' : datasets.CHE,
    'FIN' : datasets.FIN,
    'LAC' : datasets.LAC
}
'''

# =================================================================

plotfolder = './figs/delays'


# Dataset loop
for key in CSETS.keys():

    # =================================================================
    ### Process data
    print('')
    
    try:
        metadata = {**CSETS[key], **CSETS_sero[key]}
        print(metadata)

        d = cio.data_processor(metadata)
        d = cio.choose_timeline(d, target_key='deaths', first_date=FIRST_DATE, last_date=LAST_DATE)
        print(f'Found dataset <{d["isocode"]}>')

    except:    
        print(f"{colored('Failed to process','yellow')} {metadata['isocode']}")
        print(f'Error: {sys.exc_info()[0]} {sys.exc_info()[1]}')
        continue

    # Extract delays
    testdates = aux.get_datetime(metadata['test_date'])
    output = cstats.covid_extract_deltaT(d=d, testdates=testdates,\
        unfold_param=unfold_param, daily_kernels=metadata['kernel_path'], return_full=True)
    print(output['psi'])
    print(output['deltaT'])
    
    # ====================================================================
    ### Plot
    
    fig,ax = plt.subplots()
    t      = np.arange(len(output['deltaT']['Q50']))
    y      = output['deltaT']['Q50']

    # First row contains the lower errors, the second row contains the upper errors
    # # Note the <-> 16-84 up-down flip, due to inversion.
    yerr   = np.array([y-output['deltaT']['Q84'], output['deltaT']['Q16']-y])     
    plt.errorbar(x=t, y=y, yerr=yerr, marker='o', color=(0,0,0))

    #ax.set_xlim([0,max(t)])
    #ax.set_ylim([0, deltas[-1]])
    
    # Create date labels
    firstdate = datetime.strptime(metadata['test_date'][0], '%Y-%m-%d')
    dates = [firstdate + timedelta(i) for i in range(len(t))]
    labels, positions = aux.date_labels(dates=dates)

    # Set labels
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=-70)

    ax.set_ylabel('$\\Delta t \\;$ s.t. $\\; \\psi(t, \\Delta t) \\simeq 1$')
    ax.set_xlabel(f'$t$ [test day]')
    ax.set_title(f"{metadata['isocode']}")
    
    # Save the plots
    os.makedirs(f'{plotfolder}', exist_ok = True)
    plt.savefig(f'{plotfolder}/unitpsi_iso_{metadata["isocode"]}.pdf', bbox_inches='tight')


print(__name__ + f' done plots under <{plotfolder}>!')
