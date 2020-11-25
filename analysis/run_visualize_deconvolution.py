# Visualize deconvolution of COVID time-series
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
from   tqdm import tqdm


sys.path.append('./analysis')
sys.path.append('./covidgen')
sys.path.append('./dataconfig')


# ** Generate delay kernels **
import run_generate_kernels

import aux
import tools
import functions
import estimators
import cstats
import cio


# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)


plotfolder = './figs/deconvolution'

# =================================================================
### Parameters

# Time interval
FIRST_DATE = '2020-03-01'
LAST_DATE  = '2020-06-15'

# Which read-out delays to plot for psi(t,\Delta t) function    
deltas = [0, 7, 14, 21]

unfold_param = {
    'ZEROPAD' : 31,   # Convolution domain pre-extension, also the maximum $\Delta t$ considered
    'BS'      : 300,  # Number of MC samples
    'TSPAN'   : 200,  # Convolution kernel span (days), long tails
    'alpha'   : 0.15, # Regularization strength
    'mode'    : 'C',  # Inversion input ('C' for reported cases, 'F' for fatalities)

    'q'       : estimators.q95, # bootstrap custom quantiles [0,1]
    
    'data_poisson' : True, # Statistical uncertainty on
    'kernel_syst'  : True  # Systematic uncertainty on
}


# ** Load datasets **
import load_datasets
CSETS,CSETS_sero  = load_datasets.default()


# Add additional datasets here !
import datasets
import datasets_sero

CSETS['CHE']      = datasets.CHE
CSETS_sero['CHE'] = {} # no prevalance test data


# Delay kernel path
kernel_path = './output/kernels_daily.pkl'

# =================================================================


# Dataset loop
for key in CSETS.keys():

    #if key is not 'CHE':
    #    continue
    
    # -----------------------------------------------------------------
    ### Process data
    print('')
    
    try:
        metadata = {**CSETS[key], **CSETS_sero[key]}
        print(metadata)

        d = cio.data_processor(metadata)
        d = cio.choose_timeline(d, first_date=FIRST_DATE, last_date=LAST_DATE)

        print(f'Found dataset <{d["isocode"]}>')
    except:

        print(f"Failed to process {metadata['isocode']}")
        print(f'Error: {sys.exc_info()[0]} {sys.exc_info()[1]}')
        continue

    # -----------------------------------------------------------------
    ### Construct time-series arrays

    # Padd with technical zeros in front, due to deconvolution 'causal unroll'
    Cdiff  = np.hstack((np.zeros(unfold_param['ZEROPAD']), d['cases']))
    Fdiff  = np.hstack((np.zeros(unfold_param['ZEROPAD']), d['deaths']))
    #Tdiff  = np.hstack((np.zeros(unfold_param['ZEROPAD']), d['tests']))
    t      = np.arange(unfold_param['ZEROPAD'] + len(d['dt']))

    # Remove NaN
    Cdiff[~np.isfinite(Cdiff)] = 0
    Fdiff[~np.isfinite(Fdiff)] = 0

    ### Get datetime objects with domain extension (zeropadding)
    dt_orig, dt_shift, dt_tot = aux.get_datetime(dt=d['dt'], shift=(-1)*unfold_param['ZEROPAD'])

    ### Load kernel pdfs
    with open(kernel_path,'rb') as f:
        kp = pickle.load(f)

    # -----------------------------------------------------------------
    ### Deconvolution inverse

    # NOTE! alpha <- alpha * len(t)
    Idiff_hat_bs, Fdiff_hat_bs = cstats.covid_deconvolve(Cdiff=Cdiff, Fdiff=Fdiff, kp=kp, \
        mode=unfold_param['mode'], alpha=unfold_param['alpha']*len(t), BS=unfold_param['BS'], \
        TSPAN=unfold_param['TSPAN'], data_poisson=unfold_param['data_poisson'], kernel_syst=unfold_param['kernel_syst'])
    
    ### Compute psi-function (delay scale) for the visualization, for different deltaT values
    psi_bs = {}
    for k in tqdm(range(len(deltas))):
        psi_bs[str(deltas[k])] = cstats.covid_psi_random(Idiff_hat=Idiff_hat_bs, kp=kp, \
            delta=deltas[k], BS=unfold_param['BS'], TSPAN=unfold_param['TSPAN'], kernel_syst=unfold_param['kernel_syst'])
    
    # -----------------------------------------------------------------
    ### Visualize
    
    title = f"{d['isocode']} data $N={d['population']/1E6:.2f}$M"
    testdates = aux.get_datetime(metadata['test_date']) if 'test_date' in metadata else None

    fig,ax = cstats.covid_plot_deconv(dates=dt_tot, Cdiff=Cdiff, Fdiff=Fdiff, \
        Idiff_hat_bs=Idiff_hat_bs, Fdiff_hat_bs=Fdiff_hat_bs, \
        psi_bs=psi_bs, testdates=testdates, title=title, q=unfold_param['q'], Nx_ticks=10)
    
    # Save plot
    os.makedirs(f'{plotfolder}', exist_ok = True)
    plt.savefig(f'{plotfolder}/deconv_iso_{d["isocode"]}.pdf', bbox_inches='tight')
    plt.close()



print(__name__ + f' plots produced under <{plotfolder}>')
