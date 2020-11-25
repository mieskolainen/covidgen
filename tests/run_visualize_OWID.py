# Visualize Our-World-In-Data (OWID) time-series data
#
# https://ourworldindata.org/coronavirus
#
# m.mieskolainen@imperial.ac.uk, 2020


import numpy as np
import matplotlib.pyplot as plt
import numba
import os
import sys
import copy
import pandas as pd
from tqdm import tqdm
import scipy
from scipy.interpolate import interp1d
from datetime import datetime
import os

sys.path.append('./analysis')
sys.path.append('./covidgen')
import aux
import tools
import functions
import estimators
import cio

sys.path.append('./dataconfig')
import datasets

# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)

# Number of x-axis ticks
Nx_ticks = 10


# ------------------------------------------------------------------------
# Single time delay kernel [Case -> Fatality] parameters (same for all data)

# Weibull parameters
a_wei = 13            # Weibull median is a*ln(2)^(1/k)
k_wei = 1.5
a_perturbation = 3

# Weibull lambda
a_values = np.array([a_wei-a_perturbation, a_wei, a_wei+a_perturbation])

W_mean = a_values * scipy.special.gamma(1 + 1/k_wei)
W_med  = a_values * np.log(2)**(1/k_wei)

print(f'Weibull mean   = {W_mean}')
print(f'Weibull median = {W_med}')


# ------------------------------------------------------------------------
# Analysis window
FIRST_DATE = '2020-03-01' 
LAST_DATE  = '2030-01-01'

REQUIRE_MARCH = False # Require data starting from March
SUPER = 10 # Interpolation factor
W     = 7  # Window size is 2xW

# Choose all countries
ISO  = cio.get_european_isocodes()
SETS = copy.deepcopy(ISO)

# Choose a spesific set of countries
#SETS = ['DEU', 'EST', 'CHE']
# ------------------------------------------------------------------------



def get_del(x,y):
    """
    Get convolution delayed function values.
    """
    out = []
    for a_wei in a_values:
        h_param = {'a': a_wei, 'k': k_wei}
        out.append( tools.convint(t=x, f=y, kernel=functions.h_wei, kernel_param=h_param) )
    return out


def analyze1(ax, data, labelsize=7, EPS=1e-15):
    """
    Compute observables set 1.

    - Time windowed cases wC(t) over time windowed tests wT_C(t).
    - Time windowed cases convolution delayed (K * wC)(t) over time windowed tests wT_C(t).
    - Cumulative number of tests T_C(t).
    """

    N = len(data['Cdiff'])
    C_T_window   = np.zeros(N)
    C_T_window_0 = np.zeros(N)
    C_T_window_1 = np.zeros(N)
    C_T_window_2 = np.zeros(N)

    t = data['t']
    Cdiff = data['Cdiff']
    Tdiff = data['Tdiff']

    for i in range(len(C_T_window) - W - 1):
        C_T_window[i] = np.sum(Cdiff[i-W:i+W]) / np.maximum(np.sum(Tdiff[i-W:i+W]), EPS)

    delayed = get_del(t, Cdiff)

    for i in range(len(C_T_window) - W - 1):
        C_T_window_0[i] = np.sum(delayed[0][i-W:i+W]) / np.maximum(np.sum(Tdiff[i-W:i+W]), EPS)
    for i in range(len(C_T_window) - W - 1):
        C_T_window_1[i] = np.sum(delayed[1][i-W:i+W]) / np.maximum(np.sum(Tdiff[i-W:i+W]), EPS)
    for i in range(len(C_T_window) - W - 1):
        C_T_window_2[i] = np.sum(delayed[2][i-W:i+W]) / np.maximum(np.sum(Tdiff[i-W:i+W]), EPS)
    # --------------------------------------------------------------------
    
    ###
    ax.plot(t,         C_T_window, color=(0,0,0), label='$wC(t)$ / $wT_C(t)$ (windowed)')
    ax.fill_between(t, C_T_window_2, C_T_window_0, color=(0,0,0), alpha=0.1)
    ax.plot(t,         C_T_window_1, color=(0,0,0), linestyle='--', label='$wC(t)$ $\\otimes$ delay / $wT_C(t)$')
    
    ax.legend(loc='upper left', frameon=True) 
    ax.set_ylabel('Ratio')
    ax.set_xlim([0, t[-1]*1.05])
    ax.set_ylim([0, 0.25])
    ax.set_xticks(np.arange(t[0], t[-1]+1, Nx_ticks))
    ax.set_title(f"{data['d']['isocode']} data $N={data['d']['population']/1E6:.2f}$M | kernel $C \\rightarrow F$: $\\lambda={a_values[1]:0.0f} \\pm {a_perturbation}, k={k_wei:0.1f}$")
    ax.tick_params(axis='x', labelsize=labelsize)

    # Instantiate a second axes that shares the same x-axis
    ax2   = ax.twinx()
    color = 'tab:blue'

    ax2.plot(data['t'], data['Tcum'], color=color, linestyle='-', label='tests $T_C(t)$')
    ax2.set_xticks(np.arange(t[0], t[-1]+1, Nx_ticks))
    ax2.set_ylim([0, None])
    ax2.set_ylabel(f'[counts]', color=color, rotation=270, labelpad=17)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc=1)

    if data['prevalence_test_index'] is not None:
        ax.plot(np.ones((2,1))*data['prevalence_test_index'], np.array([0,np.max(data['Ccum'])*1.1]), 'k', lw=5, alpha=0.5)


def analyze2(ax, data, labelsize=7, EPS=1e-15):
    """
    Compute observables set 2.

    - [Forward delayed cases (K * C)(t + \\Delta t) over cases C(t)] ratio isocontours.
    """

    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink']
    i = 0
        
    # Convolve data
    C_del = get_del(data['t_super'], data['Ccum_super'])

    t_super = data['t_super']
    t       = data['t']

    for q in [1.25, 1.1, 0.95, 0.75, 0.5, 0.25]:

        y = [] # Find the capture shift
        for k in range(len(C_del)):
            y.append(tools.find_delay(t=t_super, F=data['Ccum_super'], Fd=C_del[k], rho=q))

        # Filter out noise by cutting away last bins
        mask  = np.ones(len(y[0]), dtype=np.bool)
        if q > 1: 
            mask[int(len(mask)*0.6):] = False

        ###
        ax.fill_between(t_super[mask], y[2][mask], y[0][mask], step="pre", alpha=0.2, color=colors[i], lw=0)
        ax.step(t_super[mask], y[1][mask], color=colors[i], label=f'$\\epsilon = {q:0.2f}$')
        i += 1

    ###
    ax.set_ylabel('$\\Delta t$ [days] | $\\frac{(K \\ast C)(t + \\Delta t)}{C(t)} = \\epsilon$')
    ax.legend(fontsize=8, loc=4)
    ax.set_xlim([0, t[-1]*1.05])
    ax.set_ylim([0, 30])
    ax.set_xticks(np.arange(t[0], t[-1]+1, Nx_ticks))
    ax.set_yticks(np.linspace(0,30,6))
    ax.tick_params(axis='x', labelsize=labelsize)

    # --------------------------------------------------------------------
    Ccum_del = get_del(data['t'], data['Ccum'])
    # --------------------------------------------------------------------

    ###
    ax2   = ax.twinx()
    color = 'gray'

    ax2.plot(t, Ccum_del[1] / np.maximum(data['Ccum'], EPS), color=color, linestyle='--', label='delay scale $\\xi(t)$')
    ax2.fill_between(t, Ccum_del[2] / np.maximum(data['Ccum'], EPS), Ccum_del[0] / np.maximum(data['Ccum'], EPS), step="pre", alpha=0.2, color=(0,0,0), lw=0)

    ###
    ax2.set_ylabel('Delay scale $\\xi(t)$', color=color, rotation=270, labelpad=17)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 1.05])
    ax2.set_yticks(np.linspace(0,1,6))
    #ax2.legend(loc=1, frameon=True)

    if data['prevalence_test_index'] is not None:
        ax.plot(np.ones((2,1))*data['prevalence_test_index'], np.array([0,30]), 'k', lw=5, alpha=0.5)


def analyze3(ax, data, labelsize=7):
    """
    Compute observables set 3.

    - Cumulative cases C(t)
    - Forward delayed cases (K * C)(t)
    - Cumulative deaths F(t)
    """

    # --------------------------------------------------------------------
    delayed = get_del(data['t'], data['Ccum'])
    # --------------------------------------------------------------------

    t = data['t']

    ###
    ax.plot(t, data['Ccum'], color=(0,0,0), label='$C(t)$')
    ax.fill_between(t, delayed[2], delayed[0], color=(0,0,0), alpha=0.1)
    ax.plot(t, delayed[1], color=(0,0,0), linestyle='--', label='$C(t)$ $\\otimes$ delay')

    ###
    ax.set_ylabel('Cases [counts]')
    ax.set_xticks(np.arange(t[0], t[-1]+1, Nx_ticks))
#    ax[2].set_ylim([0, None])
    ax.set_ylim([0, delayed[1][-1]*1.2 ])
    ax.set_xlim([0, t[-1]*1.05])
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.legend(loc=2, frameon=True)

    ###
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    
    ###
    ax2.plot(t, data['Fcum'], color=color, linestyle='-', label='deaths $F(t)$')
    
    ###
    ax2.set_ylabel('Deaths [counts]', color=color, rotation=270, labelpad=17)  # we already handled the x-label wC_Th ax1
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax2.set_ylim([0, data['Fcum'][-1]*1.2 ])
    ax2.legend(loc=4, frameon=True)
    #ax2.set_yticks(np.linspace(0,1,6))


def analyze4(ax, data, labelsize=7):
    """
    Compute observables set 4.

    - Time windowed case fatality rate wCFR(t)
    - Time windowed case fatality rate wCFR(t) with delay treatment
    """

    # --------------------------------------------------------------------
    N = len(data['Fdiff'])
    CFR_window       = np.zeros(N)
    CFR_window_del   = np.zeros(N)
    CFR_window_del_0 = np.zeros(N)
    CFR_window_del_1 = np.zeros(N)

    y = get_del(data['t'], data['Cdiff'])

    t     = data['t']
    Cdiff = data['Cdiff']
    Fdiff = data['Fdiff']

    for i in range(W, len(CFR_window) - W - 1):
        CFR_window[i]     = np.sum(Fdiff[i-W:i+W]) / np.sum(Cdiff[i-W:i+W])
    for i in range(W, len(CFR_window) - W - 1):
        CFR_window_del[i] = np.sum(Fdiff[i-W:i+W]) / np.sum(y[1][i-W:i+W])

    for i in range(W, len(CFR_window) - W - 1):
        CFR_window_del_0[i] = np.sum(Fdiff[i-W:i+W]) / np.sum(y[0][i-W:i+W])
    for i in range(W, len(CFR_window) - W - 1):
        CFR_window_del_1[i] = np.sum(Fdiff[i-W:i+W]) / np.sum(y[-1][i-W:i+W])

    # --------------------------------------------------------------------
    ###
    percent = 100

    ax.plot(t,         CFR_window * percent,     color=(0,0,0), label='wCFR', linestyle='-')
    ax.fill_between(t, CFR_window_del_0 * percent, CFR_window_del_1 * percent, alpha=0.2, color=(0,0,0), lw=0)
    ax.plot(t,         CFR_window_del * percent, color=(0,0,0), label='wCFR $\\otimes$ delay', linestyle='--')

    ###
    ax.set_ylabel('[%]')
    
    # Set x-tick labels
    dt_orig = aux.get_datetime(dt=data['d']['dt'], shift=0)
    labels, positions = aux.date_labels(dates=dt_orig, N=Nx_ticks)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=-70)
    ax.tick_params(axis='x', labelsize=labelsize)

    ax.set_xlabel("$t$ [days]")
    ax.set_ylim([0, None])

    ax.set_xlim([0, t[-1]*1.05])
    ax.legend(loc='upper left', frameon=True)



def main():

    EPS  = 1E-15

    for isocode in SETS:
        try:
            # Use generic template and set the isocode
            metadata = copy.deepcopy(datasets.TEMPLATE)
            metadata['isocode'] = isocode

            d = cio.data_processor(metadata)
            d = cio.choose_timeline(d, first_date=FIRST_DATE, last_date=LAST_DATE)

            print(f'Found dataset <{d["isocode"]}> with population <{d["population"]:0.0f}>')
        except:
            
            print(f"Failed to process {metadata['isocode']}")
            print(f'Error: {sys.exc_info()[0]} {sys.exc_info()[1]}')
            continue

        # ------------------------------------------------------------------------
        ### ONLY MEASUREMENTS STARTED MARCH OR AFTER
        
        a = datetime.strptime(d['dt'][0], '%Y-%m-%d')

        if REQUIRE_MARCH and a.month > 3: # Too late
            print('Cannot use, not data available from March \n')
            continue;

        # ------------------------------------------------------------------------
        ### Find testing date index

        if 'test_date' in metadata:
            prevalence_test_index  = np.where(dt == metadata['test_date'][0])
            print(f'test day index = {prevalence_test_index}')
        else:
            prevalence_test_index = None

        # ------------------------------------------------------------------------
        # Generate data arrays

        data = {}
        data['d']     = d
        data['prevalence_test_index'] = prevalence_test_index

        data['t']     = np.arange(0, len(d['dt']))
        data['Cdiff'] = np.maximum(copy.deepcopy(d['cases']),  EPS)
        data['Fdiff'] = np.maximum(copy.deepcopy(d['deaths']), EPS)
        data['Tdiff'] = np.maximum(copy.deepcopy(d['tests']),  EPS)

        # Remove NaN
        data['Cdiff'][~np.isfinite(data['Cdiff'])] = 0
        data['Fdiff'][~np.isfinite(data['Fdiff'])] = 0
        data['Tdiff'][~np.isfinite(data['Tdiff'])] = 0

        # Cumulative numbers
        data['Fcum']  = np.cumsum(data['Fdiff'])
        data['Ccum']  = np.cumsum(data['Cdiff'])
        data['Tcum']  = np.cumsum(data['Tdiff'])
        
        # -------------------------------------------------------------------------------
        ### Interpolate values

        # Time axis [days]
        N = len(data['t'])
        data['t_super'] = np.linspace(0, N-1, N*SUPER)

        ff         = interp1d(data['t'], data['Ccum'], kind='slinear')
        data['Ccum_super'] = ff(data['t_super'])
        ff         = interp1d(data['t'], data['Fcum'], kind='slinear')
        data['Fcum_super'] = ff(data['t_super'])
        ff         = interp1d(data['t'], data['Tcum'], kind='slinear')
        data['Tcum_super'] = ff(data['t_super'])

        # ------------------------------------------------------------------------
        ### Compute and plot

        fig,ax = plt.subplots(4,1, figsize=(6,10))
        labelsize = 6

        # Analyze
        analyze1(ax[0], data, labelsize=labelsize)
        analyze2(ax[1], data, labelsize=labelsize)
        analyze3(ax[2], data, labelsize=labelsize)
        analyze4(ax[3], data, labelsize=labelsize)

        # Save the plots                   
        os.makedirs('./figs/OWID', exist_ok = True)
        plt.savefig(f'./figs/OWID/iso_{isocode}_time_domain.pdf', bbox_inches='tight')
        plt.close()
        print('')

    os.system("gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile=./figs/OWID/merged.pdf ./figs/OWID/iso*.pdf")


if __name__ == "__main__":
    main()
    print(__name__ + ' done!')
