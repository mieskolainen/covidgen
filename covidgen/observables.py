# COVID MC generator observables and plots
#
# m.mieskolainen@imperial.ac.uk, 2020

import matplotlib.pyplot as plt 
import numpy as np
from datetime import datetime
import os

import covidgen

from estimators import *
from aux import *


def get_observables(B, B3):
    """ Compute simulation output observables.
    
    Args:
        B:  Hypercube representation data
        B3: Correlation representation data
    
    Returns:
        Observables
    """

    ### Compute observables
    OBS = dict()

    # Set indices
    A_ind  = np.arange(0, 2**3)                 # All indices [0, ..., 2^3-1]

    T_ind  = np.array([4,5,6,7])                # Indices for Tested   (first column)
    I_ind  = np.array([2,3,6,7])                # Indices for Infected (second column)
    F_ind  = np.array([1,3,5,7])                # Indices for Fatal    (third column)

    T_AND_I_ind = np.intersect1d(T_ind, I_ind)  # Infected and Tested
    T_AND_F_ind = np.intersect1d(T_ind, F_ind)  # Fatal and Tested
    I_AND_F_ind = np.intersect1d(I_ind, F_ind)  # Infected and Fatal

    # --------------------------------------------------------------------

    # Get full sample observables
    OBS['IR']    = np.sum(B[:,I_ind],       axis=1) / np.sum(B[:,A_ind], axis=1)
    OBS['IFR']   = np.sum(B[:,I_AND_F_ind], axis=1) / np.sum(B[:,I_ind], axis=1)


    # Test sample based estimates denoted with "hats"
    OBS['IR_hat']  = (np.sum(B[:,T_AND_I_ind], axis=1) / np.sum(B[:,T_ind], axis=1))
    OBS['IFR_hat'] = (np.sum(B[:,F_ind],       axis=1) / np.sum(B[:,A_ind], axis=1)) \
                   / (np.sum(B[:,T_AND_I_ind], axis=1) / np.sum(B[:,T_ind], axis=1))

    # Test sample statistics
    OBS['T_and_I']   = np.sum(B[:,T_AND_I_ind], axis=1)
    OBS['T_and_F']   = np.sum(B[:,T_AND_F_ind], axis=1)

    # Total
    OBS['I']     = np.sum(B[:,I_ind], axis=1)
    OBS['T']     = np.sum(B[:,T_ind], axis=1)
    OBS['F']     = np.sum(B[:,F_ind], axis=1)

    return OBS


def print_statistics(OBS, B, B3):
    """ Print simulations statistics.
    """

    ###
    printbar('-')
    print('3-dimensional (~ multinomial) observables')
    printbar('-')
    print('\n')

    print(' [Counts] ')
    printB(B, counts=True)

    print('\n')
    print(' [Unit normalized] ')
    printB(B / np.sum(B[0,:]), counts=False)
    print('\n\n')


    ###
    printbar('-')
    print('Count observables computed from 3-dimensional combinations')
    printbar('-')
    print('\n')

    print('Infected [counts]:              Definition: sum[I=1]')
    pf('<I>',  OBS['I'], counts=True)
    print('\n')

    print('Infected and Tested [counts]:   Definition: sum[T=1 and I=1]')
    pf('<I_and_T>',  OBS['T_and_I'], counts=True)
    print('\n')

    print('Fatal [counts]:                 Definition: sum[F=1]')
    pf('<F>',  OBS['F'], counts=True)
    print('\n')

    print('Fatal and Tested [counts]:      Definition: sum[T=1 AND F=1]')
    pf('<F_and_T>',  OBS['T_and_F'], counts=True)
    print('\n')

    ###
    printbar('-')
    print('Ratio observables computed from 3-dimensional combinations')
    printbar('-')
    print('\n')

    print('Infection Rate [%]:             Definition: sum[I=1] / sum[ANY]')
    pf('<IR>',  OBS['IR'] * 100)
    print('\n')


    print('Infection Fatality Rate [%]:    Definition: sum[I=1 AND F=1] / sum[I=1]')
    pf('<IFR>', OBS['IFR'] * 100)
    print('\n\n')


    ###
    printbar('-')
    print('3-dimensional expectation+correlation observables')
    printbar('-')
    print('\n')
    

    print(f'Expectation values in [0,1]')
    pf('E[T]', B3[:,0])
    pf('E[I]', B3[:,1])
    pf('E[F]', B3[:,2])
    print('\n')
    
    print(f'2-point correlation coefficient in [-1,1]')
    pf('C[TI]', B3[:,3])
    pf('C[TF]', B3[:,4])
    pf('C[IF]', B3[:,5])
    print('\n')
    
    print(f'3-point correlation coefficient in [-1,1]')
    pf('C[TIF]', B3[:,6])
    print('\n')
    printbar()


def plot_histograms(OBS):
    """ Plot histograms.
    (this should be automated a bit, with a loop over the observables)
    """

    os.makedirs('./figs/sim/', exist_ok = True)

    # \mathrm{} == \text{}
    tex_name = {
        'I'       : '\\mathrm{{ I }}',
        'F'       : '\\mathrm{{ F }}',

        'T_and_I' : '\\mathrm{{ T \\wedge I }}',
        'T_and_F' : '\\mathrm{{ T \\wedge F }}',

        'IR'      : '\\mathrm{{ IR  }}',
        'IR_hat'  : '\\widehat{{ \\mathrm{{ IR }}  }}',
        'IFR'     : '\\mathrm{{ IFR }}',
        'IFR_hat' : '\\widehat{{ \\mathrm{{ IFR }} }}'

    }

    # Current date and time
    now = datetime.now()
    timestamp = 1#datetime.timestamp(now)
    
    
    percent = 100

    # Create normal legends
    legs = dict()
    for key in ['I','T_and_I','F', 'T_and_F']:
        legs[key] = '$\\langle {:s} \\rangle$ = {:.3} $[{:.3}, {:.3}]_{{Q68}}$ $[{:.3}, {:.3}]_{{Q95}}$'.format(
                tex_name[key],
                np.mean(OBS[key]),
                np.percentile(OBS[key], Q68[0]), np.percentile(OBS[key], Q68[1]),
                np.percentile(OBS[key], Q95[0]), np.percentile(OBS[key], Q95[1]))

    # Create ratio legends (100 x %)
    R_legs = dict()
    for key in ['IR', 'IR_hat', 'IFR', 'IFR_hat']:
        R_legs[key] = '$\\langle {:s} \\rangle$ = {:.2f} % $[{:.2f}, {:.2f}]_{{Q68}}$ % $[{:.2f}, {:.2f}]_{{Q95}}$ %'.format(
                tex_name[key],
                np.mean(OBS[key]) * percent,
                np.percentile(OBS[key], Q68[0]) * percent, np.percentile(OBS[key], Q68[1]) * percent,
                np.percentile(OBS[key], Q95[0]) * percent, np.percentile(OBS[key], Q95[1]) * percent)

    
    # ------------------------------------------------------------------------
    # I
    fig,ax = plt.subplots(1,1,figsize=aux.set_fig_size())
    binedges_I = np.linspace(0, 2100, 500)
    
    ax.hist(OBS['I'],       bins=binedges_I, histtype='step', lw=1.5, fill=False, alpha=1.0)
    ax.hist(OBS['T_and_I'], bins=binedges_I, histtype='step', lw=1.5, fill=False, alpha=1.0)
    
    plt.legend([legs['I'], legs['T_and_I']])
    plt.xlabel(f'Infected [counts]')
    #plt.title(f'v{covidgen.VER} simulation')
    plt.ylabel('MC simulation runs')
    figname = f'I_{timestamp}.pdf'
    plt.savefig('./figs/sim/' + figname, bbox_inches='tight')
    print(f'Figure saved to ./figs/sim/{figname}')


    # ------------------------------------------------------------------------
    # F
    fig,ax = plt.subplots(1,1,figsize=aux.set_fig_size())
    binedges_F = np.arange(-0.5, 25, 0.25)
    
    ax.hist(OBS['F'],       bins=binedges_F, histtype='step', lw=1.5, fill=False, alpha=1.0)
    ax.hist(OBS['T_and_F'], bins=binedges_F, histtype='step', lw=1.5, fill=False, alpha=1.0)
    
    plt.legend([legs['F'], legs['T_and_F']])
    plt.xlabel(f'Fatal [counts]')
    #plt.title(f'v{covidgen.VER} simulation')
    plt.ylabel('MC simulation runs')
    figname = f'F_{timestamp}.pdf'
    plt.savefig('./figs/sim/' + figname, bbox_inches='tight')
    print(f'Figure saved to ./figs/sim/{figname}')


    # ------------------------------------------------------------------------
    # (I,F)
    fig,ax = plt.subplots(1,1)

    cc = ax.hist2d(OBS['I'], OBS['F'], bins=(np.linspace(1600, 2100, 500), binedges_F), rasterized=True, cmap=plt.cm.RdBu)
    #plt.title(f'v{covidgen.VER} simulation')
    ax.set_xlabel(f'Infected  ${tex_name["I"]}$  [counts]')
    ax.set_ylabel(f'Fatal  ${tex_name["F"]}$  [counts]')
    ratio = 1.0; ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

    figname = f'IF_{timestamp}.pdf'
    plt.savefig('./figs/sim/' + figname, bbox_inches='tight')
    print(f'Figure saved to ./figs/sim/{figname}')
    

    # ------------------------------------------------------------------------
    # IR and IR_hat
    fig,ax = plt.subplots(1,1,figsize=aux.set_fig_size())
    binedges_IR = np.linspace(12, 18, 80)

    ax.hist(OBS['IR']*percent,     bins=binedges_IR, histtype='step', lw=1.5, fill=False, alpha=1.0)
    ax.hist(OBS['IR_hat']*percent, bins=binedges_IR, histtype='step', lw=1.5, fill=False, alpha=1.0)

    plt.legend([R_legs['IR'], R_legs['IR_hat']])
    plt.xlabel(f'Infection Rate [%]')
    #plt.title(f'v{covidgen.VER} simulation')
    plt.ylabel('MC simulation runs')
    plt.ylim([0, len(OBS['IR']) / 5])
    figname = f'IR_{timestamp}.pdf'
    plt.savefig('./figs/sim/' + figname, bbox_inches='tight')
    print(f'Figure saved to ./figs/sim/{figname}')


    # ------------------------------------------------------------------------
    # IFR and IFR_hat
    fig,ax = plt.subplots(1,1,figsize=aux.set_fig_size())
    binedges_IFR = np.linspace(0, 1.0, 150)

    ax.hist(OBS['IFR']*percent,     bins=binedges_IFR, histtype='step', lw=1.5, fill=False, alpha=1.0)
    ax.hist(OBS['IFR_hat']*percent, bins=binedges_IFR, histtype='step', lw=1.5, fill=False, alpha=1.0)

    plt.legend([R_legs['IFR'], R_legs['IFR_hat']])
    plt.xlabel(f'Infection Fatality Rate ($r \\times 100$) [%]')
    #plt.title(f'v{covidgen.VER} simulation')
    plt.xticks(np.arange(0, max(binedges_IFR), 0.1))
    plt.ylabel('MC simulation runs')
    plt.xlim([0, 1.001])
    plt.ylim([0, len(OBS['IFR']) / 14])
    figname = f'IFR_{timestamp}.pdf'
    plt.savefig('./figs/sim/' + figname, bbox_inches='tight')
    print(f'Figure saved to ./figs/sim/{figname}')


    # ------------------------------------------------------------------------
    # (IFR,IFR_hat)
    fig,ax = plt.subplots(1,1)

    ax.hist2d(OBS['IFR']*percent, OBS['IFR_hat']*percent, \
        bins=(np.linspace(0,0.6,150), np.linspace(0,0.6,150)), rasterized=True, cmap=plt.cm.RdBu)
    #plt.colorbar()
    #ax.set_title(f'v{covidgen.VER} simulation')
    ax.set_xlabel(f'${tex_name["IFR"]}$  [%]')
    ax.set_ylabel(f'${tex_name["IFR_hat"]}$  [%]')
    #ax.set_figaspect(1.0)
    ratio = 1.0; ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

    figname = f'IFR_IFR_hat_{timestamp}.pdf'
    plt.savefig('./figs/sim/' + figname, bbox_inches='tight')
    print(f'Figure saved to ./figs/sim/{figname}')



    # ------------------------------------------------------------------------
    # (IR,IFR)
    fig,ax = plt.subplots(1,1)

    ax.hist2d(OBS['IR']*percent, OBS['IFR']*percent, bins=(binedges_IR, binedges_IFR), rasterized=True, cmap=plt.cm.RdBu)
    #plt.title(f'v{covidgen.VER} simulation')
    ax.set_xlabel(f'Infection Rate ${tex_name["IR"]}$ [%]')
    ax.set_ylabel(f'Infection Fatality Rate ${tex_name["IFR"]}$ [%]')
    ratio = 1.0; ax.set_aspect(1.0/ax.get_data_ratio()*ratio)

    figname = f'IR_IFR_{timestamp}.pdf'
    plt.savefig('./figs/sim/' + figname, bbox_inches='tight')
    print(f'Figure saved to ./figs/sim/{figname}')

