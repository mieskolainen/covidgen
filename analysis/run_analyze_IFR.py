# Posterior PDF combination analysis
#
# Before running this script, first execute:
#  >> python ./analysis/run_compute_counts.py
#
# m.mieskolainen@imperial.ac.uk, 2020

import os
import ot
import numpy as np
import pickle
import pandas as pd
import copy
import scipy
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.interpolate import interp1d

# Import local path
import sys

sys.path.append('./analysis')
sys.path.append('./covidgen')

import covidgen
import tools
import estimators
import aux
import functions
import cstats


# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)


# Output folders
texfolder  = './tex'
plotfolder = './figs/combined'

# Latex output
texfile   = f'{texfolder}/run_analyze_counts.tex'
OF        = open(texfile, 'w')


def dprint(text, end='\n'):
    """ Dual print to tex and stdout """
    print(text, end=end)
    OF.write(text + end)


def get_pdf_stats(PDF):
    """
    Copy posterior pdfs into a matrix.
    Calculate basic pdf moments (mean, sigma)
    """
    Y       = np.array([])
    mean    = np.zeros(len(PDF))
    sigma   = np.zeros(len(PDF))

    N = 0
    for key in PDF.keys():
        if N == 0:
            Y    = np.zeros((len(PDF[key]['pdf']), len(PDF)))
            xval = PDF[key]['val']

        # Full pdf
        Y[:,N]   = copy.deepcopy(PDF[key]['pdf'])

        # Mean and sigma
        mean[N]  = scipy.integrate.simps(x=PDF[key]['val'], y=PDF[key]['pdf']*PDF[key]['val'])
        sigma[N] = np.sqrt(scipy.integrate.simps(x=PDF[key]['val'], y=PDF[key]['pdf']*(mean[N] - PDF[key]['val'])**2))
        N += 1
        
    return Y,mean,sigma,xval


def get_tex_string(xval, pdf, CR_val, name):
    """
    Return formatted string of estimates & uncertanties.
    """
    percent = 100
    CR_val *= percent

    # Mean value integral
    IFR     = scipy.integrate.simps(x=xval, y=pdf*xval) * percent

    # Maximum a Posteriori estimate
    maxi    = np.argmax(pdf)
    IFR_MAP = xval[maxi] * percent

    return f'{name:<35} & {IFR_MAP:.2f} & {IFR:.2f} & $[{CR_val[1]:.2f}, {CR_val[2]:.2f}]$ & $[{CR_val[0]:.2f}, {CR_val[3]:.2f}]$ \\\\'


def get_tex_string_alternative(IFR_MAP, IFR, CI_val, name):
    """
    Return formatted string.
    """
    percent = 100

    IFR_MAP *= percent
    IFR     *= percent
    CI_val  *= percent

    return f'{name:<35} & {IFR_MAP:.2f} & {IFR:.2f} & $[{CI_val[1]:.2f}, {CI_val[2]:.2f}]$ & $[{CI_val[0]:.2f}, {CI_val[3]:.2f}]$ \\\\'


def plot_LLR(LLR_all):

    # Loop over all deltaT
    for key in LLR_all.keys():

        aux.printbar()

        deltaT = int(key)
        LLR    = LLR_all[key]
        print(f'\n<deltaT = {deltaT if deltaT >= 0 else "[optimal]"} [days]>\n')
        
        percent = 100
        IFR, CI_val, x2LL_sum, r0_value = estimators.joint_likelihood_combine(x2LL=LLR, return_full=True)
        IFR        *= percent
        CI_val     *= percent

        print(f'LLR method: IFR = {IFR}, CI = {CI_val}')

        for key in LLR.keys():
            r0_value  = LLR[key]['val'] * percent
            LLR_value = LLR[key]['llr']
            plt.plot(r0_value, LLR_value, label=key)

        # Plot vertical bar and lines
        #for i in [0,3]: # CI95
        #    plt.plot(np.ones(2)*CI_val[i], np.array([0, 1e4]), ls=':', color=(0,0,0), alpha=0.5)
        plt.plot(np.ones(2)*IFR, np.array([0, 1e4]), ls=':', color=(0,0,0), alpha=1.0)
        plt.fill_between(np.array([CI_val[0], CI_val[3]]), np.array([0, 0]), np.array([1e4,1e4]), alpha=0.075, color=(0,0,0), linewidth=0)

        # Plot LLR curve
        plt.plot(r0_value, x2LL_sum, color=(0,0,0), ls='-', label='Joint LLR', lw=1.75)
        
        plt.xlabel('Infection Fatality Rate $(r \\times 100)$ [%]')
        plt.ylabel('$-2$ log-likelihood ratio')
        plt.legend(loc=1)
        plt.xticks(np.linspace(0,1.2,13))
        #plt.yscale('log')
        plt.ylim([1e-2, 250])
        plt.xlim([0, 1.0])
        plt.title(f'$\\Delta t=$ {deltaT if deltaT >= 0 else "[optimal]"} days')
        #plt.show()

        ## Save events
        os.makedirs(f'{plotfolder}', exist_ok = True)
        plt.savefig(f'{plotfolder}/combined_LLR_deltaT_{deltaT}.pdf', bbox_inches='tight')
        
        plt.close()


# ========================================================================
### Read computed input data
# ========================================================================


percent  = 100
qval     = estimators.q68_q95

filename = './output/computed_counts.pkl'
with open(filename, 'rb') as f:
    obj, analysis_param, unfold_param = pickle.load(f)

cstats.covid_print_stats(obj=obj, dprint=dprint)


# ========================================================================
### Combine data
# ========================================================================


plot_LLR(obj['LLR'])

# Plotting parameters
plot_DELTA2 = False


# Optimal Transport entropic regularization strength
# (minimum possible such that the results are stable)
# Used only with bregman iteration based algorithm from OT toolbox.
OT_reg = 3e-3
OT_algorithm = 'quantile'


# Loop over all deltaT
for key in obj['PDF'].keys():

    aux.printbar()

    delta = int(key)
    PDF   = obj['PDF'][key]
    LLR   = obj['LLR'][key]


    dprint(f'\n<delta = {delta if delta >= 0 else "[optimal]"} [days]>\n')
    Y,mean,sigma,xval = get_pdf_stats(PDF)

    # ========================================================================
    ### Normalized product and sum estimators
    # ========================================================================
    K = Y.shape[1] # Number of datasets
    
    # 1/Z Product (~ geometric mean type)
    prodpdf  = np.prod(Y, axis=1)
    prodpdf /= scipy.integrate.simps(x=xval, y=prodpdf)
    
    # 1/K Sum (~ arithmetic mean type)
    sumpdf   = np.sum(Y, axis=1) / K
    
    # ========================================================================
    ### Method of Moments estimator
    # ========================================================================

    rhat_mom, delta2_mom, w_mom, err_rhat_mom = estimators.mom_combine(r=mean, s2=sigma**2, N=10, debug=True)

    # Take equivalent Gaussian pdf (note we plot here err_rhat for sigma)
    mompdf = functions.normpdf(x=xval, mu=rhat_mom, std=err_rhat_mom); dprint('\n')
    

    # ========================================================================
    ### Normal-Normal likelihood based estimator
    # ========================================================================

    rhat_nll, delta2_nll, err_rhat_nll, err_delta2_nll = estimators.nnl_combine(r=mean, s2=sigma**2, N=10, debug=True, alpha=0.32)
    dprint(f'NL: rhat = {rhat_nll:0.5f} +- {err_rhat_nll}, delta = {np.sqrt(delta2_nll):0.5f} +- {np.sqrt(err_delta2_nll)}')

    # Take equivalent Gaussian pdf (note we plot here err_rhat for sigma)
    err_rhat_sym = np.mean(np.abs(rhat_nll - err_rhat_nll))
    nnlpdf = functions.normpdf(x=xval, mu=rhat_nll, std=err_rhat_sym); dprint('\n')


    # ========================================================================
    ### Wasserstein barycenter pdf fusion estimators
    # ========================================================================

    # Decimate factor (no aliasing protection needed here, pdf is very smooth / low freqs)
    # (USE ONLY FOR TESTING)
    DECIM   = 1
    xval_   = xval[0::DECIM]
    Y_      = Y[0::DECIM,:]


    wpdf    = estimators.wass_combine(xval=xval_, A=Y_, reg=OT_reg, weighted=False, algorithm=OT_algorithm)
    wpdf_W  = estimators.wass_combine(xval=xval_, A=Y_, reg=OT_reg, weighted=True, w=1.0/sigma**2, algorithm=OT_algorithm)
    
    # ========================================================================
    ### Plot combined PDF estimates

    # Collect all
    all_pdf = {
        'MoM' :                  (xval, mompdf),
        'NL'  :                  (xval, nnlpdf),
        'OT'  :                  (xval_, wpdf),
        '$1/\\sigma_i^2$ OT' :   (xval_, wpdf_W),
        '$1/K$ SUM'  :           (xval, sumpdf),
        '$1/Z$ PROD' :           (xval, prodpdf),
    }

    # Print combination estimator results
    for key in all_pdf.keys():
        x_dense, pdf_dense, CR_value = tools.get_credible_regions(xval=all_pdf[key][0], pdf=all_pdf[key][1], prc=qval)
        tex_str = get_tex_string(xval=x_dense, pdf=pdf_dense, CR_val=CR_value, name=key)
        dprint(f'{tex_str}')


    # ========================================================================
    ### Joint-log likelihood estimator
    # ========================================================================

    IFR, CI_val = estimators.joint_likelihood_combine(x2LL=LLR)
    dprint(get_tex_string_alternative(IFR_MAP=IFR, IFR=IFR, CI_val=CI_val, name='Joint LLR'))
    dprint('')


    # ========================================================================
    ### Plot combination results

    fig,ax     = plt.subplots(1,1,figsize=aux.set_fig_size())
    linestyles = ['-', ':', '-', '--', '-.', ':']
    colors     = [ 'tab:blue', 'tab:blue', (0,0,0), (0,0,0), (0.5,0.5,0.5), (0.5,0.25,0.25)]

    # --------------------------------
    # Plot delta2 band

    if plot_DELTA2:
        MAX = 1e5
        for i in [0,1]:
            sign = -1 if i == 0 else 1 # visualized by centering around \hat{r}
            x = (rhat_nll + sign*np.sqrt(err_delta2_nll)) * percent
            plt.plot((rhat_nll + sign*np.sqrt(delta2_nll))*np.ones(2) * percent, np.array([0,MAX]), color='tab:blue', ls=':', alpha=0.4)
            if i == 0:
                param = {'label': '$\\hat{r} \\pm \\hat{\\Delta}$'}
            else:
                param = {}

            # Uncertainty band
            plt.fill_between(x, np.array([0, 0]), np.array([MAX,MAX]), alpha=0.075, color='tab:blue', linewidth=0, **param)
    # --------------------------------
    
    # Add each method
    i = 0
    for key in all_pdf.keys():
        plt.plot(all_pdf[key][0] * percent, all_pdf[key][1], linestyle=linestyles[i], color=colors[i], label=key)
        i += 1

    plt.xlabel('Infection Fatality Rate $(r \\times 100)$ [%]')
    plt.ylabel('Posterior pdf')
    plt.legend()
    plt.xticks(np.linspace(0,1.2,13))
    plt.ylim([0,1000])
    plt.xlim([0,1.0])
    plt.title(f'$\\Delta t=$ {delta if delta >= 0 else "[optimal]"} days')

    ## Save events
    os.makedirs(f'{plotfolder}', exist_ok = True)

    #plt.show()
    plt.savefig(f'{plotfolder}/combined_pdf_deltaT_{delta}.pdf', bbox_inches='tight')


    # ========================================================================
    ### Plot individual PDFs

    fig,ax = plt.subplots(1,1, figsize=aux.set_fig_size())

    for key in PDF.keys():
        plt.plot(PDF[key]['val'] * percent, PDF[key]['pdf'], label=key)
        dprint(f"{get_tex_string(xval=PDF[key]['val'], pdf=PDF[key]['pdf'], CR_val=PDF[key]['CR_value'], name=key)}")

    plt.xlabel('Infection Fatality Rate $(r \\times 100)$ [%]')
    plt.ylabel('Posterior pdf')
    plt.legend()
    plt.xticks(np.linspace(0,1.2,13))
    plt.ylim([0,1400])
    plt.xlim([0,1.0])
    plt.title(f'$\\Delta t=$ {delta if delta >= 0 else "[optimal]"} days')

    ## Save events
    os.makedirs(f'{plotfolder}', exist_ok = True)
    #plt.show()
    plt.savefig(f'{plotfolder}/pdf_deltaT_{delta}.pdf', bbox_inches='tight')
    #plt.close()

    # ========================================================================
    # Plot normal-normal model 2D-likelihood

    df = 2 # degrees of freedom
    rhat_scan  = np.linspace(-0.0,  0.01, 500)
    delta_scan = np.linspace(-0.0, 0.005, 500)
    
    # Obtain likelihood values
    ZZ = np.zeros((len(rhat_scan), len(delta_scan)))
    for i in range(len(rhat_scan)):
        for j in range(len(delta_scan)):
            ZZ[j,i] = estimators.nnl_2logL(rhat=rhat_scan[i], delta2=delta_scan[j]**2, r=mean, s2=sigma**2)

    ZZ     = -(ZZ - np.max(ZZ[:]))
    XX,YY  = np.meshgrid(rhat_scan*percent, delta_scan*percent)
    
    # Plot iso-contours
    fig,ax = plt.subplots()
    ccx    = ax.contour(XX, YY, ZZ,
        levels = (stats.chi2.ppf(q=0.68, df=df), stats.chi2.ppf(q=0.95, df=df), stats.chi2.ppf(q=0.99, df=df)),
        colors = ['black', 'tab:gray', 'tab:red'], linestyles = ['solid', 'solid', 'dotted'])

    # Plot the maximum Likelihood value
    minind2d = np.unravel_index(ZZ.argmin(), ZZ.shape)
    plt.plot(rhat_scan[minind2d[1]]*percent, delta_scan[minind2d[0]]*percent, 'ko')

    # Change styles
    fmt  = {}
    strs = ['CI 68', 'CI 95', 'CI 99']
    for l,s in zip(ccx.levels, strs):
        fmt[l] = s
    plt.xticks(np.linspace(0, 1.0, 11))
    plt.yticks(np.linspace(0, 0.5, 11))
    
    plt.clabel(ccx, inline=1, fontsize=10, fmt=fmt)
    plt.xlabel('Infection Fatality Rate $(r \\times 100)$ [%]')
    plt.ylabel('Dispersion $(\\Delta \\times 100)$ [%]')
    plt.title(f'$\\Delta t=$ {delta if delta >= 0 else "[optimal]"} days')

    plt.savefig(f'{plotfolder}/combined_nll_2D_deltaT_{delta}.pdf', bbox_inches='tight')


print('\n')
print(__name__ + f': Output plots produced under <{plotfolder}> and output tex under <{texfolder}>')


