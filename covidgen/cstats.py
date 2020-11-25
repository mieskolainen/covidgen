# COVID statistics functions
#
# m.mieskolainen@imperial.ac.uk, 2020

import numba
import numpy as np
import bisect
import copy
import pickle
import matplotlib.pyplot as plt
import scipy
from   termcolor import colored,cprint
import sys
import datetime

from tqdm import tqdm

from scipy.integrate import simps
from scipy.interpolate import interp1d

import estimators
import functions
import tools
import aux
import cio


def covid_psi_random(Idiff_hat, kp, t=None, delta=0, BS=1000, TSPAN=200, kernel_syst=True):
    """
    COVID psi(t,\\delta t) delay scale function with uncertainties
    by re-sampling of kernels and Idiff_hat estimates.
    Args:
        Idiff_hat:    re-sampled and estimated daily infections array (BS x dim)
        kp:           kernel parameters dictionary
        t:            time array, default None (constructed automatically)
        TSPAN:         minimum convolution domain span of the kernel
        delta:        time delay (array indices) for the death count 'read-out'
        BS:           number of bootstrap resamples
        kernel_syst:  convolution kernel perturbations on/off according to their uncertainties
    Returns:
        psi:          random sample of psi functions
    """

    if Idiff_hat.shape[0] != BS:
        raise Exception('covid_psi: BS != I_hat.shape[0]')        
    
    if t is None:
        # Construct convolution domain
        t = np.arange(0, len(Idiff_hat) + TSPAN)

    # Monte Carlo re-sampling
    psi = np.zeros((BS, Idiff_hat.shape[1]))
    for i in range(BS):
        
        # Generate new kernels according to their uncertainties
        if kernel_syst:
            K = covid_kernels(t=t, mu=kp['mu'], sigma=kp['sigma'], mu_std=kp['mu_std'], sigma_std=kp['sigma_std'])
        else:
            K = covid_kernels(t=t, mu=kp['mu'], sigma=kp['sigma'], mu_std=None, sigma_std=None) 

        # Turn the daily density into CDF distribution
        I = np.cumsum(Idiff_hat[i,:])

        # Calculate the ratio
        psi[i,:] = covid_psi(I=I, K_F=K['F'], K_S=K['S'], delta=delta)
    
    return psi


def gen_random_wei_kernel(t, mu,sigma, mu_std, sigma_std, mu_min=0.1, sigma_min=0.1):
    """ Generate new random Weibull kernel
    Args:
        t:                  Discretization values array
        mu, sigma:          Distribution mean and sigma
        mu_std, sigma_std:  Uncertainty (1 sigma) on mu and sigma
        mu_min, sigma_min   Sampling lower bounds on the parameters
    Returns:
        New kernel
    """

    # Conversion to Weibull param
    new_mu    = np.max([mu_min,    mu + np.random.rand()*mu_std])
    new_sigma = np.max([sigma_min, sigma + np.random.rand()*sigma_std])
    W_a, W_k  = tools.get_weibull_param(new_mu, new_sigma)

    # Compute the discretized kernel function
    K = functions.h_wei(t=t, a=W_a, k=W_k, normalize=True) # Note normalize!
    
    return K


def covid_kernels(t, mu, sigma, mu_std=None, sigma_std=None):
    """ 
    COVID time delay kernels.

    Arrays are returned with a continuum integral unit normalization.
    Normalize them to sum=1 if used with discrete convolution.
    
    Args:
        t:                discretization values array
        mu,sigma :        parameter values in dictionaries
        mu_std,sigma_std: parameter uncertainties in dictionaries
    Returns
        kernels
    """
    K = {}

    # No randomization
    if (mu_std is None) & (sigma_std is None):
        for key in mu.keys():
            K[key] = gen_random_wei_kernel(t=t, mu=mu[key], sigma=sigma[key], mu_std=0, sigma_std=0)
    # Random variations    
    else:
        for key in mu_std.keys():
            K[key] = gen_random_wei_kernel(t=t, mu=mu[key], sigma=sigma[key], mu_std=mu_std[key], sigma_std=sigma_std[key])

    ### Combined kernels by numerical convolutions
    K['C'] = tools.convint_(t, K['I2O'], K['O2C']) # Infection -> Onset (x) Onset -> Case report 
    K['S'] = tools.convint_(t, K['I2O'], K['O2S']) # Infection -> Onset (x) Onset -> Seroconversion
    K['F'] = tools.convint_(t, K['C'],   K['C2F']) # Infection -> Case report (x) Case report -> Fatality

    ### Re-normalize the integral to one (numerical protection)
    for key in ['C','S','F']:
        norm    = np.trapz(x=t, y=K[key])
        K[key] /= norm
    
    return K


def covid_deconvolve(Cdiff, Fdiff, kp, t = None, mode='C', alpha=1, BS=1000, TSPAN=200, data_poisson=True, kernel_syst=True):
    """
    COVID time-series deconvolution.
    
    Args:
        Cdiff:         observed daily cases array
        Fdiff:         observed daily fatalities array
        kp:            kernel parameters dictionary
        t:             time points array, default None (constructed automatically)
        mode:          use 'C' for invertion based on cases or 'F' for fatality based
        alpha:         regularization strength for the deconvolution
        BS:            number of bootstrap/MC samples
        TSPAN:         minimum convolution domain span of the kernel
        data_poisson:  measurement statistical bootstrap fluctuation on / off
        kernel_syst:   kernel systematic uncertainties on / off

    Returns:
        Id_hat:        daily infections estimate obtained via deconvolution
        Fd_hat:        daily fatalities from push-forward of the estimate: (K_F * dI/dt)(t)
    """
    print(__name__ + '.covid_deconvolve: Running ...')

    if len(Cdiff) != len(Fdiff):
        raise Exception('covid_deconvolve: input C length != F length')

    if t is None:
        # Construct convolution domain
        t = np.arange(0, len(Fdiff) + TSPAN)

    # Monte Carlo re-sampling of perturbed kernels
    Id_hat = np.zeros((BS, len(Cdiff)))
    Fd_hat = np.zeros((BS, len(Fdiff)))

    for i in tqdm(range(BS)):

        # Generate new kernel
        if kernel_syst:
            K = covid_kernels(t=t, mu=kp['mu'], sigma=kp['sigma'], mu_std=kp['mu_std'], sigma_std=kp['sigma_std'])
        else:
            K = covid_kernels(t=t, mu=kp['mu'], sigma=kp['sigma'], mu_std=None, sigma_std=None)

        # Deconvolution
        if mode == 'C':
            if data_poisson:
                y = np.random.poisson(Cdiff) # Poisson fluctuate measurement
            else:
                y = Cdiff

            y_zp = tools.zeropad_after(x=y, reference=K['C']) # Take care of the tail unrolling
            output, _, _ = tools.nneg_tikhonov_deconv(y=y_zp, kernel=K['C'], alpha=alpha, mass_conserve=True)
            Id_hat[i,:]  = output[0:len(y)]

        elif mode == 'F':
            if data_poisson:
                y = np.random.poisson(Fdiff) # Poisson fluctuate measurement
            else:
                y = Fdiff

            y_zp = tools.zeropad_after(x=y, reference=K['F']) # Take care the tail unrolling
            output, _, _ = tools.nneg_tikhonov_deconv(y=y_zp, kernel=K['F'], alpha=alpha, mass_conserve=True)
            Id_hat[i,:]  = output[0:len(y)]

        else:
            raise Except(__name__ + '.covid_deconvolve: Error: unknown mode.')

        # Push-forward, take care of the tail unrolling by zeropad
        N = len(Id_hat[i,:])
        I_zp       = tools.zeropad_after(x=Id_hat[i,:], reference=K['F'])
        F_diff_hat = tools.conv_(I_zp, K['F'])[0:N]

        # Normalization for visualization and comparisons
        # (absolute normalization not obtained by convolution alone)
        if np.sum(Id_hat[i,:]) > 0:
            Id_hat[i,:] /= np.sum(Id_hat[i,:])
        
        if np.sum(F_diff_hat)  > 0:
            F_diff_hat  /= np.sum(F_diff_hat)

        Id_hat[i,:] *= np.sum(Cdiff)
        F_diff_hat  *= np.sum(Fdiff)
        Fd_hat[i,:]  = F_diff_hat
    
    return Id_hat, Fd_hat


@numba.njit
def covid_psi(I, K_F, K_S, delta=0, EPS=1E-15, reflect=False):
    """
    COVID delay ratio function: psi(t, delta T) = (I * K_F)(t + delta T) / (I * K_S)(t),
    where * means convolution.
    
    Convolution is calculated here via discrete convolution.
    
    Args:
        I     : cumulative infections distribution
        K_F   : sampled delay kernel function in the numerator
        K_S   : sampled delay kernel function in the denominator
        delta : numerator time shift [indices of t]
    Returns:
        Ratio function
    """
    N = len(I)

    # Handle the long tail by zero padding.
    # Kernel array should be a long enough, because it provides reference.
    I_zp = tools.zeropad_after(x=I, reference=K_F)

    # Compute via discrete convolutions
    # Kernel normalization to sum=1 guarantees count normalization
    IKF = tools.conv_(I_zp, K_F / np.sum(K_F))[0:N]
    IKS = tools.conv_(I_zp, K_S / np.sum(K_S))[0:N]

    # Numerator is read out at t + deltaT
    psi = IKF[delta:] / np.maximum(IKS[0:N - delta], EPS)
    out = np.zeros(N)
    out[0:len(psi)] = psi

    if reflect: # Continue with the boundary value
        out[len(psi):] = psi[-1]
    return out

@numba.njit
def covid_gamma(I, K_F, delta=0, EPS=1E-15, reflect=False):
    """
    COVID delay function: gamma(t, deltaT) = (I * K_F)(t + deltaT) / (I * K_F)(t),
    where * means convolution.

    Convolution is calculated here via discrete convolution.
    
    Args:
        I     : cumulative infections distribution
        K_F   : sampled delay kernel function in the numerator and denominator
        delta : numerator time shift [indices of t]
    Returns:
        Ratio function
    """
    N = len(I)

    # Handle the long tail by zero padding.
    # Kernel array should be a long enough, because it provides reference.
    I_zp = tools.zeropad_after(x=I, reference=K_F)

    # Kernel normalization to sum=1 guarantees count normalization
    IK = tools.conv_(I_zp, K_F / np.sum(K_F))[0:N]

    # Numerator is read out at t + deltaT
    gamma = IK[delta:] / np.maximum(IK[0:N - delta], EPS)
    out   = np.zeros(N)
    out[0:len(gamma)] = gamma

    if reflect: # Continue with the boundary value
        out[len(gamma):] = gamma[-1]
    return out


def covid_extract_deltaT(d, testdates, unfold_param, daily_kernels, return_full=False):
    """
    COVID delay scale psi(t,deltaT) == 1 inverted for the optimal deltaT values.
    
    Args:
        d             :  data dictionary
        testdates     :  datetime array [start, end]
        unfold_param  :  unfolding parameters dictionary
        daily_kernels :  daily convolution kernels pickle file
        return_full   :  return all information
    
    Returns:
        deltaT values for the period defined by 'testdates'
    """

    # =================================================================
    ### Construct time-series arrays

    print(__name__ + '.covid_extract_deltaT: Running ...')

    # Pad with technical zeros in FRONT, due to deconvolution 'causal unroll'
    Cdiff  = np.hstack((np.zeros(unfold_param['ZEROPAD']), d['cases']))
    Fdiff  = np.hstack((np.zeros(unfold_param['ZEROPAD']), d['deaths']))
    Tdiff  = np.hstack((np.zeros(unfold_param['ZEROPAD']), d['tests']))

    # Remove NaN
    Cdiff[~np.isfinite(Cdiff)] = 0
    Fdiff[~np.isfinite(Fdiff)] = 0

    ### Get datetime objects. The domain extension is done here using the shift parameter
    dt_orig, dt_shift, dt_tot = aux.get_datetime(dt=d['dt'], shift= (-1)*unfold_param['ZEROPAD'])
    t = np.arange(len(dt_tot))

    ### Load kernel pdfs
    with open(daily_kernels,'rb') as f:
        kp = pickle.load(f)

    # =================================================================
    ### Deconvolution inverse

    # NOTE! alpha <- alpha * len(t)
    Idiff_hat, Fdiff_hat = covid_deconvolve(Cdiff=Cdiff, Fdiff=Fdiff, kp=kp, \
        mode=unfold_param['mode'], alpha=unfold_param['alpha']*len(t), BS=unfold_param['BS'], \
        TSPAN=unfold_param['TSPAN'], data_poisson=unfold_param['data_poisson'], kernel_syst=unfold_param['kernel_syst'])
    
    # Get percentiles
    Idhat = tools.get_bs_percentiles(X=Idiff_hat, q=unfold_param['q'])
    Fdhat = tools.get_bs_percentiles(X=Fdiff_hat, q=unfold_param['q'])

    # --------------------------------------------------------------------
    ### Extract the optimal delays

    deltas = np.arange(0, unfold_param['ZEROPAD'])

    # Loop over different delta
    psi = {
        'Q16'  : np.zeros((len(deltas), len(t))),
        'Q50'  : np.zeros((len(deltas), len(t))),
        'Q84'  : np.zeros((len(deltas), len(t))),
        'Qlo'  : np.zeros((len(deltas), len(t))),
        'Qhi'  : np.zeros((len(deltas), len(t)))
    }

    # Quantiles [0,1]
    qval = np.array([0.16, 0.50, 0.84, unfold_param['q'][0], unfold_param['q'][1]])

    for k in tqdm(range(len(deltas))):
        
        # Compute delay scale function value
        X   = covid_psi_random(Idiff_hat=Idiff_hat, kp=kp, delta=deltas[k], \
            BS=unfold_param['BS'], TSPAN=unfold_param['TSPAN'], kernel_syst=unfold_param['kernel_syst'])

        out = tools.get_bs_percentiles(X=X, q=qval)
        psi['Q16'][k,:] = out[0]
        psi['Q50'][k,:] = out[1]
        psi['Q84'][k,:] = out[2]
        psi['Qlo'][k,:] = out[3]
        psi['Qhi'][k,:] = out[4]

    deltaT = {
        'Q16' : np.ones(len(t)),
        'Q50' : np.ones(len(t)),
        'Q84' : np.ones(len(t)),
        'Qlo' : np.ones(len(t)),
        'Qhi' : np.ones(len(t))
    }

    # Find \Delta t such that \psi(t,\Delta t) \approx 1.0
    target = 1.0
    for i in range(len(t)):

        for key in deltaT.keys():
            diff = psi[key][:,i] - target

            # +1 due to discretization, outer min() for protecting not going over the boundary
            deltaT[key][i]  = np.min([np.argmin(np.abs(diff))+1, len(diff)-1])

    # --------------------------------------------------------------------
    # Extract optimal delays
    
    # Note that we are matching datetime objects here, so any convolution (zero padded)
    # domain extension is properly taken into account here too.
    index           = np.arange(dt_tot.index(testdates[0]), dt_tot.index(testdates[1])+1)

    # Choose only indexed days
    for key in deltaT.keys():
        deltaT[key] = deltaT[key][index]
        psi[key]    = psi[key][:,index]

    print(__name__ + f".covid_extract_deltaT: testdates = {testdates} | optimal deltaT[Q50] = {deltaT['Q50']} days")
    
    if return_full:
        return {'deltaT' : deltaT, 'psi': psi}
    else:
        return deltaT['Q50']


def covid_plot_deconv(dates, Cdiff, Fdiff, Idiff_hat_bs, Fdiff_hat_bs, psi_bs,
    title="", testdates=None, q=np.array([0.025, 0.975]), Nx_ticks=10):
    """
    Plot deconvolution results together with psi(t, \\Delta t) function.
    
    Args:
        dates        :  Datetime objects array
        Cdiff        :  Daily cases
        Fdiff        :  Daily deaths
        Idiff_hat_bs :  Daily infections; from deconvolution, with bootstraps
        Fdiff_hat_bs :  Daily deaths; re-projected with convolution, with bootstraps
        psi_bs       :  Psi function values [dictionary, with keys as different deltaT]; with bootstraps
        title        :  Plot title
        testdates    :  Array of datetime objects of test period (start, end), if any
        q            :  Lower and upper quantiles used in plots [Qlo, Qhi] (0,1)
        Nx_ticks     :  x-axis ticks
        
    Returns:
        fix, ax      :  Figure and axis
    """

    t = np.arange(len(dates))
    q = np.array([q[0], 0.5, q[1]]) # Add median in the middle

    # Get percentiles
    Idhat = tools.get_bs_percentiles(X=Idiff_hat_bs, q=q)
    Fdhat = tools.get_bs_percentiles(X=Fdiff_hat_bs, q=q)

    # =================================================================
    ### PLOT 1: Plot densities
    
    fig,ax = plt.subplots(3, 1, figsize=(6,9))

    ax[0].plot(t, Cdiff,    label = '$dC(t)/dt$',         color=(0,0,0), linestyle='-', drawstyle='steps-mid')
    ax[0].plot(t, Idhat[1], label = '~$d\\hat{I}(t)/dt$', color=(0,0,0), linestyle=':', drawstyle='steps-mid')
    ax[0].fill_between(t, Idhat[0], Idhat[-1], color=(0,0,0), alpha=0.25, linewidth=0)
    ax[0].legend(loc=2)

    #ax[0].set_xlabel('$t$ [days]')
    ax[0].set_title(f'{title}')
    ax[0].set_ylabel('[counts]')
    ax[0].set_xlim([0, np.max(t)])
    ax[0].set_ylim([0, np.max(Cdiff)*1.3])
    ax[0].set_xticks(np.arange(0,np.max(t),Nx_ticks))

    ax2   = ax[0].twinx()
    color = 'tab:red'
    ax2.plot(t, Fdiff,    label = '$dF(t)/dt$',         color=color, linestyle='-', drawstyle='steps-mid')
    ax2.plot(t, Fdhat[1], label = '~$d\\hat{F}(t)/dt$', color=color, linestyle=':', drawstyle='steps-mid')
    ax2.fill_between(t, Fdhat[0], Fdhat[-1],            color=color, alpha=0.2, linewidth=0)

    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel(f'[counts]', color=color, rotation=270, labelpad=17)
    ax2.set_ylim([0, np.max(Fdiff)*1.5])
    ax2.legend(loc=1)

    # =================================================================
    ### PLOT 2: Plot cumulative distributions
    
    ax[1].plot(t, np.cumsum(Cdiff),          label = '$C(t)$',         color=(0,0,0), linestyle='-', drawstyle='steps-mid')
    ax[1].plot(t, tools.cdfint(t, Idhat[1]), label = '~$\\hat{I}(t)$', color=(0,0,0), linestyle=':', drawstyle='steps-mid')
    ax[1].fill_between(t, tools.cdfint(t, Idhat[0]), tools.cdfint(t, Idhat[-1]), color=(0,0,0), alpha=0.25, linewidth=0)
    
    #ax[1].set_xlabel('$t$ [days]')
    ax[1].set_ylabel('[counts]')
    ax[1].set_xlim([0,np.max(t)])
    ax[1].set_ylim([0,None])
    ax[1].legend(loc=2)
    ax[1].set_xticks(np.arange(0,np.max(t),Nx_ticks))

    ax2   = ax[1].twinx()
    color = 'tab:red'

    ax2.plot(t, np.cumsum(Fdiff),          label = '$F(t)$',         color=color, linestyle='-', drawstyle='steps-mid')
    ax2.plot(t, tools.cdfint(t, Fdhat[1]), label = '~$\\hat{F}(t)$', color=color, linestyle=':', drawstyle='steps-mid')
    ax2.fill_between(t, tools.cdfint(t, Fdhat[0]), tools.cdfint(t, Fdhat[-1]), color=color, alpha=0.2, linewidth=0)
    
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel(f'[counts]', color=color, rotation=270, labelpad=17)
    ax2.set_ylim([0,np.max(np.cumsum(Fdiff))*1.5])        
    ax2.legend(loc=4)

    # =================================================================
    ### PLOT 3: Plot scale factor \psi(t,\delta t)
    
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:gray', 'tab:yellow', 'tab:black']
    
    # Loop over different deltaT (read-out delay) choises
    k = 0
    for key in psi_bs.keys():

        psi = tools.get_bs_percentiles(X=psi_bs[key], q=q)
        ind = (Cdiff > 1) & (psi[0] > 1e-6) & (psi[-1] > 1e-6) # Filter out crude noise

        # Plot
        ax[2].fill_between(t[ind], psi[0][ind], psi[-1][ind], label=f'$\\Delta t = {key}$', alpha=0.3, linewidth=1, color=colors[k])
        ax[2].plot(t[ind], psi[1][ind], linestyle='-', color=colors[k])
        k += 1
    
    ax[2].plot(t, np.ones(len(t)), color=(0.5,0.5,0.5), linestyle='--')

    ### Plot test date interval with a filled bar (if it exists)
    if testdates is not None:
        print(__name__ + '.covid_plot_deconv: Found "testdates" input.')
        t_hot = np.zeros(2)

        for i in range(len(dates)):
            if dates[i] == testdates[0]:
                t_hot[0] = i
        for i in range(len(dates)):
            if dates[i] == testdates[1]:
                t_hot[1] = i

        print(f'testdates: {testdates} [{t_hot}]')
        ax[2].fill_between(t_hot, np.ones(2)*2.5, np.zeros(2), color='black', alpha=0.2) 

    ax[2].set_xlim([0,max(t)])
    ax[2].set_ylabel('$\\psi(t,\\Delta t)$')
    ax[2].set_ylim([0.5,2.5])

    # Create date labels
    labels, positions = aux.date_labels(dates=dates, N=Nx_ticks)
    ax[2].set_xticks(positions)
    ax[2].set_xticklabels(labels, rotation=-70)
    ax[2].set_xlabel(f'$t$ [days]')
    ax[2].legend()

    return fig,ax

def shift_pick(dFdt, dates, testdates, deltaT, adaptive=True):
    """
    Helper function for picking death counts from a time-series.

    Args:
        dFdt:      daily death counts
        dates:     datetime array
        testdates: datetime arrray (start,end)
        deltaT:    dictionary with optimal delay information, or a fixed delay
        adaptive:  input is with adaptive delay information
    
    Returns:
        average death counts over the interval
    """
    deaths = 0

    # Period length
    T = (testdates[1] - testdates[0]).days + 1

    # Loop over the test period, average deaths
    for shift in range(T):

        # Use optimal local delay
        if adaptive:
            D = deltaT[shift]
        # Use fixed global delay
        else:
            D = deltaT

        dday = testdates[0] + datetime.timedelta(shift + D)
    
        for i in range(len(dates)):
            if (dates[i] == dday):

                # Cumulative deaths between [0,i]
                F = np.sum(dFdt[0:i+1])
                deaths += F
                print(__name__ + f".shift_pick: shift = {shift:2g}, DT = {D} | Match at index {i:3g} with SUM_{{T}}(dFdt) = {F}")

    deaths /= T # Average over the test interval
    return deaths


def covid_analyze(CSETS, CSETS_sero, analysis_param, unfold_param, minE=1e-6):
    """
    COVID analysis function.
    
    Args:
        CSETS:          time-series dataset dictionaries
        CSETS_sero:     prevalence test dictionaries
        analysis_param: analysis parameter dictionary
        unfold_param:   unfolding parameters
    
    Returns:
        analysis output dictionary
    """

    PDF = {}
    LLR = {}
    for d in analysis_param['deltas']: PDF[str(d)] = {}
    for d in analysis_param['deltas']: LLR[str(d)] = {}

    deaths = {} # Death counts
    dFF    = {} # NUMERATOR NUISANCE:   Relative uncertainty on optimal death counts
    dPP    = {} # DENOMINATOR NUISANCE: Relative uncertainty on the corrected positive counts

    deltaT = {} # Optimal read-out delays

    # Dataset loop
    for key in CSETS.keys():

        # =================================================================
        ### Process data
        print('')
        
        try:
            metadata = {**CSETS[key], **CSETS_sero[key]}
            print(metadata)

            d = cio.data_processor(metadata)
            d = cio.choose_timeline(d, target_key='deaths', \
                first_date=analysis_param['FIRST_DATE'], last_date=analysis_param['LAST_DATE'])

            print(f'Found dataset <{d["isocode"]}>')
        except:

            print(f"{colored('Failed to process','yellow')} {metadata['isocode']}")
            print(f'Error: {sys.exc_info()[0]} {sys.exc_info()[1]}')
            continue

        # =================================================================
        # Pick the death counts at different dates after the test date

        dates     = aux.get_datetime(d['dt'])
        testdates = aux.get_datetime(metadata['test_date'])

        # =================================================================
        # Extract optimal delays

        deltaT_     = covid_extract_deltaT(d=d, testdates=testdates, unfold_param=unfold_param, \
                        daily_kernels=metadata['kernel_path'], return_full=True)
        # Save it
        deltaT[key] = deltaT_
        
        # Cumulative death counts for each read-out delay
        deaths_ = np.zeros(len(analysis_param['deltas']))

        deaths_lo = 0
        deaths_hi = 0
        
        # Daily deaths
        dFdt = copy.deepcopy(d['deaths'])
        dFdt[~np.isfinite(dFdt)] = 0
        
        # Loop over different fixed delays
        for k in range(len(analysis_param['deltas'])):

            # ============================================================
            # Note the low-high quantile flip, due to inversion

            if analysis_param['deltas'][k] == -1: # Adaptive
                deaths_lo  = shift_pick(dFdt=dFdt, dates=dates, testdates=testdates,\
                deltaT=deltaT_['deltaT']['Q84'], adaptive=True)

                deaths_[k] = shift_pick(dFdt=dFdt, dates=dates, testdates=testdates,\
                deltaT=deltaT_['deltaT']['Q50'], adaptive=True)
                
                deaths_hi  = shift_pick(dFdt=dFdt, dates=dates, testdates=testdates,\
                deltaT=deltaT_['deltaT']['Q16'], adaptive=True)
                
            else:
                deaths_[k] = shift_pick(dFdt=dFdt, dates=dates, testdates=testdates,\
                deltaT=analysis_param['deltas'][k], adaptive=False)
            # ============================================================

            print(__name__ + f'.covid_analyze: With delta = {analysis_param["deltas"][k]} found <F> = {deaths_[k]:0.1f} \n')

        # Save death statistics for this country/city
        deaths[metadata['isocode']] = deaths_

        # Data
        n1 = metadata['population']
        k2 = metadata['positive']
        n2 = metadata['tested']

        # ------------------------------------------------------------
        # Denominator (type I and II test inversion) systematic scale uncertainty

        s  = metadata['specificity']
        ds = metadata['specificity_error']
        
        v  = metadata['sensitivity']
        dv = metadata['sensitivity_error']        

        # Already corrected (inverted) data as input
        if metadata['corrected']:

            # This returns us the new and old (pure binomial) error
            dp_new, dp_orig = estimators.renormalize_test12_error_corrected_input(k=k2, N=n2, s=s,v=v, ds=ds, dv=dv)

        # Do the test error inversion
        else:
            cprint(__name__ + f'.covid_analyze: Positive counts before Type I/II inversion: k2 = {k2}', 'yellow')
            
            # This returns us the new and old (pure binomial) error
            dp_new, dp_orig = estimators.renormalize_test12_error_raw_input(k=k2, N=n2, s=s,v=v, ds=ds, dv=dv)

            # Compute the true prevalence estimate p from the raw counts
            q  = k2 / n2
            p  = estimators.inv_raw2cor(q=q, s=s, v=v)

            # Finally replace
            k2 = np.round(p*n2)
            cprint(__name__ + f'.covid_analyze: Positive counts after  Type I/II inversion: k2 = {k2}', 'yellow')
        
        # We work around the unbiased corrected value of 1 +- relative additional uncertainty
        phat     = k2/n2
        b        = 1.0
        sigma_b  = np.max([np.sqrt((dp_new/phat)**2 - (dp_orig/phat)**2), minE])
        
        # Save it
        dPP[key] = sigma_b

        # ------------------------------------------------------------
        
        # Compute IFR counting uncertainties

        # Loop over fixed delays
        for k in range(len(analysis_param['deltas'])):
            k1     = deaths_[k]

            ## Profile likelihood estimator >>
            r0_val = np.linspace(1e-6, analysis_param['RMAX'], 1000)
            _, r0_dense, LLR_dense, _ = estimators.profile_LLR_binom_ratio_err(k1=k1,n1=n1, k2=k2,n2=n2, \
                r0_val=r0_val, alpha=1-0.95, return_full=True)

            ## Bayesian posterior estimator >>

            # Optimal read-out delay treatment (indicated with -1)
            if analysis_param['deltas'][k] == -1:

                optimal_delay = deltaT_['deltaT']['Q50']
                
                # Loop over test period, compute delay scale relative uncertainty
                dpsi = np.zeros(len(optimal_delay))
                for i in range(len(optimal_delay)):
                    Q16  = deltaT_['psi']['Q16'][int(optimal_delay[i]), i]
                    Q50  = deltaT_['psi']['Q50'][int(optimal_delay[i]), i]
                    Q84  = deltaT_['psi']['Q84'][int(optimal_delay[i]), i]

                    # Symmetrize (lower and upper) and compute relative uncertainty
                    dpsi = np.mean([np.abs(Q16 - Q50), np.abs(Q84 - Q50)]) / Q50
                    print(f'Test day {i:2g} [optimal deltaT = {optimal_delay[i]}]: psi(t,deltaT) = [{Q16:0.6f}, {Q50:0.6f}, {Q84:0.6f}], d(psi)/psi = {dpsi:0.2E}')

                # Symmetrize (lower and upper) and compute relative uncertainty
                dFF[key] = np.mean([np.abs(deaths_lo-deaths_[k]), np.abs(deaths_hi-deaths_[k])]) / deaths_[k]
                print(f'\nDeaths at [t + deltaT]: {deaths_[k]:0.1f} [{deaths_lo:0.1f}, {deaths_hi:0.1f}], d(F)/F = {dFF[key]:0.2E} \n')

                # --------------------------------------------------------
                # k1 (deaths counts) have been already optimized to be with scale (psi(t,\Delta t) \approx 1

                a       = 1.0 # We work around the unbiased value
                sigma_a = np.max([dFF[key], minE])
                if (sigma_a < minE or ~np.isfinite(sigma_a)): sigma_a = minE
                
                print(__name__ + f'.covid_analyze: Read-out delay nuisance scale (mu,std) = ({a:0.3E}, {sigma_a:0.3E})')
                
            # Fixed delays
            # no nuisance parameters estimated
            else:
                a       = None
                sigma_a = None
            # ------------------------------------------------------------

            # Compute Bayesian posterior
            CI_B = estimators.bayes_binomial_ratio_err(rmax=analysis_param['RMAX'], 
                k1=k1,n1=n1, k2=k2,n2=n2, a=a, sigma_a=sigma_a, b=b, sigma_b=sigma_b, \
                prior1='Jeffrey', prior2='Jeffrey', CL=analysis_param['qbayes'], renorm=True)
            
            # Save them
            PDF[str(analysis_param['deltas'][k])][key] = CI_B
            LLR[str(analysis_param['deltas'][k])][key] = {'val': r0_dense, 'llr': LLR_dense}

    obj = {
        'CSETS':      CSETS,      # Death count configurations
        'CSETS_sero': CSETS_sero, # Serology configurations
        'PDF':        PDF,        # Bayesian posteriori densities
        'LLR':        LLR,        # Likelihood ratios
        'deaths':     deaths,     # Running death counts
        'dFF':        dFF,        # Relative systematic uncertainty on death counts
        'dPP':        dPP,        # Relative systematic uncertainty on positive test counts
        'deltaT':     deltaT      # Fixed death count read-out delays used
    }

    return obj


def covid_print_stats(obj, dprint, print_deltaT=True, print_psi=False, print_dFF=True, print_dPP=True, psi_arg_deltaT=7):
    """
    Print out analysis statistics.
    
    Args:
        obj            : analysis output dictionary
        dprint         : output printing function handle
        print_deltaT   : print optimal $\\Delta t$ values
        print_psi      : print psi-function values
        print_dFF      : print death count relative systematic uncertainty 
        print_dPP      : print positive test count relative systematic uncertainty
        psi_arg_deltaT : psi-function $\\Delta t$ argument
    """

    CSETS      = obj['CSETS']
    CSETS_sero = obj['CSETS_sero']
    PDF        = obj['PDF']
    LLR        = obj['LLR']
    deaths     = obj['deaths']
    dFF        = obj['dFF']
    dPP        = obj['dPP']
    deltaT     = obj['deltaT']

    ### Descriptions
    dprint('')
    dprint('Dataset & Prevalence test period & Type & Age \\\\')
    dprint('\\hline')
    for key in CSETS.keys():
        metadata = {**CSETS[key], **CSETS_sero[key]}
        dprint(f"{metadata['region']} ({key}) \\cite{{{metadata['test_tex']}}} & [{metadata['test_date'][0]}, {metadata['test_date'][1]}] & {metadata['test_type']} & {metadata['age']} \\\\")

    # --------------------------------------------------------------------

    deltas = []
    for key in PDF.keys():
        deltas.append(key)

    ### Data counts
    dprint('')
    dprint('Dataset & Positive & Tested & Prevalence & Population', end='')
    for i in range(len(deltas)):
        dprint(f' & Fatal $\\Delta t={deltas[i]}$', end='')
    dprint(' \\\\')
    dprint('\\hline')

    for key in CSETS.keys():
        metadata = {**CSETS[key], **CSETS_sero[key]}
        dprint(f"{key} & {metadata['positive']:0.0f} & {metadata['tested']:0.0f} & {metadata['positive']/metadata['tested']:0.1E} & {metadata['population']}", end="")

        for value in deaths[key]:
            dprint(f' & {value:0.0f}', end='')

        dprint(' \\\\ ')
    dprint('')

    ### IFR estimates
    prc = 100 # Percent
    dprint('Dataset', end='')
    for i in range(len(deltas)):
        dprint(f' & IFR $\\Delta t={deltas[i]}$', end='')

    dprint(' \\\\')

    for key in CSETS.keys():
        dprint(f'{key}', end="")
        metadata = {**CSETS[key], **CSETS_sero[key]}

        for i in range(len(deaths[key])):
            index  = str(deltas[i])

            # Maximum Likelihood
            IFR_ML = (deaths[key][i] / metadata['population']) / (metadata['positive'] / metadata['tested'])
            # Mean
            IFR    = scipy.integrate.simps(x=PDF[index][key]['val'], y=PDF[index][key]['pdf']*PDF[index][key]['val'])
            # Credible intervals
            CR     = PDF[index][key]['CR_value']

            dprint(f' & {IFR*prc:0.2f} [{CR[0]*prc:0.2f}, {CR[-1]*prc:0.2f}]', end='')
        dprint(' \\\\')
    dprint('\\hline')
    dprint('')

    # --------------------------------------------------------------------
    ### Nuisance parameter encapsulated uncertainties

    dprint('Dataset', end='')
    if print_deltaT: dprint(f' & $\\Delta t \\leftarrow \\psi(t,\\Delta t)$ (CI68)', end='')
    if print_dFF:    dprint(f' & $\\delta \\gamma$ [%] (CI68)', end='')
    if print_psi:    dprint(f' & $\\psi(t, \\Delta t = {psi_arg_deltaT:0.0f})$ (CI68)', end='')
    if print_dPP:    dprint(f' & $\\delta \\lambda$ [%] (CI68)', end='')
    dprint(' \\\\')
    
    for key in CSETS.keys():
        dprint(f'{key}', end="")
        metadata = {**CSETS[key], **CSETS_sero[key]}

        # These are arrays, because they contain $\Delta t$ for
        # each measurement day of the test period.

        if print_deltaT:
            dt_lo   = np.mean(deltaT[key]['deltaT']['Q84'])
            dt_med  = np.mean(deltaT[key]['deltaT']['Q50'])
            dt_hi   = np.mean(deltaT[key]['deltaT']['Q16'])

            dprint(f' & {dt_med:0.2g} [{dt_lo:0.2g}, {dt_hi:0.2g}]', end='')

        if print_dFF:
            dprint(f' & {dFF[key] * 100:0.2g}', end='') # Relative systematic uncertainty print

        if print_psi:
            psi_lo  = np.mean(deltaT[key]['psi']['Q16'][psi_arg_deltaT,:])
            psi_med = np.mean(deltaT[key]['psi']['Q50'][psi_arg_deltaT,:])
            psi_hi  = np.mean(deltaT[key]['psi']['Q84'][psi_arg_deltaT,:])

            dprint(f' & {psi_med:0.2g} [{psi_lo:0.2g}, {psi_hi:0.2g}]', end='')

        if print_dPP:
            dprint(f' & {dPP[key] * 100:0.2g}', end='') # Relative systematic uncertainty print

        dprint(' \\\\')
    dprint('\\hline')


