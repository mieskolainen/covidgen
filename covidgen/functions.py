# Time domain response functions
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import numba
import scipy
import copy
from scipy.integrate import trapz
from scipy.integrate import simps

import tools


def gamma_pdf(x, k, theta):
    """ Gamma pdf density.
    
    Args:
        x     : input argument
        k     : shape > 0
        theta : scale > 0

    Returns:
        pdf values
    """
    xx = copy.deepcopy(x)
    xx[x < 0] = 0
    y = 1.0/(scipy.special.gamma(k)*theta**k) * (xx**(k-1)) * np.exp(-xx/theta)

    return y

@numba.njit
def normpdf(x,mu,std):
    """ Normal pdf
    Args:
        x   : array of argument values
        mu  : mean value
        std : standard deviation
    Returns:
        density values for each x
    """
    return 1/np.sqrt(2*np.pi*std**2) * np.exp(-(x-mu)**2/(2*std**2))


#@numba.njit
def h_exp(t, a, normalize=False):
    """ Exponential density
    Args:
        t:         input argument (array)
        a:         parameter > 0
        normalize: trapz integral normalization over t
    Returns:
        function values
    """
    y = np.zeros(len(t))
    y[t>0] = np.exp(-t[t>0] / a) / a

    y[np.isinf(y) | np.isnan(y)] = 0 # Protect underflows
    if normalize:
        y /= np.abs(trapz(x=t, y=y)) # abs for numerical protection
    return y

#@numba.njit
def h_wei(t, a, k, normalize=False):
    """ Weibull density
    Args:
        t:         input argument (array)
        a:         scale parameter > 0
        k:         shape parameter > 0
        normalize: trapz integral normalization over t
    Returns:
        function values
    """
    y = np.zeros(len(t))
    y[t>0] = (k/a) * (t[t>0]/a)**(k-1) * np.exp(-(t[t>0]/a)**k)

    y[np.isinf(y) | np.isnan(y)] = 0 # Protect underflows
    if normalize:
        y /= np.abs(trapz(x=t, y=y)) # abs for numerical protection
    return y

#@numba.njit
def h_lgn(t, mu, sigma, normalize=False):
    """ Log-normal density
    Args:
        t:         input argument (array)
        mu:        mean parameter (-infty,infty)
        sigma:     std parameter > 0
        normalize: trapz integral normalization over t
    Returns:
        function values
    """
    y = np.zeros(len(t))
    y[t>0] = 1/(t[t>0]*sigma*np.sqrt(2*np.pi)) * np.exp(-(np.log(t[t>0]) - mu)**2 / (2*sigma**2))

    y[np.isinf(y) | np.isnan(y)] = 0 # Protect underflows
    if normalize:
        y /= np.abs(trapz(x=t, y=y)) # abs for numerical protection
    return y


def I_log(t, i0, beta, L):
    """ Fixed beta logistic equation solution
    
    Args:
        t    : time
        i0   : initial condition
        beta : fixed growth rate
        L    : solution maximum
    
    """
    return L / (1 - (1-L/i0)*np.exp(-beta*t))


def dIdt_log(t, i0, beta, L):
    """ Fixed beta logistic equation solution time derivative
    """
    return (-i0*np.exp(-beta*t)*(i0-L)*L*beta) / (i0 + np.exp(-beta*t)*(L-i0))**2


def I_log_running(t, i0, L, beta, beta_param):
    """ Running beta logistic equation solution
    """
    beta_   = beta(t, **beta_param)
    betaint = np.zeros(len(t))
    for i in range(1,len(t)):
        tval = t[0:i]
        betaint[i] = simps(x=tval, y=beta(tval, **beta_param))

    return (-i0*L) / ((i0-L)*np.exp(-betaint) - i0)


def dIdt_log_running(t, i0, L, beta, beta_param):
    """ Running beta logistic equation solution time derivative
    """
    beta_   = beta(t, **beta_param)
    betaint = np.zeros(len(t))
    for i in range(1,len(t)):
        tval = t[0:i]
        betaint[i] = simps(x=tval, y=beta(tval, **beta_param))

    return (-i0*np.exp(-betaint)*(i0-L)*L*beta_) / (i0 + np.exp(-betaint)*(L-i0))**2


def betafunc(t, beta_0, beta_D, beta_lambda):
    """ Running effective beta-function
    """
    y = np.ones(len(t))
    for i in range(len(t)):
        if t[i] < beta_D:
            y[i] = beta_0
        else:
            y[i] = beta_0 * np.exp(-(t[i] - beta_D)/beta_lambda)
    y[y < 0] = 0
    return y

