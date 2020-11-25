# Binomial proportion confidence interval estimators and others
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import numba
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import ot
from   termcolor import colored,cprint


# Large numbers
import mpmath
from mpmath import mp

import scipy
from scipy.special import factorial
from scipy.special import comb
from scipy.stats import beta as betapdf
from scipy.stats import binom

from scipy.integrate import simps

from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.integrate import tplquad

from scipy.interpolate import interp1d
from scipy.optimize import bisect

import scipy.stats as stats
from scipy.stats import norm

# Own
import aux
import tools
import functions

# -------------------------------------------------------------------
# Percentile/Quantile definitions

# Between [0,100]
Q68 = np.array([16, 84])
Q90 = np.array([5, 95])
Q95 = np.array([2.5, 97.5])

# Between [0,1]
q68 = copy.deepcopy(Q68) / 100
q90 = copy.deepcopy(Q90) / 100
q95 = copy.deepcopy(Q95) / 100

q68_q95 = np.array([q95[0], q68[0], q68[1], q95[1]])

# Normal distribution z-scores
z68 = 1.00 
z95 = 1.96
# -------------------------------------------------------------------

def wass_combine(xval, A, reg=3e-3, weighted=False, w=None, algorithm='quantile'):
    """
    Wasserstein barycenter 1D-PDF fusion (combination) estimator
    
    Note that algorithm 'bregman' from OT toolbox may produce unnaturally wide distribution,
    i.e. oversmoothed due to strong regularization.
    
    Args:
        xval      : x-axis values (n)
        A         : array of pdfs with (n x K components)
        reg       : entropic regularization parameter (only for 'bregman')
        weighted  : True or False
        w         : array of weigths (for example 1/sigma**2) (K)
        algorithm : 'bregman' or 'quantile'
    
    Returns:
        combined pdf
    """

    n = A.shape[0] # number of bins
    n_distributions = A.shape[1]

    ### Loss matrix + normalization
    # dist0 uses standard quadratic transport loss
    M  = ot.utils.dist0(n)
    M /= M.max()

    # Barycenter computation weights
    if   weighted is False:
        weights  = np.ones(n_distributions) / n_distributions # 0 <= alpha_i <= 1
    else:
        weights  = w
        weights /= np.sum(weights) # normalize sum to 1

    # Wasserstein via OT toolbox
    if   algorithm == 'bregman':
        g = ot.bregman.barycenter(A, M, reg, weights)

    # Via quantile functions
    elif algorithm == 'quantile':
        g = frechet_mean(PDF=A, xval=xval, w=w)

    else:
        raise Except(__name__ + f'.wass_combine: ERROR: Unknown algorithm {algorithm}')

    '''
    # Get transport matrix
    gamma = np.zeros((n,n))
    fig, ax = plt.subplots()
    for i in range(A.shape[1]):
        f = copy.deepcopy(A[:,i])
        gamma += ot.sinkhorn(g, f, M, reg)  
    '''

    # Normalize to continuum pdf
    g /= scipy.integrate.simps(x=xval, y=g)
    
    return g


def frechet_mean(PDF, xval, w=None, EPS=1E-15):
    """
    Quantile function based Frechet (Wasserstain) mean of 1D PDFs.
    
    Args:
        PDF     : Discretized densities with (number of points x number of PDFs)
        xval    : x-value array
        w       : Weights for the weighted estimate (default None)
    
    Returns:
        Optimal transported solution
    """

    N = PDF.shape[1]

    if   w is None:
        weights  = np.ones(N) / N
    else:
        weights  = w
        weights /= np.sum(weights) # normalize sum to 1

    zval = np.linspace(0, 1, PDF.shape[0])
    G    = np.zeros((len(zval), N))
    
    # Compute inverse CDF for each pdf
    for i in range(N):
        f    = PDF[:,i]
        f[f < EPS] = EPS # improve stability
        
        cdf  = tools.cdfint(xval, f)
        zval = np.linspace(0, np.max(cdf), PDF.shape[0])

        invcdf = lambda x : interp1d(cdf, xval, kind='linear', fill_value='extrapolate')(x)
        G[:,i] = invcdf(zval)

    # (Weighted) sum of inverse CDFs
    sG = np.zeros(PDF.shape[0])
    for i in range(N):
        sG += G[:,i] * weights[i]

    # Invert
    Fnew = lambda x : interp1d(sG, zval, kind='linear', fill_value='extrapolate')(x)
    sPDF = Fnew(xval)

    # Differentiate and normalize
    sPDF = np.diff(sPDF, prepend=0)
    sPDF[sPDF < 0] = 0
    sPDF /= scipy.integrate.simps(x=xval, y=sPDF)

    return sPDF


def mom_combine(r, s2, N=10, debug=False):
    """ Meta-analysis / combination analysis
        methods of moments estimator of type improved DerSimonian-Laird
    Args:
        r      : values of the parameters in each K studies
        s2     : variance estimates in each K studies
        N      : number of iterations
    
    Returns:
        rhat      : global parameter estimate
        delta2    : global variance estimate 
        w         : weights (K x 1)
        rhat_err  : standard error given by se(rhat) = 1/sqrt(np.sum(w)) (Wald test like)
    """

    def est(w):

        # Weighted estimate
        rhat = np.sum(w*r) / np.sum(w)

        # Test statistic (approximately ~ chi^2(ndf=k-1) under H0: r_i = r for all i)
        Q = np.sum(w*(r - rhat)**2)

        # Between study variance
        val = (Q - np.sum(w*s2) + np.sum(w**2*s2) / np.sum(w)) / \
            (np.sum(w) - np.sum(w**2) / np.sum(w))
        delta2 = np.max([0, val])

        return rhat, delta2

    # Iterate
    delta2 = 0
    for i in range(N):
        w = 1.0/(s2 + delta2)
        rhat, delta2 = est(w=w)
        if debug:
            print(__name__ + f'.mom_combine: iter {i}: rhat = {rhat:0.3}, delta2 = {delta2:0.3}')

    # Standard error
    rhat_err = np.sqrt(1.0/np.sum(w))

    return rhat, delta2, w, rhat_err


def joint_likelihood_combine(x2LL, alpha1 = 0.32, alpha2 = 0.05, return_full=False):
    """
    Combine independent log-likelihood ratios with a sum (~product likelihood).
    
    Args:
        x2LL           : dictionary of 2xlog-likelihood value with keys ('val', 'llr')
        alpha1, alpha2 : confidence levels (default CI68, CI95)
        return_full    : return full information
    
    Returns:
        maximum likelihood estimate, quantile points
    """

    # Combine log-likelihood ratios by summing them
    x2LL_sum  = np.sum([x2LL[item]['llr'] for item in x2LL], axis=0)

    # x-axis values (must be the same for all datasets, take the first one)
    r0_value = [x2LL[item]['val'] for item in x2LL][0]
    
    # Find minimum 
    x2LL_min = np.min(x2LL_sum)
    minind   = np.argmin(x2LL_sum)
    IFR      = r0_value[minind]

    # Find first interval
    chi2 = stats.chi2.ppf(q=1-alpha1, df=1)
    ind  = (np.where(x2LL_sum - x2LL_min <= chi2))[0] # Note <= not <
    CI68 = np.array([r0_value[ind[0]], r0_value[ind[-1]]])
    
    # Find second interval
    chi2 = stats.chi2.ppf(q=1-alpha2, df=1)
    ind  = (np.where(x2LL_sum - x2LL_min <= chi2))[0] # Note <= not <
    CI95 = np.array([r0_value[ind[0]], r0_value[ind[-1]]])

    CI_val = np.array([CI95[0], CI68[0], CI68[1], CI95[1]])

    if not return_full:
        return IFR, CI_val
    else:
        return IFR, CI_val, x2LL_sum, r0_value


def chi2range(x2LL, x2LL_hat, alpha=0.05, ndf=1):
    """
    Find 1D-confidence interval based on 2xlog-likelihood values and chi2 asymptotics.
    """
    chi2       = stats.chi2.ppf(1 - alpha, df=ndf)
    ind        = (np.where(x2LL > x2LL_hat - chi2))[0] # Note >
    return ind


@numba.njit
def nnl_2logL(rhat, delta2, r, s2, EPS=1E-15):
    """
    Normal-Normal model log-likelihood function: ln L(r,delta2; r_j)
    
    (see e.g. Hardy, Thomson,
        A likelihood approach to meta-analysis with random effects, 1996)

    Args:
        rhat   : parameter r
        delta2 : parameter r
        r      : measured values of the parameters in each K studies
        s2     : measured variances of the parameters in each K studies
    
    Returns:
        2 * log-likelihood function value
    """
    A = 2*np.pi*(s2 + delta2)
    A = np.maximum(A, EPS)
    
    return 2*(-np.sum(0.5*np.log( A )) \
               -np.sum((r-rhat)**2 / (2*(s2 + delta2))) )


def nnl_combine(r, s2, N=10, debug=False, alpha=0.05, rang=5, rang_N=1000, EPS=1E-15):
    """ Meta-analysis / combination estimator based on
        Normal-Normal likelihood construction [see also nnl_logL() function]
    
    Args:
        r      : values of the parameters in each K studies
        s2     : variance estimates in each K studies
        N      : number of iterations
        alpha  : confidence level
        rang   : +- range for the uncertainty interval scans
        N      : number of points in the profile likelihood scan
    
    Returns:
        rhat   : global parameter estimate
        delta2 : global variance estimate
    """

    def est(rhat, delta2):

        # .maximum(,) operates element wise
        rhat   = np.sum(r / np.maximum(s2 + delta2, EPS)) / \
                (np.sum(1.0 / np.maximum(s2 + delta2, EPS)))
        delta2 = np.sum(((r - rhat)**2 - s2) / np.maximum(s2 + delta2, EPS)**2) / \
                 np.sum(1.0 / np.maximum(s2 + delta2, EPS)**2)

        # protection
        rhat   = np.abs(rhat)
        delta2 = np.abs(delta2)

        return rhat, delta2

    # Iterate
    rhat   = np.mean(r)
    delta2 = np.var(r)
    for i in range(N):
        rhat, delta2 = est(rhat,delta2)
        if debug:
            _2logL = nnl_2logL(rhat=rhat, delta2=delta2, r=r, s2=s2)
            print(__name__ + f'.nnl_combine: iter {i}: rhat = {rhat:0.3}, delta2 = {delta2:0.3} | 2lnL = {_2logL:0.5e}')
    
    # ....................................................................
    # Profiled log-likelihood ratio for the uncertainties

    x2LL_hat = nnl_2logL(rhat=rhat, delta2=delta2, r=r, s2=s2)
    
    ### RHAT
    theta_val = np.linspace(rhat/rang, rhat*rang, rang_N)
    x2LL = np.zeros(len(theta_val))
    for i in range(len(x2LL)):
        x2LL[i] = nnl_2logL(rhat=theta_val[i], delta2=delta2, r=r, s2=s2)

    ind        = chi2range(x2LL=x2LL, x2LL_hat=x2LL_hat, alpha=alpha)
    
    if len(ind) >= 2:
        rhat_err = np.array([theta_val[ind[0]], theta_val[ind[-1]]])
    else:
        cprint(__name__ + f'.nnl_combine: WARNING: Could not obtain uncertainty on rhat, setting zero', 'red')
        rhat_err = np.array([0,0])
    
    # --------------------------------------------------------------------
    ### DELTA2
    theta_val = np.linspace(delta2/rang, delta2*rang, rang_N)
    x2LL = np.zeros(len(theta_val))
    for i in range(len(x2LL)):
        x2LL[i] = nnl_2logL(rhat=rhat, delta2=theta_val[i], r=r, s2=s2)

    ind        = chi2range(x2LL=x2LL, x2LL_hat=x2LL_hat, alpha=alpha)

    if len(ind) >= 2:
        delta2_err = np.array([theta_val[ind[0]], theta_val[ind[-1]]])
    else:
        cprint(__name__ + f'.nnl_combine: WARNING: Could not obtain uncertainty on delta2, setting zero', 'red')
        delta2_err = np.array([0,0])

    # Check for NaN
    if np.isnan(rhat):
        cprint(__name__ + f'.nnl_combine: WARNING: rhat is NaN, setting zero', 'red')
    if np.isnan(delta2):
        cprint(__name__ + f'.nnl_combine: WARNING: delta2 is NaN, setting zero', 'red')

    return rhat, delta2, rhat_err, delta2_err


def poisson_significance(s,b):
    """ Poisson statistics based significance
    in the order of large N (Gaussian limit) -> s / sqrt(b)

    Args:
        s : signal
        b : background
    Returns:
        statistical significance
    """
    return np.sqrt(2*((s+b)*np.log(1+s/b)-s))


def norm_icdf(x):
    """ Inverse standard normal CDF.
    """
    return norm.ppf(x)


def contour_cumsum(logZ):
    """ Map a (log-likelihood) like density 2D array to cumulative integral.
    """
    Z = np.exp(logZ)

    shape = Z.shape
    Z = Z.ravel()

    ind_sort   = np.argsort(Z)[::-1]
    ind_unsort = np.argsort(ind_sort)

    Z_cumsum  = Z[ind_sort].cumsum()
    Z_cumsum /= Z_cumsum[-1]

    return Z_cumsum[ind_unsort].reshape(shape)


def binomC(k,n):
    """ C(n,k) [ n!/(k!(n-k)!) ] coefficient
    """
    return np.double( comb(n, k, exact=1) )


@numba.njit
def binom_pdf(k, n,p,binom):
    """ Binomial Likelihood pdf(k | n,p)
    """
    return binom * p**k * (1-p)**(n-k)


def beta(a,b, precision=50):
    """ Euler Beta function.
    """
    mp.dps = precision
    #return float( mpmath.gamma(a) * mpmath.gamma(b) / mpmath.gamma(a+b) )
    return float(mpmath.beta(a,b))


def betabinom_B(k,n, alpha,beta, precision=50):
    """ Beta function (not distribution) for Beta-Binomial posterior.
    """
    mp.dps = precision # Set precision
    #return float( mpmath.gamma(k+alpha) * mpmath.gamma(n-k+beta) / mpmath.gamma(alpha+n+beta) )
    return float(mpmath.beta(k+alpha, n-k+beta))

#@numba.njit
def binom_post(p, k,n, alpha,beta):
    """ Binomial Posteriori density P(p | k,n) using a Beta(alpha,beta) prior,
    which is equal to Beta(k+alpha, n−k+beta) density
    """
    f = betapdf.pdf(x=p, a=k+alpha, b=n-k+beta)
    return f

#@numba.njit
def binom_post_2D(p1,p2, k1,n1, k2,n2, alpha1,beta1, alpha2,beta2):
    """ Two independent binomial posteriori pdf(p1,p2 | k1,n1, k2,n2).
    """
    f = binom_post(p=p1, k=k1,n=n1, alpha=alpha1,beta=beta1) * \
        binom_post(p=p2, k=k2,n=n2, alpha=alpha2,beta=beta2)
    return f

#@numba.njit
def binom_post_alt(p, k,n, B, alpha,beta):
    """ Binomial Posteriori density P(p | k,n) using a Beta(alpha,beta) prior,
    which is equal to Beta(k+alpha, n−k+beta) density.

    [Function with pre-computed Beta function B]
    """
    return p**(k+alpha-1) * (1 - p)**(n-k+beta-1) / B

#@numba.njit
def binom_post_2D_alt(p1,p2, k1,n1, k2,n2, B1, B2, alpha1,beta1, alpha2,beta2):
    """ Two independent binomial posteriori pdf(p1,p2 | k1,n1, k2,n2).
    
    [Function with pre-computed Beta functions B1,B2]
    """
    f = binom_post_alt(p=p1, k=k1,n=n1, B=B1, alpha=alpha1,beta=beta1) * \
        binom_post_alt(p=p2, k=k2,n=n2, B=B2, alpha=alpha2,beta=beta2)
    return f

@numba.njit
def binom_ratio_func(y,z, k1,n1, k2,n2, alpha1,beta1, alpha2,beta2):
    """  Define ratio integral f(z) in terms of the joint posteriori distribution P(X,Y).
    Based on a change of variables.
    """
    return y * binom_post_2D(p1=z*y, p2=y, k1=k1,n1=n1, k2=k2,n2=n2, \
        alpha1=alpha1,beta1=beta1, alpha2=alpha2,beta2=beta2)


def binom_err_midp(k,n, CL = np.array([0.025, 0.975])):
    """ Lancaster mid-P(probability) modified
    binomial uncertainty confidence intervals.
    
    Example:
        find parameter theta such that
        p_0 + p_1 + ... + p_{k-1} + 1/2 * p_k = 0.025 or 0.975
    """

    out = np.zeros(CL.size)
    for j in range(len(out)):

        def f(p):
            psum   = 0.0
            factor = 1.0
            for i in range(0,k+1):
                if i == k:
                    factor = 1/2
                psum += factor * binom.pmf(k=i,n=n, p=p)
            return psum

        def rootwrap(p):
            return f(p) - (1-CL[j]) # Note one minus

        res    = bisect(rootwrap, a=k/n/10, b=k/n*10)
        out[j] = res

    return out


@numba.njit(parallel=False)
def binom_err(k,n, z = 1.0):
    """ Normal (Wald) approximation interval for binomial uncertainty.
    """
    p = k/n # mean
    err = np.sqrt(p*(1-p) / n)
    return np.array([p - z*err, p + z*err])


@numba.njit(parallel=False)
def wilson_err(k,n, z = 1.0):
    """ Wilson score interval for binomial uncertainty.
    """
    p = k/n # mean

    centre_adj_p = p + z*z/(2*n)
    denom        = 1 + z*z/n
    adj_std      = np.sqrt((p*(1 - p) + z*z / (4*n)) / n)
    
    lower = (centre_adj_p - z*adj_std) / denom
    upper = (centre_adj_p + z*adj_std) / denom

    return np.array([lower, upper])


def wilson_cc_err(k,n, z=1.0):
    """ Wilson score interval for binomial uncertainty
     + continuity correction
    
    Newcombe, R. G. (1998).
    "Two-sided confidence intervals for the single proportion: comparison of seven methods".
    """
    p = k/n # mean

    denom  = 2*(n + z*z)
    numer  = z * np.sqrt(z**2 - 1/n + 4*n*p*(1-p) + (4*p-2)) + 1
    common = 2*n*p + z*z
    
    lower = np.max([0.0, (common - numer)/denom])
    upper = np.min([1.0, (common + numer)/denom])

    return np.array([lower, upper])


def clopper_pearson_err(k, n, CL=[0.025, 0.975]):
    """ Clopper-Pearson binomial proportion confidence interval.
    Below, beta.ppf (percent point functions) returns inverse CDF for quantiles.
    
    Args:
        k  : observed success counts
        n  : number of trials
        CL : confidence levels
    Returns:
        corresponding interval points
    """
    # Special case must be treated separately
    if   k == 0:
        lower = 0
        upper = 1 - (1-CL[1])**(1/n)

    # Special case must be treated separately
    elif k == n:
        lower = CL[0]**(1/n)
        upper = 1

    # Normal case
    else:
        lower = stats.beta.ppf(q=CL[0], a=k, b=n-k+1)
        upper = stats.beta.ppf(q=CL[1], a=k+1, b=n-k)

    return np.array([lower, upper])


@numba.njit
def LLR_binom(k, n, p0, EPS=1E-15):
    """ Log likelihood ratio test statistic for the single binomial pdf.
    Args:
        k  :  number of counts (numpy array)
        n  :  number of trials
        p0 :  null hypothesis parameter value
    Returns:
        individual log-likelihood ratio values
    """
    phat = k/n # maximum likelihood estimate
    phat[phat < EPS] = 2*EPS

    # Log-likelihood (density) ratios
    LLR = 2*( (k*np.log(phat)+(n-k)*np.log(1-phat)) - (k*np.log(p0)+(n-k)*np.log(1-p0)))
    return LLR


@numba.njit
def LLR_poisson(k, mu0, b=0, EPS=1e-15):
    """ Log likelihood ratio test statistic for
    the poisson pdf with a known background b.
    Args:
        k   :  number of observed counts (signal + background) (numpy array)
        mu0 :  null hypothesis Poisson parameter value
        b   :  background counts mean value
    Returns:
        individual log-likelihood ratio values
    """
    muhat = k - b # maximum likelihood non-negative truncated estimate
    muhat[muhat < EPS] = 2*EPS

    # Log-likelihood (density) ratios
    # -2[ Log(l(mu_0)) - Log(l(muhat))]
    LLR = 2*(mu0 - muhat + k*np.log((b+muhat) / (b+mu0)))
    return LLR

def beltscan_binom_err(k, n=0, b=0, theta0=None, alpha=0.05, MC=100000, 
        rang=10, mode='exact-LLR', stat='binom', return_full=False, EPS=1E-15):
    """
    Neyman belt scan for the binomial proportion (or Poisson) uncertainty.
    
    1. Loop over discretized theta0, which is the parameter of interest.
    2. For each theta0 point, generate a toy MC sample from binom distribution, calculate ordering statistic (e.g. LLR) for each.
    3. Compute threshold quantiles, go back to 1.
    4. Cross the obtained band vertically (= invert the (test) statistic) at the observed point.
    
    Ref: Spjotvoll, "Unbiasedness of Likelihood Ratio Confidence Sets in Cases without Nuisance Parameters", 1971 and references there.
         Feldman, Cousins, "A Unified Approach to the Classical Statistical Analysis of Small Signals", 1998
        
           ^ 
    theta0 |        --------------
           |       ------------
           x      *---------
           |     -|-------
           |    --|----  
           |   ---|-- 
           x  ----*
           |----  |
           |----------------------->
                 observed         k
    
    Args:
        k      : observed counts (can be a numpy array)
        n      : number of trials (for stat = 'binom' only)
        b      : known background (for stat = 'poisson' only)
        theta0 : the discretized parameter array, otherwise generated automatically
        alpha  : confidence level
        MC     : number of toys used to sample the PDF
        rang   : parameter range (not used if theta0 is not None)
        mode   : 'exact-LLR' (numerical quantile LLR), 'asymp-LLR' (asymptotic chi2 LLR),
                      'cPDF' (Clopper-Pearson like central pdf)
        stat   : 'binom' or 'poisson'
        return_full : return all generated data
    """

    if hasattr(k, "__len__") == False:
        k = np.array([k])

    if theta0 is None:
        if stat == 'binom':
            theta_hat = np.max(k)/n
            a = np.max([EPS, theta_hat/rang])
            b = np.min([1 - EPS, theta_hat*rang])

        elif stat == 'poisson':
            a = np.max([EPS, k/rang])
            b = np.min([1 - EPS, k*rang])

        theta0 = np.linspace(a,b, 1000)
        print(f'beltscan_binom_err: scanning parameter range = [{a:0.2E}, {b:0.2E}]')

    k_maxval = np.zeros(len(theta0))
    k_minval = np.zeros(len(theta0))
    delta    = np.zeros(len(theta0))

    if mode == 'cPDF':
        delta = np.zeros((len(theta0),2))

    # Loop over each parameter theta value
    b_toy = np.random.poisson(b, size=MC)
    for i in tqdm(range(len(theta0))):
        
        # 1. Draw toy MC ~ Binom(n, theta0)
        if   stat == 'binom':
            k_toy = np.random.binomial(n, theta0[i], size=MC)
            t     = LLR_binom(k=k_toy, n=n, p0=theta0[i])
        elif stat == 'poisson':
            k_toy = np.random.poisson(theta0[i], size=MC) + b_toy
            t     = LLR_poisson(k=k_toy, mu0=theta0[i], b=b)
        else:
            raise Exception(f'LLR_binom_err: unknown parameter stats = {stats}')

        # 2 and 3. Compute test statistic and the local threshold value
        if   mode == 'exact-LLR':
            # Exact ratio distribution quantile, instead of asymptotic chi2-based
            # Interpolate to higher value if between two discrete values, to be conservative
            delta[i] = np.percentile(t, 100*(1-alpha), interpolation='higher')
            
        elif mode == 'asymp-LLR':
            # Use asymptotic (large N) result of the likelihood ratio test
            delta[i] = stats.chi2.ppf(q=1-alpha, df=1)

        elif mode == 'cPDF':
            # Central pure PDF ordering based on quantiles
            delta[i,:] = np.array([ np.percentile(k_toy, 100*(alpha/2),   interpolation='lower'),\
                                    np.percentile(k_toy, 100*(1-alpha/2), interpolation='higher') ])

        else:
            raise Exception(__name__ + f': Unknown parameter mode = {mode}')

        # 4. Extract "horizontal" endpoint values based on MC sample points
        #
        # acceptance_set(theta_0) = [k_\min(theta_0), k_\max(theta_0)]
        #
        if mode == 'exact-LLR' or mode == 'asymp-LLR':
            ind         = (np.where(t <= delta[i]))[0] # Note <= not <
            k_minval[i] = np.min(k_toy[ind])
            k_maxval[i] = np.max(k_toy[ind])
        
        if mode == 'cPDF':
            k_minval[i] = np.min(k_toy[k_toy >= delta[i,0]]) # Note <= not <
            k_maxval[i] = np.max(k_toy[k_toy <= delta[i,1]])

    # 5. Extract the confidence interval in "vertical" direction
    # (we find here only the set extremum endpoints, not full set)
    # 
    # confidence_interval(\theta; k) = [theta_\min, theta_\max]
    # 
    # Use here custom arg_min and arg_max to have explicit control over < and <= operators
    min_ind = np.zeros(len(k), dtype=int)
    max_ind = np.zeros(len(k), dtype=int)
    CI      = np.zeros((len(k),2))

    for i in range(len(k)):
        min_ind[i] = tools.arg_min(x=k[i], values=k_minval)
        max_ind[i] = tools.arg_max(x=k[i], values=k_maxval)
        CI[i,:] = [theta0[max_ind[i]], theta0[min_ind[i]]]

    # Squeeze dimensions (for len(k) == 1)
    CI = CI.squeeze()

    if return_full:
        return CI, k_minval, k_maxval, theta0, delta
    else:
        return CI


def llr_binom_err(k, n, p0=None, alpha=0.05, EPS=1E-15, rang=10):
    """ Likelihood ratio test based binomial proportion uncertainty.
    [ASYMPTOTIC approximation]
    """
    phat = np.max([EPS, k/n])
    if p0 is None:
        p0 = np.linspace(np.max([EPS, phat/rang]), np.min([1 - EPS, phat*rang]), 1000)

    # Log-likelihood ratio
    LLR = 2*( (k*np.log(phat)+(n-k)*np.log(1-phat)) - (k*np.log(p0)+(n-k)*np.log(1-p0)))

    # Find chi2 distribution limit (log-likelihood ratio
    # distributed asymptotically for H0 like chi2)
    chi2    = stats.chi2.ppf(q=1-alpha, df=1)
    ind     = (np.where(LLR <= chi2))[0] # Note <= not <
    
    return np.array([p0[ind[0]], p0[ind[-1]]])


def profile_LLR_binom_ratio_err(k1,n1, k2,n2, alpha=0.05, EPS=1E-15, rang=10, nd=200,
    nd_interp=2000, r0_val=None, tol=1e-9, return_full=False):
    """ Profile likelihood ratio test based two binomial ratio r = (k1/n1) / (k2/n2) uncertainty.
    """
    
    def logL(x):

        # Numerical protection
        phi = np.clip(x[0], EPS, 1-EPS)
        p1  = np.clip(x[1], EPS, 1-EPS)

        # Other terms do not contribute in the ratio, than these below
        ll1 = k1*np.log(p1)     + (n1-k1)*np.log(1 - p1)
        ll2 = k2*np.log(p1/phi) + (n2-k2)*np.log(1 - p1/phi)
        return ll1 + ll2

    ### Find the numerical Maximum Likelihood
    """
    x0     = np.array([(k1/n1) / (k2/n2), (k1/n1)])
    res    = scipy.optimize.minimize(lambda x : -logL(x), x0=x0,  method='Nelder-Mead', tol=tol)
    r_MLE  = res.x[0]
    p1_MLE = res.x[1]
    """
    # Closed form
    r_MLE  = np.clip((k1/n1) / (k2/n2), EPS, 1-EPS)
    p1_MLE = np.clip((k1/n1), EPS, 1-EPS)
    
    # ------------------------------------------------------------------------
    # Profile likelihood on the ratio r = p1/p2

    if r0_val is None:
        r0_val = np.linspace(r_MLE / rang, r_MLE*rang, nd)

    LLR    = np.zeros(len(r0_val))

    # Closed-form quadratic solution [negative branch]
    @numba.njit
    def p1star_closed_form(r0):
        return (k1+n2+k2*r0+n1*r0 - np.sqrt((-k1-n2-k2*r0-n1*r0)**2 - 4*(n1+n2)*(k1*r0+k2*r0))) / (2*(n1+n2))

    # Discretize over r0
    for i in range(len(r0_val)):
        r_0 = r0_val[i]
        def profileNegLL(p1):
            return -logL(np.array([r_0, p1]))

        # r_0 is our parameter of interest (non-nuisance), profile over the nuisance parameters
        #res     = scipy.optimize.minimize(profileNegLL, x0=[p1_MLE], method='Nelder-Mead', tol=tol)
        #p1_star = res.x[0]
        p1_star = p1star_closed_form(r_0)

        # Profile log-likelihood ratio
        LLR[i]  = 2 * ( logL(np.array([r_MLE, p1_MLE])) - logL(np.array([r_0, p1_star])) )
        
    LLR[np.isnan(LLR)] = 0
    LLR[np.isinf(LLR)] = 0

    # Interpolate values
    func_LLR  = interp1d(r0_val, LLR, kind='cubic', fill_value='extrapolate')
    r0_dense  = np.linspace(np.min(r0_val), np.max(r0_val), nd_interp)
    LLR_dense = func_LLR(r0_dense)

    # Find chi2 distribution limit (log-likelihood ratio
    # distributed asymptotically for H0 like chi2)
    chi2    = stats.chi2.ppf(1 - alpha, df=1)
    ind     = (np.where(LLR_dense <= chi2))[0] # Note <= not <
    min_ind = ind[0]
    max_ind = ind[-1]

    if return_full:
        return np.array([r0_dense[min_ind], r0_dense[max_ind]]), r0_dense, LLR_dense, chi2
    else:
        return np.array([r0_dense[min_ind], r0_dense[max_ind]])


def binom_ratio_cond_err(k1,n1, k2,n2, CL=[0.025, 0.975], method='CP'):
    """ Confidence interval for a ratio between two confidence intervals.
    
    Args:
        k1, n1 : for the binomial 1
        k2, n2 : for the binomial 2
    
    (k1/n1) / (k2/n2)

    Based on the conditional odds ratio ~ p1 / (p1 + p2)
    """
    if   method == 'CP':
        C = clopper_pearson_err(k=k1, n=k1+k2, CL=CL)
    elif method == 'mid-P':
        C = binom_err_midp(k=k1, n=k1+k2, CL=CL)
    else:
        raise Exception(f'binom_ratio_cond_err: unknown basic method = {method}')

    lower = n2 / n1 * C[0] / (1 - C[0])
    upper = n2 / n1 * C[1] / (1 - C[1])

    return np.array([lower, upper])


@numba.njit
def katz_binomial_ratio_err(k1,n1, k2,n2, z=1.0):
    """ Katz el al. ratio confidence interval of two binomial proportions.
    """

    RR      = (k1/n1) / (k2/n2) # mean

    logRR   = np.log(RR)
    seLogRR = np.sqrt(1/k1 + 1/k2 - 1/n1 - 1/n2)

    lower   = np.exp(logRR - z*seLogRR)
    upper   = np.exp(logRR + z*seLogRR)

    return np.array([lower, upper])


@numba.njit
def newcombe_binomial_ratio_err(k1,n1, k2,n2, z=1.0):
    """ Newcombe-Brice-Bonnett ratio confidence interval of two binomial proportions.
    """
    
    RR      = (k1/n1) / (k2/n2) # mean

    logRR   = np.log(RR)
    seLogRR = np.sqrt(1/k1 + 1/k2 - 1/n1 - 1/n2)

    ash   = 2 * np.arcsinh(z/2 * seLogRR)
    lower = np.exp(logRR - ash)
    upper = np.exp(logRR + ash)

    return np.array([lower, upper])


def bayes_binom_err(k, n, prior=[0.5,0.5], CL=[0.025, 0.975]):
    """ Bayesian posterior credible interval for the binomial proportion 
    using a prior Beta(alpha, beta).
    """

    if prior == 'Flat':
        prior = [1, 1]

    if prior == 'Jeffrey':
        prior = [0.5, 0.5]

    if prior == 'Haldane':
        prior = [0, 0]
    
    # Calculate credible level values using inverse Beta CDF quantiles
    a = k     + prior[0]
    b = n - k + prior[1]

    # Special case must be treated separately
    if   k == 0:
        lower = 0
        upper = stats.beta.ppf(q=CL[1], a=a, b=b)

    # Special case must be treated separately
    elif k == n:
        lower = stats.beta.ppf(q=CL[0], a=a, b=b)
        upper = 1

    # Normal case
    else:
        lower = stats.beta.ppf(q=CL[0], a=a, b=b)
        upper = stats.beta.ppf(q=CL[1], a=a, b=b)

    return np.array([lower, upper])


#@ray.remote
def bayes_posterior_ratio(z, k1,n1, k2,n2, alpha1,beta1, alpha2,beta2, precision=50, maxterms=10**12, maxprec=10**12):
    """
    Bayesian binomial ratio function based on using two 
    independent binomial likelihoods and two independent beta priors.
    
    """
    mp.dps = precision # Set precision

    y1 = mpmath.power(z, alpha1+k1-1) 

    # Gamma functions
    y2 = mpmath.gamma(alpha1+alpha2+k1+k2) * mpmath.gamma(beta2-k2+n2)

    # Regularized Gauss 2F1 function
    y3 = mpmath.hyp2f1(alpha1+alpha2+k1+k2, 1-beta1+k1-n1, alpha1+alpha2+beta2+k1+n2, z, maxprec=maxprec, maxterms=maxterms) \
            / mpmath.gamma(alpha1+alpha2+beta2+k1+n2)

    # Beta functions
    y4 = mpmath.beta(alpha1+k1, beta1-k1+n1) * mpmath.beta(alpha2+k2, beta2-k2+n2)

    return float(y1 * y2 * y3 / y4)


def gamma_param_estimate(mu,sigma):
    """
    Method of Moments estimate of Gamma distribution parameters.

    Args:
        mu    : mean E[X]
        sigma : standard deviation sqrt[Var[X]]
    Returns:
        k,theta
    """
    k     = (mu/sigma)**2
    theta = sigma**2/mu

    return k,theta


def bayes_binomial_ratio_err(k1,n1, k2,n2, prior1=[0.5,0.5], prior2=[0.5,0.5],
    a = None, sigma_a = None, b = None, sigma_b = None, ab_prior_type=['Normal', 'Normal'],
    nd=1000, nd_interp=2000, rmax = None, rval = None, CL=[0.025, 0.975],
    nd_y=1500, nd_nuisance=20, int_nncut=5, int_prec=0.1, numerics='numerical', renorm=True,
    gEPS = 0.1):
    
    """
    Bayesian two binomial sampling process (double) ratio.
    
    Args:
        k1,n1:          binomial trial 1 parameters (accepted, trials)
        k2,n2:          binomial trial 2 parameters (accepted, trials)
        prior1          conjugate beta-prior(alpha,beta) 2-array
        prior2          conjugate beta-prior(alpha,beta) 2-array
        
        a,sigma_a:      multiplicative scale constraint prior parameter applied on k1 counts,
                        implement through a Gamma or Gaussian pdf prior with (mean,std) = (a,sigma_a)
        
        b,sigma_b:      like the prior constraint parameter a, but applied on k2 counts
        
        ab_prior_type: 'Gamma' or 'Normal' scale contraint distribution type
        
        CL:             confidence (credible) levels
        numerics:       numerical evaluation method: 'numerical' or 'mpmath' (only without nuisance param.)
    

    Parameters:
        nd:           number of posterior P(r) sampling points
        nd_y:         number of change of variable integral Int{dy} points
        nd_nuisance:  number of prior (nuisance) parameter Int{da,db} integral points
        int_nncut:    number of sigmas in the prior (nuisance) integration
        int_prec:     pdf integral test error threshold, to spot inadequate discretization
        
        rmax:         maximum double ratio value used in the discretization
        rval:         manual discretization as an array
        renorm:       final numerical re-normalization of the posterior integral to one
        
        gEPS:         minimum [sigma_a / a] threshold before forcing Normal prior pdf
    
    Returns:
        Full posterior pdf P(r) at sampling points
    
    Notes:
        Try increasing (integration) parameter discretization counts if you see
        oscillation or distortions in the output. Integral over y is the most
        sensitive to the underlying hypergeometric functions.
    """

    # --------------------------------------------------------------------
    # Numerical protection
    if a is not None:
        if (sigma_a / a) < gEPS:
            cprint(f'Forcing normal prior(a) pdf for numerical protection','yellow')
            ab_prior_type[0] = 'Normal'

    if b is not None:
        if (sigma_b / b) < gEPS:
            cprint(f'Forcing normal prior(b) pdf for numerical protection','yellow')
            ab_prior_type[1] = 'Normal'
    # --------------------------------------------------------------------

    if prior1 == 'Flat':
        prior1 = [1, 1]
    if prior1 == 'Jeffrey':
        prior1 = [0.5, 0.5]
    if prior1 == 'Haldane':
        prior1 = [0, 0]

    if prior2 == 'Flat':
        prior2 = [1, 1]
    if prior2 == 'Jeffrey':
        prior2 = [0.5, 0.5]
    if prior2 == 'Haldane':
        prior2 = [0, 0]

    print(__name__ + f'.bayes_binomial_ratio: prior1 = {prior1}, prior2 = {prior2}')

    # Beta prior parameters
    alpha1,beta1 = prior1[0],prior1[1]
    alpha2,beta2 = prior2[0],prior2[1]

    # --------------------------------------------------------------------
    # y-integral samples for each pdf(r) point
    def integrand(r, y, k1_new, k2_new):
        return np.abs(y)*binom_post_2D(p1=r*y, p2=y, \
            k1=k1_new,n1=n1, k2=k2_new,n2=n2, alpha1=alpha1,beta1=beta1, alpha2=alpha2,beta2=beta2)

    # --------------------------------------------------------------------
    # Return scale prior pdf values
    def get_ab_prior_pdf(x,mu,sigma, mode):

        if   mode == 'Gamma':
            gamma_k, gamma_theta = gamma_param_estimate(mu=mu, sigma=sigma)
            print(f'Gamma pdf param k={gamma_k:0.5f}, theta={gamma_theta:0.5f}')

            return functions.gamma_pdf(x=x, k=gamma_k, theta=gamma_theta)

        elif mode == 'Normal':
            return functions.normpdf(x=x, mu=mu, std=sigma)

        else:
            raise Except(f'.bayes_binomial_ratio_err: Unknown scale prior type = {ab_prior_type}')

    # --------------------------------------------------------------------
    # Integration range
    def genrange(u, sigma_u, k, n):

        MIN = u - int_nncut*sigma_u
        MAX = u + int_nncut*sigma_u
        
        # Boundary control
        if MIN*k < 1: MIN = 1/k 
        if MAX*k > n: MAX = n/k

        return np.linspace(MIN, MAX, nd_nuisance)

    # --------------------------------------------------------------------

    # Set maximum ratio to the upper tail
    if rmax is None:
        rmax = 6 * (k1/n1) / (k2/n2)

    # Random variable p discretized on a reasonably large interval (loop checks the discretization)
    trials = 1
    while True:
        if rval is None or trials > 1:
            rval = np.linspace(0, rmax, trials * nd)
            pdf  = np.zeros(len(rval))

        # Via arbitrary precision library (can be very slow for large numbers)
        if   numerics == 'mpmath':
            
            pdf = [bayes_posterior_ratio(rval[i], k1,n1, k2,n2, alpha1,beta1, alpha2,beta2) for i in tqdm(range(len(rval)))]

        # Via numerical integration
        elif numerics == 'numerical':

            pdf  = np.zeros(len(rval))
            yval = np.linspace(0,1, nd_y)

            # ============================================================
            # Nuisance scale parameters

            k1_new = None
            k2_new = None

            if a is not None:
                aval     = genrange(u=a, sigma_u=sigma_a, k=k1, n=n1)
                a_prior  = get_ab_prior_pdf(x=aval, mu=a, sigma=sigma_a, mode=ab_prior_type[0])
                k1_new   = aval*k1

                # Compute re-normalization (can be crucial near zero, when the left tail is truncated)
                Z        = simps(x=aval, y=a_prior); print(f'Prior scale param [a] {ab_prior_type[0]} pdf norm. integral: {Z}')
                a_prior /= Z

            if b is not None:
                bval     = genrange(u=b, sigma_u=sigma_b, k=k2, n=n2)
                b_prior  = get_ab_prior_pdf(x=bval, mu=b, sigma=sigma_b, mode=ab_prior_type[1])
                k2_new   = bval*k2

                # Compute re-normalization (can be crucial near zero, when the left tail is truncated)
                Z        = simps(x=bval, y=b_prior); print(f'Prior scale param [b] {ab_prior_type[1]} pdf norm. integral: {Z}')
                b_prior /= Z

            # ============================================================
            # Construct PDF(r) numerically. Bayes denominator (normalization) already handled.

            # Apply prior scales a (b) to k1 (k2) and the binomial boundary condition.
            # [Note: cannot apply to p1 (p2) => would result formally
            # in an unidentifiable model (singular Fisher information), at least if a (b)
            # would be floating parameters.

            # Only a
            if a is not None and b is None:
                print(__name__ + f'.bayes_binomial_ratio_err: Numerator prior scale param a = ({a}, {sigma_a})')
                
                for i in tqdm(range(len(rval))):
                    Ia = np.zeros(len(aval))

                    for j in range(len(aval)):
                        I     = integrand(r=rval[i], y=yval, k1_new=k1_new[j], k2_new=k2)
                        Ia[j] = simps(x=yval, y=I)

                    # ***
                    pdf[i] = simps(x=aval, y=Ia*a_prior)

            # Only b
            elif a is None and b is not None:
                print(__name__ + f'.bayes_binomial_ratio_err: Denominator prior scale param b = ({b}, {sigma_b})')
                
                for i in tqdm(range(len(rval))):
                    Ib = np.zeros(len(bval))

                    for j in range(len(bval)):
                        I     = integrand(r=rval[i], y=yval, k1_new=k1, k2_new=k2_new[j])
                        Ib[j] = simps(x=yval, y=I)

                    # ***
                    pdf[i] = simps(x=bval, y=Ib*b_prior)

            # Both a and b
            elif a is not None and b is not None:
                print(__name__ + f'.bayes_binomial_ratio_err: Num. and denom. prior scale param a = ({a}, {sigma_a}) and b = ({b}, {sigma_b})')

                for i in tqdm(range(len(rval))):

                    Ia = np.zeros(len(aval))
                    for j in range(len(aval)):

                        Ib = np.zeros(len(bval))
                        for k in range(len(bval)):
                            I     = integrand(r=rval[i], y=yval, k1_new=k1_new[j], k2_new=k2_new[k])
                            Ib[k] = simps(x=yval, y=I)

                        Ia[j] = simps(x=bval, y=Ib*b_prior)

                    # ***
                    pdf[i] = simps(x=aval, y=Ia*a_prior)

            # The no nuisance parameters case
            else:
                print(__name__ + f'.bayes_binomial_ratio_err: No prior (scale) parameters.')

                for i in tqdm(range(len(rval))):
                    I = np.abs(yval)*binom_post_2D(p1=rval[i]*yval, \
                        p2=yval, k1=k1,n1=n1, k2=k2,n2=n2, alpha1=alpha1,beta1=beta1, alpha2=alpha2,beta2=beta2)
                    pdf[i] = simps(x=yval, y=I)
        else:
            raise Exception(__name__ + f'.bayes_binomial_ratio_err: Unknown numerics method {numerics}')

        # Interpolate
        f2        = interp1d(rval, pdf, kind='quadratic', fill_value='extrapolate')
        r_dense   = np.linspace(0, rmax, nd_interp)
        pdf_dense = f2(r_dense)
        
        # Check normalization
        I = simps(y=pdf_dense, x=r_dense)
        if np.abs(I-1) > int_prec:
            trials += 1
            if numerics == 'numerical':
                nd_y        *= 2
                nd_nuisance *= 2
            print(__name__ + f'.bayes_binomial_ratio_err: Posterior integral {I:.6f} => increasing discretization')
            if trials > 10:
                raise Exception(__name__ + f'bayes_binomial_ratio_err: PDF(r) normalization I={I} error (set tech-parameters manually)') 
        else:
            break
    
    # Normalization of the posterior PDF to unit integral
    if renorm:
        pdf_dense /= simps(x=r_dense, y=pdf_dense)

    print(__name__ + f' >> Posterior integral before: {I:.6f} | after: {simps(x=r_dense, y=pdf_dense)}')

    discrete_pdf  = pdf_dense / np.sum(pdf_dense) # Normalize to discrete PDF
    discrete_cdf  = np.cumsum(discrete_pdf)       # Discrete CDF
    CR_val,CR_ind = tools.cdf_percentile(discrete_cdf, r_dense, CL)
    
    output = {
        'val'         : r_dense,
        'pdf'         : pdf_dense,
        'discrete_pdf': discrete_pdf,
        'discrete_cdf': discrete_cdf,
        'CR_value'    : CR_val,
        'CR_index'    : CR_ind
    }
    return output


#@numba.njit
def jackknife_1D(x, t_func):
    """ Jackknife

    Args:
        x      : A numpy array of measurement data points (n)
        t_func : A function handle to calculate the test statistic
    
    Returns:
        mu   : The mean
        std  : The standard deviation
        d    : Jacknife differences

    Reference:
        http://statweb.stanford.edu/~ckirby/brad/papers/2018Automatic-Construction-BCIs.pdf
    """
    n     = len(x)
    theta = np.zeros(n)
    ind   = np.arange(n)

    # Leave one out, calculate the observable
    for i in range(n):
        tot_ind  = np.hstack([ind[:i], ind[i+1:]])
        theta[i] = t_func(x[tot_ind]) # size n - 1

    # Mean and Jacknife differences
    mu = np.mean(theta)
    d  = theta - mu

    # Jacknife std
    sigma = np.sqrt((n-1) / n * np.sum(( theta - mu )**2))

    return mu,sigma,d


#@numba.njit
def jackknife_ND(x, t_func):
    """ Jackknife

    Args:
        x      : A numpy array of measurement data points (k x N dim)
        t_func : A function handle to calculate the test statistic

    Returns:
        mu   : The mean
        std  : The standard deviation
        d    : Jacknife differences
    """
    n     = len(x)

    theta = np.zeros(n)
    ind   = np.arange(n)

    # Leave one out, calculate the observable
    for i in range(n):
        theta[i] = t_func(np.vstack([x[ind[:i],:], x[ind[i+1:],:] ])) # size n - 1

    # Mean and Jacknife differences
    mu = np.mean(theta)
    d  = theta - mu

    # Jacknife std
    sigma = np.sqrt((n-1) / n * np.sum(( theta - mu )**2))

    return mu,sigma,d


def ecdf(x):
    """ Generate empirical CDF
    """
    n  = len(x)
    xs = np.sort(x)
    ys = np.arange(1, n+1)/n
    return xs, ys


def binom_bca_bootstrap_err(k, n, B=10000, CL=[0.025, 0.975], acceleration=True, return_full=False):
    """ BCA bootstrap for the binomial uncertainty.
    Args:
        k   : Number of success
        n   : Number of trials
        B   : Number of bootstrap samples
        prc : Confidence interval percentile points in [0,1]
        
    Returns:
        CI  : Confidence interval level values
    """
    theta_MLE = k/n
    k_i = bootstrap_sample_binomial(k, n, B)

    # Bootstrap estimates of the parameter
    theta_i = k_i / n
    theta0_star = np.sum(theta_i) / B
    print(f'theta_MLE = {theta_MLE}, theta0_star = {theta0_star}')

    # -------------------------------------------------
    # Original binomial sample created as a vector
    x = np.zeros(n)   
    x[0:k] = 1        # 1 == success

    # We are interested in the mean
    def t_func(x):
        return np.sum(x) / n

    # Jackknife the sample
    mu,sigma,d = jackknife_1D(x, t_func)
    print(f'mu = {mu}, sigma = {sigma}')

    # Calculate acceleration
    if acceleration:
        a = bootstrap_acceleration(d)
    else:
        a = 0
    print(f'a = {a}')
    # -------------------------------------------------

    # Empirical CDF
    xs,ys    = ecdf(theta_i)
    
    # Construct CDF and inverse CDF
    G_cdf    = lambda x : interp1d(xs, ys, kind='nearest', fill_value='extrapolate')(x)
    G_invcdf = lambda y : interp1d(ys, xs, kind='nearest', fill_value='extrapolate')(y)

    # z0 = \Phi^{-1} \hat{G}( \hat{\theta}  )
    z0 = norm.ppf( G_cdf( theta_MLE ) )
    print(f'z0 = {z0}')
    
    # BCA interval estimates
    interval = np.zeros(len(CL))
    for i in range(len(CL)):
        z_alpha = norm.ppf(CL[i])
        interval[i] = G_invcdf( norm.cdf(z0 + (z0 + z_alpha)/(1 - a*(z0 + z_alpha))) )

    if return_full == True:
        return interval,d,a,k_i
    else:
        return interval

#@numba.njit
def bootstrap_binom_err(k, n, CL=[0.025, 0.975], B=10000, type='percentile'):
    """ Bootstrap based binomial proportion confidence interval estimator.
    """
    phat = k/n

    # Special case must be treated separately
    if   k == 0:
        lower = 0
        upper = 1 - (1-CL[1])**(1/n)

    # Special case must be treated separately
    elif k == n:
        lower = CL[0]**(1/n)
        upper = 1

    # Normal
    else:
        bs    = bootstrap_sample_binomial(k=k, n=n, B=B)
        if   type == 'percentile':
            lower = np.percentile(bs, CL[0]*100, interpolation='lower') / n
            upper = np.percentile(bs, CL[1]*100, interpolation='higher') / n
        elif type == 'basic':
            lower = 2*phat - np.percentile(bs, CL[1]*100, interpolation='higher') / n
            upper = 2*phat - np.percentile(bs, CL[0]*100, interpolation='lower') / n
        else:
            raise Exception(f'bootstrap_binom_err: unknown bootstrap type = {type}')

    return np.array([lower, upper])


#@numba.njit
def bootstrap_binom_ratio_err(k1,n1, k2,n2, CL=[0.025, 0.975], B=10000, type='percentile'):
    """ Bootstrap based two binomial proportion ratio confidence interval estimator.
    """
    rhat = (k1/n1)/(k2/n2)
    bsR,B1,B2 = bootstrap_binomial_ratio(k1,n1, k2,n2, B)

    print(np.mean(bsR))

    out  = np.zeros(len(CL))
    if   type == 'percentile':
        for i in range(len(CL)):
            out[i] = np.percentile(bsR, CL[i]*100)
    elif type == 'basic':
        CL = CL[::-1] # flip intervals
        for i in range(len(CL)):
            out[i] = 2*rhat - np.percentile(bsR, CL[i]*100)
    else:
        raise Exception(f'bootstrap_binom_err: unknown bootstrap type = {type}')

    return out


def binom_ratio_bca_bootstrap_err(k1, n1, k2, n2, B=1000, CL=[0.025, 0.975], acceleration=True, return_full=False):
    """ BCA bootstrap for the ratio of two binomial uncertainties.
    
    Args:
        k1,k2   : Number of success
        n1,n2   : Number of trials
        B       : Number of bootstrap samples
        prc     : Confidence interval percentile points in [0,1]
        
    Returns:
        CI  : Confidence interval level values
    """
    theta_MLE = (k1/n1) / (k2/n2)

    # --------------------------------------------------
    interval,d1,a1,k1_i = binom_bca_bootstrap_err(k1, n1, B, CL=CL, return_full=True)
    interval,d2,a2,k2_i = binom_bca_bootstrap_err(k2, n2, B, CL=CL, return_full=True)

    # Bootstrap estimates of the parameter
    theta_i  = (k1_i/n1) / (k2_i/n2)

    nn = np.min([len(d1),len(d2)])
    
    # Calculate acceleration
    if acceleration:
        a = bootstrap_acceleration(d1[0:nn] / d2[0:nn])
    else:
        a = 0
    print(f'a = {a}')
    # --------------------------------------------------

    # Empirical CDF
    xs,ys    = ecdf(theta_i)
    
    # Construct CDF and inverse CDF
    G_cdf    = lambda x : interp1d(xs, ys, kind='nearest', fill_value='extrapolate')(x)
    G_invcdf = lambda y : interp1d(ys, xs, kind='nearest', fill_value='extrapolate')(y)

    # z0 = \Phi^{-1} \hat{G}( \hat{\theta}  )
    z0 = norm.ppf( G_cdf( theta_MLE ) )
    print(f'z0 = {z0}')

    # BCA interval estimates
    interval = np.zeros(len(CL))
    for i in range(len(CL)):
        z_alpha = norm.ppf(CL[i])
        interval[i] = G_invcdf( norm.cdf(z0 + (z0 + z_alpha)/(1 - a*(z0 + z_alpha))) )

    if return_full == True:
        return interval, d1,a1,k1_i, d2,a2,k2_i
    else:
        return interval


@numba.njit
def bootstrap_acceleration(d):
    """ Bootstrap (BCA) acceleration term.
    Args:
        d : Jackknife differences
    Returns:
        a : Acceleration
    """
    return np.sum(d**3) / np.sum(d**2)**(3.0/2.0) / 6.0


@numba.njit
def bootstrap_sample_binomial(k, n, B):
    """ Create binomial bootstrap sample.
    
    Args:
        k: number of success
        n: number of trials
        B: number of bootstraps
    Returns:
        y: the bootstrap sample
    """

    vec = np.zeros(n)   # Original sample created as a vector
    vec[0:k] = 1        # 1 == success
    y = np.zeros(B)     # Bootstrap statistics of # success
    
    for k in range(B):
        y[k] = np.sum(np.random.choice(vec,n)) # default is with replacement

    return y


@numba.njit
def bootstrap_binomial_ratio(k1,n1, k2,n2, B):
    """ Two independent bootstrapped binomial (bernoulli) processes and their ratio.
    """

    B1 = bootstrap_sample_binomial(k=k1, n=n1, B=B)
    B2 = bootstrap_sample_binomial(k=k2, n=n2, B=B)
    R  = (B1/n1) / (B2/n2)

    return R, B1, B2


def inv_raw2cor(q,s,v):
    """ Type I & II error inversion formula. From raw to corrected.

    Args:
        q : raw positive fraction [0,1]
        s : specificity [0,1]
        v : sensitivity [0,1]

    Returns:
        p : corrected (prevalance) fraction [0,1]
    """
    if q < (1-s):
        print(__name__ + f'WARNING: inv_raw2cor: q < (1-s) (ill-posed domain, your specificity is too low)')
    if v < q:
        print(__name__ + f'WARNING: inv_raw2cor: v < q (ill-posed domain; your sensitivity is too low)')
    
    return (q+s-1)/(v+s-1)


def inv_cor2raw(p,s,v):
    """ Type I & II error inversion in reverse direction. From corrected to raw.
    
    Args:
        p : corrected positive (prevalance) fraction [0,1]
        s : specificity [0,1]
        v : sensitivity [0,1]
    
    Returns:
        q : raw positive fraction [0,1]
    """
    q = p*(v+s-1)-s+1
    return q


def inv_p_error(q,s,v, dq,ds,dv):
    """ Error propagation (1st Order Taylor expansion)
    of the inversion formula inv_raw2cor()

    Args:
        q        : raw positive fraction [0,1]
        s        : specificity [0,1]
        v        : sensitivity [0,1]
        dq,ds,dv : 1-sigma uncertainties
    Returns:
        uncertainty on corrected prevalance fraction p
    """
    return np.sqrt( (ds**2*(q-v)**2 + dv**2*(q+s-1)**2 + dq**2*(v+s-1)**2)/(v+s-1)**4 )


def renormalize_test12_error_corrected_input(k, N, s,v, ds, dv):
    """
    USE with corrected counts k.
    
    This function computes, how much there is extra relative uncertainty
    in a test type I and II error corrected positive count data wrt.
    pure binomial uncertainty of the same data.
    
    Args:
        k  : number of positive test counts AFTER corrections
        N  : test sample size
        s  : specificity [0,1]
        v  : sensitivity [0,1]
        ds : 1 sigma uncertainty on s
        dv : 1 sigma uncertainty on r
    
    Output:
        dp_new  : new uncertainty
        dp_orig : original uncertainty
    """

    cprint(__name__ + f'.renormalize_test12_error_corrected_input: \n', 'yellow')
    print(f'specificity s: {s:0.6} +- {ds:0.6e}  [{ds/s*100:0.3e}] %')
    print(f'sensitivity v: {v:0.6} +- {dv:0.6e}  [{dv/v*100:0.3e}] %')
    print('')
    
    # -------------------------------------------
    # Compute the prevalance
    p  = k / N

    # Pure binomial Wilson error on this rate
    err     = wilson_err(k=k, n=N, z=1)
    dp_orig = (err[1] - err[0])/2 # Turn into one sigma equivalent

    print(f'p       = {k} / {N} = {p:0.5f}')
    print(f'dp_orig = {dp_orig:0.5f} [relative dp_orig/p = {dp_orig/p:0.5f}]')

    # -------------------------------------------
    # Do the type I and II error inversion

    # Compute backwards: find out the corresponding raw rate
    q   = inv_cor2raw(p=p, s=s, v=v)
    err = wilson_err(k=np.round(q*N), n=N, z=1)
    dq  = (err[1] - err[0])/2 # Turn into one sigma equivalent

    # Propagate errors
    dp_new = inv_p_error(q=q,s=s,v=v, dq=dq,ds=ds,dv=dv)

    print(f'q       = {np.round(q*N)} / {N} = {q:0.5f}')
    print(f'dq      = {dq:0.5f} [relative dq/q = {dq/q:0.5f}]')
    print(f'dp_new  = {dp_new:0.5f} [relative dp_new/p = {dp_new/p:0.5f}] (after error propagation)')
    print(f' => Relative increases in 1-sigma error ')
    print(f'    (dp_new - dp_orig) / p             = {(dp_new - dp_orig)/p * 100:0.1f} %')
    print(f'    (dp_new / dp_orig - 1)             = {(dp_new / dp_orig-1) * 100:0.1f} %')
    print(f'    sqrt[(dp_new/p)^2 - (dp_orig/p)^2] = {np.sqrt((dp_new/p)**2 - (dp_orig/p)**2) * 100:0.1f} %')
    print('\n')

    return dp_new, dp_orig


def renormalize_test12_error_raw_input(k, N, s,v, ds, dv):
    """
    USE with raw uncorrected counts k.
    
    This function computes, how much there is extra relative uncertainty
    in a test type I and II error corrected positive count data wrt.
    pure binomial uncertainty of the same data.
    
    Args:
        k  : number of raw positive test counts BEFORE corrections
        N  : test sample size
        s  : specificity [0,1]
        v  : sensitivity [0,1]
        ds : 1 sigma uncertainty on s
        dv : 1 sigma uncertainty on v
    
    Output:
        dp_new  : new uncertainty
        dp_orig : original uncertainty
    """

    cprint(__name__ + f'.renormalize_test12_error_raw_input: \n', 'yellow')
    print(f'specificity s: {s:0.6} +- {ds:0.6e}  [{ds/s*100:0.3e}] %')
    print(f'sensitivity v: {v:0.6} +- {dv:0.6e}  [{dv/v*100:0.3e}] %')
    print('')
    
    # -------------------------------------------
    # Compute the raw prevalance
    q   = k / N
    err = wilson_err(k=k, n=N, z=1)
    dq  = (err[1] - err[0])/2 # Turn into one sigma equivalent

    print(f'q       = {k} / {N} = {q:0.5f}')
    print(f'dq      = {dq:0.5f} [relative dq/q = {dq/q:0.5f}]')

    # -------------------------------------------
    # Do the type I and II error inversion
    
    # Compute corrected prevalance
    p       = inv_raw2cor(q=q, s=s, v=v)

    # Pure binomial Wilson error on this rate
    err     = wilson_err(k=np.round(p*N), n=N, z=1)
    dp_orig = (err[1] - err[0])/2 # Turn into one sigma equivalent

    # Propagate errors
    dp_new  = inv_p_error(q=q,s=s,v=v, dq=dq,ds=ds,dv=dv)

    print(f'p       = {np.round(p*N)} / {N} = {p}')
    print(f'dp_orig = {dp_orig:0.5f} [relative dp_orig/p = {dp_orig/p:0.5f}]')    
    print(f'dp_new  = {dp_new:0.5f} [relative dp_new/p = {dp_new/p:0.5f}] (after error propagation)')
    print(f' => Relative increases in 1-sigma error ')
    print(f'    (dp_new - dp_orig) / p             = {(dp_new - dp_orig)/p * 100:0.1f} %')
    print(f'    (dp_new / dp_orig - 1)             = {(dp_new / dp_orig-1) * 100:0.1f} %')
    print(f'    sqrt[(dp_new/p)^2 - (dp_orig/p)^2] = {np.sqrt((dp_new/p)**2 - (dp_orig/p)**2) * 100:0.1f} %')
    
    print('\n')

    return dp_new, dp_orig
