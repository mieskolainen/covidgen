# Tool functions
#
# m.mieskolainen@imperial.ac.uk, 2020

import numba
import numpy as np
import bisect
import copy
import pickle
from   datetime import datetime, timedelta
from   tqdm import tqdm

import scipy
import scipy.special as special
from scipy.integrate import trapz
from scipy.optimize  import fsolve
from scipy.optimize  import least_squares
from scipy.optimize  import nnls

import functions
import aux
import cstats


@numba.njit
def zeropad_after(x, reference):
    """
    Array zero-padding

    Args:
        x         : values array
        reference : reference array with len(reference) > len(x)
    Returns:
        y         : [x; 0,0,...,0]
    """
    #if len(reference) < len(x):
    #    raise Except(__name__ + '.zeropad_after: Error: len(reference) < len(x)')
    
    y = np.zeros(len(reference))
    y[0:len(x)] = x

    return y


@numba.njit
def arg_min(x, values):
    """
    argmin operation
    """
    min_ind = 0
    d_best  = 1e32
    d = np.abs(values - x)
    for i in range(len(d)):
        if d[i] <= d_best:
            min_ind = i
            d_best  = d[i]
    return int(min_ind)

@numba.njit
def arg_max(x, values):
    """
    argmax operation
    """
    max_ind = 0
    d_best  = 1e32
    d = np.abs(values - x)
    for i in range(len(d)-1, -1, -1):
        if d[i] <= d_best:
            max_ind = i
            d_best  = d[i]
    return int(max_ind)


def get_credible_regions(xval, pdf, prc = np.array([0.25, 0.975]), ifactor=5):
    """
    Extract credible regions from a sampled pdf via interpolation
    
    Args:
        xval      : x-axis points
        pdf       : sampled pdf values at xval points
        CR        : credible levels
        ifactor   : interpolation factor

    Returns:
        x_dense   : interpolated x-values
        pdf_dense : interpolated pdf-values
        CR_val    : credible levels (equal tailed)
    """
    # Interpolate
    f2            = scipy.interpolate.interp1d(xval, pdf, kind='cubic')
    x_dense       = np.linspace(0, max(xval), len(xval)*ifactor)
    pdf_dense     = f2(x_dense)

    discrete_pdf  = pdf_dense / np.sum(pdf_dense) # Normalize to discrete PDF
    discrete_cdf  = np.cumsum(discrete_pdf)       # Discrete CDF
    CR_val,CR_ind = cdf_percentile(xarr=x_dense, cdf=discrete_cdf, prc=prc)

    return x_dense, pdf_dense, CR_val



def get_bs_percentiles(X, q=np.array([0.025, 0.5, 0.975])):
    """ Get lower and upper percentiles for a bootstrap sample matrix
    Args:
        X:    bootstrapped estimates (samples x dimensions)
        q:    percentile level(s) (array) [0,...,1]
    Returns:
        percetile values
    """
    numinf = np.sum(np.isinf(X))
    numnan = np.sum(np.isnan(X))
    if numinf > 0:
        print(f'get_bs_percentiles: Warning; found inf: {numinf} / {np.prod(X.shape)}')
    if numnan > 0:
        print(f'get_bs_percentiles: Warning; found nan: {numnan} / {np.prod(X.shape)}')

    X[np.isinf(X)] = 0
    X[np.isnan(X)] = 0

    P = np.zeros((len(q), X.shape[1]))
    for k in range(len(q)):
        for j in range(X.shape[1]):
            P[k,j] = np.percentile(X[:,j], q[k] * 100)

    return P


def get_weibull_param(mean, std, method='moments', x0=(1,1)):
    """ Obtain Weibull distribution parameters based on mean and std.
    Args:
        mean, std:  parameters to be converted
        x0:         initial estimate for non-linear least squares
        method:     closed-form method of moments 'moments' or 'numerical'
    Returns:
        Weibull lambda, k
    """
    if method == 'numerical':
        def func(x):
            lambd,k = x[0],x[1]
            return [mean - lambd * special.gamma(1 + 1/k), std**2 - lambd**2 * (special.gamma(1 + 2/k) - special.gamma(1+1/k)**2)]

        res = least_squares(func, x0, bounds = ((0, 0), (mean*10, std*10)))
        return res.x[0],res.x[1]

    # Method of moments
    elif method == 'moments':
        k    = (std / mean) ** (-1.086)
        lamb = mean / special.gamma(1 + 1/k)

        k    = np.abs(k)
        lamb = np.abs(lamb)
        return lamb, k

    else:
        raise Exception('get_weibull_param: unknown method')


def get_f_mean_sigma(t, f_t):
    """ Compute mean and std for a sampled continuum function.
    Args:
        t : time values
        f_t : sampled function values
    Returns:
        mean, std
    """
    mean  = trapz(x=t, y=t * f_t)
    sigma = np.sqrt(trapz(x=t, y=(t - mean)**2 * f_t))
    return mean, sigma


@numba.njit
def find_delay(t, F, Fd, rho = 0.95):
    """ Find delay values.
    Note that F and Fd need to have the same (relative) scale!
    
    Args:
        t   : time values array
        F   : cumulative array
        Fd  : delayed cumulative array
        rho : threshold value ( Fd/F[i] ~= rho )
    Returns:
        delays in an array
    """
    dd = np.zeros(len(F))
    
    for i in range(len(F)):
        ind   = np.argmin( np.abs(Fd/F[i] - rho))
        dd[i] = t[ind] - t[i]

    dd[dd < 0] = 0
    return dd

@numba.njit
def conv_(f, kernel):
    """ Discrete convolution sum

    Implement proper domain span [len(f) + len(k)] outside this function.
    
    Args:
        f:       Function values array
        k:       Kernel values array
    Returns:
        Convolved signal
    """
    if len(f) != len(kernel):
        raise Exception('conv_: input f and kernel should be same length')

    N = len(f)
    y = np.zeros(N)
    for n in range(0,N):
        m    = np.arange(0,n+1)
        y[n] = np.sum(kernel[n-m] * f[m])

    return y


def convint_(t, f, kernel):
    """ Continuum convolution integral
    
    Implement proper domain span [len(f) + len(k)] outside this function.

    Args:
        t:    Time point values
        f:    Function values array
        k:    Kernel values array
    Returns:
        Convolved function array
    """
    if len(f) != len(kernel):
        raise Exception('conv_: input f and kernel should be same length')
    if len(t) != len(f):
        raise Exception('convint_: input t and f should be same length')

    N = len(f)
    y = np.zeros(N)
    for n in range(0,N):
        m         = np.arange(0,n+1)
        integrand = kernel[n-m] * f[m]
        y[n]      = trapz(x=t[m], y=integrand)
    return y


def convint(t, f, kernel, kernel_param):
    """ Continuum convolution integral with direct kernel function.
    Args:
        t:             Time point values
        f:             Function values array
        k:             Kernel function handle
        kernel_param:  Kernel function parameters
    Returns:
        Convolved function array
    """
    if len(t) != len(f):
        raise Exception('convint: input t and f should be same length')

    N = len(f)
    y = np.empty(N)
    for n in range(0,N):

        u         = t[0:n+1]
        integrand = kernel(t[n] - u, **kernel_param) * f[0:n+1]
        y[n]      = trapz(x=u, y=integrand)
    return y


def grad2matrix(N):
    """ Finite difference laplacian matrix.
    
    [[-1.  2. -1. ...  0.  0.  0.]
     [ 0. -1.  2. ...  0.  0.  0.]
     ...
     [ 0.  0.  0. ...  2. -1.  0.]
     [ 0.  0.  0. ... -1.  2. -1.]]
    
    Args:
        N : dimension
    Returns:
        matrix
    """

    G = np.eye(N) # last elements are just diagonal
    v = np.array([-1,2,-1])

    #for i in np.arange(1,N): # rows
    #    ind = np.arange(i-1,i+2)

    for i in np.arange(0,N): # rows
        ind = np.arange(i,i+3)

        if ind[-1] < N:
            G[i,ind] = v
    return G


def nneg_tikhonov_deconv(y, kernel, alpha=1e-6, x0=None, regmatrix='grad2', mass_conserve=True, verbose=False):
    """ Tikhonov ||Ax - b||^2 + alpha^2 ||L(x-x0)||^2 subject to x >= 0,
        regularized Non-Negative Constrained Least Squares deconvolution.
    
    Args:
        y:          measured signal array
        kernel:     convolution kernel
        x0:         additional constraint vector (default None)
        alpha:      regularization strength
        regmatrix:  point-to-point regularization matrix: 'grad2' or diagonal 'diag'
    
    Returns:
        x:          deconvolved signal
    """
    if (len(y) != len(kernel)):
        raise Exception('nneg_tikhonov_deconv: y and kernel with different lengths')

    # Make it column vector
    y = y.reshape(-1)

    N = len(y)
    C = convmatrix(kernel)
    C = C[0:N, 0:N]            # Same sized convolution

    if   regmatrix == 'grad2': # Laplacian
        L = grad2matrix(N)
    elif regmatrix == 'diag':  # Diagonal
        L = np.eye(N)
    else:
        raise Exception('nneg_tikhonov_deconv: unknown regularization matrix type') 

    # Additional constraint
    if x0 is None:
        x0 = np.zeros(len(y))
    else:
        x0 = x0.reshape(-1) # make sure it is column vector

    # Non-negative problem cast by matrix augmentation
    A = np.vstack((C, alpha*L))
    b = np.hstack([y, alpha*L@x0])

    # Solve via Active Set Method, using Karush-Kuhn-Tucker (KKT) conditions
    sol = nnls(A, b, maxiter=None)
    x   = sol[0]

    # Compute reconstruction and regularization cost
    rec_err = np.linalg.norm(A@x - b)**2
    reg_err = np.linalg.norm(L@(x-x0))**2

    if verbose:
        print(f'nneg_tikhonov_deconv: alpha = {alpha:0.5f}, err(REC) = {rec_err:0.5f}, err(REG) = {reg_err:0.5f}')

    if mass_conserve:
        if np.sum(x) > 0:
            x = x / np.sum(x)
        x = x * np.sum(y)
    
    return x,rec_err,reg_err



def tikhonov_deconv(y, kernel, alpha=1e-6, regmatrix='grad2', positive=True, mass_conserve=True):
    """ Tikhonov ||Lf||^2 regularized Least Squares deconvolution.
    
    Args:
        y:          measured signal array
        kernel:     convolution kernel
        alpha:      regularization strength
        positive:   a posteriori enforce non-negativity by truncation (see non-negative version)
        regmatrix:  point-to-point regularization matrix: 'grad2' or diagonal 'diag'
    
    Returns:
        x:          deconvolved signal
    """
    if (len(y) != len(kernel)):
        raise Exception('Tikhonov_deconv: y and kernel with different lengths')

    # Make it column vector
    y = y.reshape(-1, 1)

    N = len(y)
    C = convmatrix(kernel)
    C = C[0:N, 0:N]        # Same sized convolution

    if   regmatrix == 'grad2': # Laplacian
        L = grad2matrix(N)
    elif regmatrix == 'diag':  # Diagonal
        L = np.eye(N)
    else:
        raise Exception('tikhonov_deconv: unknown regularization matrix type') 

    # Least squares (regularized Moore-Penrose pseudoinverse)
    x = np.linalg.inv(C.transpose()@C + alpha*L)@C.transpose() @ y

    # Compute reconstruction and regularization cost
    rec_err = np.linalg.norm(C@x - y)**2
    reg_err = np.linalg.norm(L@x)**2

    print(f'tikhonov_deconv: alpha = {alpha:0.5f}, err(REC) = {rec_err:0.5f}, err(REG) = {reg_err:0.5f}')

    if positive:
        x[x < 0] = 0

    if mass_conserve:
        if np.sum(x) > 0:
            x = x / np.sum(x)
        x = x * np.sum(y)

    return x,rec_err,reg_err


def FFT_deconv_naive(y, kernel, alpha=1, mass_conserve=True):
    """ FFT (Fourier) naive deconvolution

    (HIGH NOISE AMPLIFICATION -- USE ONLY FOR DEBUG PURPOSES).
        
    Args:
        y:       measured signal array
        kernel:  kernel array
        alpha:   regularization in the denominator
    Returns:
        deconvolved signal
    """
    if (len(y) != len(kernel)):
        raise Exception('FFT_deconv_naive: y and kernel with different lengths')

    YF = np.fft.fft(y)
    HF = np.fft.fft(kernel)

    x  = np.real(np.fft.ifft(YF / HF))

    if mass_conserve:
        if np.sum(x) > 0:
            x = x / np.sum(x)
        x = x * np.sum(y)

    return x


def FFT_deconv(y, kernel, alpha=1e-6, mass_conserve=True):
    """ FFT (Fourier) deconvolution with regularization

    f* = argmin_f ||y - Hf||^2 + alpha * ||f||^2,

    where H is the smearing operator.
    
    Args:
        y:       measured signal array
        kernel:  kernel array
        alpha:   regularization strength
    Returns:
        deconvolved signal
    """
    if (len(y) != len(kernel)):
        raise Exception('FFT_deconv: y and kernel with different lengths')

    YF = np.fft.fft(y)
    HF = np.fft.fft(kernel)
    D  = np.abs(HF)**2 + alpha

    x  = np.real(np.fft.ifft(YF*HF / D))

    if mass_conserve:
        if np.sum(x) > 0:
            x = x / np.sum(x)
        x = x * np.sum(y)

    return x


def RL_deconv(y, kernel, iterations=4, mass_conserve=True):
    """ Richardson-Lucy (EM-iteration) 1D deconvolution
        y = x (*) h
    Args:
        y :             measured signal array
        h :             kernel function array
        iterations :    number of iterations (implicit regularization)
        reg :           naive regularization (default 1e-9)
        mass_conserve : conserve function sum
    Returns
        deconvolved signal
    """
    y_mass = np.sum(y)
    x_hat  = np.ones(y.shape)
    r      = np.ones(y.shape)

    # Iterate towards ML estimate for the latent signal
    for i in range(iterations):

        y_hat   = conv_(x_hat, kernel)
        r[y_hat > 0] = y[y_hat > 0] / y_hat[y_hat > 0]
        x_hat  *= conv_(r, kernel)

        if mass_conserve:
            x_hat /= np.sum(x_hat)
            x_hat *= np.sum(y)

        print(f'RL_deconv: iter = {i+1}, KL-div[y_hat|y] = {np.sum(y_hat[y_hat > 0]*np.log2(y_hat[y_hat > 0] / y[y_hat > 0])):.1f} \
            H[r] = {np.sum(r[r > 0]*np.log2(r[r > 0])):.1f}, \
            H[y_hat] = {np.sum(y_hat[y_hat > 0]*np.log2(y_hat[y_hat > 0])):.1f} \
            Ht[x_hat] = {np.sum(x_hat[x_hat > 0]*np.log2(x_hat[x_hat > 0])):.1f}')

    if mass_conserve:
        if np.sum(x) > 0:
            x = x / np.sum(x)
        x = x * np.sum(y)

    return x_hat

def RL_deconv_cont(t, y, kernel, kernel_param, iterations=4, mass_conserve=True, EPS=1e-15):
    """ Richardson-Lucy (EM-iteration) continuum 1D deconvolution
        y = x (*) h
    Args:
        t :             time values
        y :             function values
        kernel :        kernel function handle
        kernel_param :  kernel function parameters
        iterations :    number of iterations (implicit regularization)
        mass_conserve : conserve function integral
    Returns
        Deconvolved signal
    """

    y_mass = trapz(x=t, y=y)
    x_hat  = copy.deepcopy(y)
    x_hat[x_hat < EPS] = EPS
    
    r      = np.ones(y.shape)

    # Iterate towards ML estimate for the latent signal
    for i in range(iterations):

        y_hat        = convint(t=t, f=x_hat, kernel=kernel, kernel_param=kernel_param)
        r[y_hat > 0] = y[y_hat > 0] / y_hat[y_hat > 0]
        x_hat        = x_hat * convint(t=t, f=r, kernel=kernel, kernel_param=kernel_param)
        
        if mass_conserve:
            x_hat /= trapz(x=t, y=x_hat)
            x_hat *= y_mass

        print(f'RL_deconv: iter = {i+1}, KL-div[y_hat|y] = {np.sum(y_hat[y_hat > 0]*np.log2(y_hat[y_hat > 0] / y[y_hat > 0])):.1f} \
            H[r] = {np.sum(r[r > 0]*np.log2(r[r > 0])):.1f}, \
            H[y_hat] = {np.sum(y_hat[y_hat > 0]*np.log2(y_hat[y_hat > 0])):.1f} \
            Ht[x_hat] = {np.sum(x_hat[x_hat > 0]*np.log2(x_hat[x_hat > 0])):.1f}')

    return x_hat


def heaviside(x):
    """
    Heaviside step function.
    """
    x = np.array(x)
    if x.shape != ():
        y = np.zeros(x.shape)
        y[x > 0.0]  = 1
        y[x == 0.0] = 0.5
    else: # special case for 0-dim array
        if x > 0: y = 1
        elif x == 0: y = 0.5
        else: y = 0
    return y


def cdfint(t, df):
    """ CDF integral from continuum pdf
    Args:
        t:   discrete time steps
        dt:  density function values in an array
    Returns:
        y:   CDF values
    """
    y = np.empty(shape=(0,))
    for i,v in enumerate(t, start=1):
        integral = trapz(x=t[0:i], y=df[0:i])
        y = np.append(y, integral)
    return y


@numba.njit
def MC_integral_1D(N, func, kwargs):
    """ Simple 1D MC integral.
    """
    W  = 0
    W2 = 0
    for i in range(N):
        x  = np.random.rand() # [0,1] interval

        # Integrand
        w  = func(x, *kwargs)
        W  += w
        W2 += w*w

    val = W/N
    err = np.sqrt((W2/N - (W/N)**2)/N)
    return val,err


@numba.njit
def MC_integral_2D(N, func, kwargs):
    """ Simple 2D integral.
    """

    W  = 0
    W2 = 0
    for i in range(N):
        p1 = np.random.rand() # [0,1] interval
        p2 = np.random.rand() # [0,1] interval

        # Integrand
        w  = func(p1,p2, *kwargs)
        W  += w
        W2 += w*w

    # Value and error estimate
    val = W/N
    err = np.sqrt((W2/N - (W/N)**2)/N)
    return val,err


#@numba.njit
def MC_integral_2D_mask(N, func, mask, x1_arr, x2_arr, kwargs):
    """ Masked 2D integral with [j,i] indexing in the boolean mask!
    """

    W  = 0
    W2 = 0
    for i in range(N):
        x1 = np.random.rand()
        x2 = np.random.rand()
        
        # Integrand
        w  = func(x1,x2, *kwargs)

        # Find where the random numbers meet
        _, ind1 = nearest_of_sorted(x1_arr, x1)
        _, ind2 = nearest_of_sorted(x2_arr, x2)

        # Apply mask, [j,i] indexing !
        w *= int(mask[ind2, ind1])

        W  += w
        W2 += w*w

    # Value and error estimate
    val = W/N
    err = np.sqrt((W2/N - (W/N)**2)/N)
    return val,err


@numba.njit
def cdf_function(cdf, xarr, x):
    """ Dicretized CDF function.
    Computes F(x) = percentile from a discretized CDF F(x) array

    Args:
        cdf  : CDF array
        xarr : x-axis discretization array
        x    : List of x points
        
    Returns:
        val  : F(x) values
        ind  : F(x) array indices
    """
    val = np.zeros(len(x))
    ind = np.zeros(len(x))
    for k in range(len(x)):
        val[k],ind[k] = nearest_of_sorted(xarr, x[k])

    return val, ind


@numba.njit
def cdf_percentile(cdf, xarr, prc=[0.025, 0.975]):
    """ Discretized inverse CDF function.
    Computes F^{-1}(percentile) = x from a discretized CDF F(x) array
    using a brute force loop.
    
    Args:
        cdf  : CDF array
        xarr : x-axis discretization array
        prc  : List of percentile points
    
    Returns:
        val  : F^{-1}(prc) values
        ind  : F^{-1}(prc) array indices
    """
    val = np.zeros(len(prc))
    ind = np.zeros(len(prc))
    for k in range(len(prc)):
        for i in range(len(xarr)):
            if cdf[i] > prc[k]:
                val[k] = xarr[i]
                ind[k] = i;
                break;
    return val, ind


def nearest_of_sorted(arr, x):
    """ Input of SORTED values in arr, compare with x.
    
    Args:
        arr :  Sorted array values
    Returns:
        value, index
    """
    
    # -------------------------------
    # Check trivial cases first
    if (x >= arr[-1]):
        return arr[-1], len(arr)-1

    if (x <= arr[0]):
        return arr[0], 0
    # -------------------------------

    ind = bisect.bisect_left(arr, x)
    if ind == 0:
        return arr[0]
    if ind == len(arr):
        return arr[-1]
    before = arr[ind - 1]
    after  = arr[ind]

    if (after - x) < (x - before):
        return after,  ind-1
    else:
        return before, ind-1

def convmatrix(h, mode='full', N=None):
    """
    Toeplitz (band) convolution matrix

    Args:
        h:    Kernel vector array
        mode: 'full', 'valid' or 'same' (see numpy convolution)
        N:    Integer dimension (default none)
    Returns:
        convolution operation matrix
    """

    M = len(h)
    N = M if N is None else N

    if   mode == 'full':
        D     = M + N - 1
        shift = 0
    elif mode == 'valid':
        D     = 1 - min(N,M) + max(N,M)
        shift = min(N,M) - 1
    elif mode == 'same':
        D     = max(N,M)
        shift = (min(N,M) - 1) // 2
    else:
        raise Exception(f"convmatrix: unknown dimension mode = {mode}")

    H = np.hstack([h, np.zeros(D)])
    n = np.arange(D)[:, np.newaxis]
    m = np.arange(N)

    return H[shift + n - m]
