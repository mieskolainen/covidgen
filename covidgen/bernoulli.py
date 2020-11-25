# Multidimensional Bernoulli distribution functions
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import numba


@numba.njit
def bernoulli3_combinations(nT,nI,nF):
    """ Count 3-Bernoulli combinations.
    
    Args:
        nT,nI,nF: three arrays with {0,1} (Bernoulli) values
    
    Returns:
        Numpy array with 8 elements
    """

    # Right endian binary expansion
    bstr = nT*4 + nI*2 + nF

    # Accumulate
    B = np.zeros(2**3, dtype=np.double)
    for i in range(len(nT)):
        B[bstr[i]] += 1

    return B


@numba.njit
def bernoulli3_parameters(nT,nI,nF, EPS = 1E-15):
    """ Compute 3-point (multivariate) Bernoulli parameters: 2**3-1 = 7.
    
    Args:
        nT,nI,nF: are arrays containing Bernoulli random numbers

    Returns:
        Numpy array containing 7 parameters
    """
    # Expectation values
    P_T   = np.mean(nT)
    P_I   = np.mean(nI)
    P_F   = np.mean(nF)

    # Standard deviations
    std_T = np.std(nT)
    std_I = np.std(nI)
    std_F = np.std(nF)

    std_T = EPS if std_T < EPS else std_T
    std_I = EPS if std_I < EPS else std_I
    std_F = EPS if std_F < EPS else std_F    
    
    # Correlation coefficients (2-point and 3-point)
    C_TI  = np.mean((nT - P_T) * (nI - P_I)) / (std_T * std_I)
    C_TF  = np.mean((nT - P_T) * (nF - P_F)) / (std_T * std_F)
    C_IF  = np.mean((nI - P_I) * (nF - P_F)) / (std_I * std_F)
    C_TIF = np.mean((nT - P_T) * (nI - P_I) * (nF - P_F)) / (std_T * std_I * std_F)

    return np.array([P_T, P_I, P_F, C_TI, C_TF, C_IF, C_TIF])


def bernoulli2_is_valid(EX, EY, rho):
    """ Calculate phase-space admissibility of the 2D-Bernoulli distribution parameters.

    Args:
        EX:  expectation value of X
        EY:  expectation value of Y
        rho: correlation coefficient [-1,1] between (X,Y)
    
    Returns:
        True or False
    """

    # First get the representation
    P = bernoulli2_rep(EX,EY,rho)

    # Then see, if it is within the probability phase-space
    if (np.all(P >= 0)) & (np.all(P <= 1)) & (np.sum(P) <= 1):
        return True
    else:
        return False


def bernoulli2_rhorange(EX, EY, n=10000):
    """ Get valid rho-parameter range given EX and EY.

    Args:
        EX: the expectation value of X
        EY: the expectation value of Y

    Returns:
        minv : minimum value
        maxv : maximum value
    """
    # Find valid range
    rhoval = np.linspace(-1, 1, n)
    valid  = np.zeros(len(rhoval))
    for i in range(len(rhoval)):
        valid[i] = bernoulli2_is_valid(EX=EX, EY=EY, rho=rhoval[i])

    # Find minimum
    minv = 0
    for i in range(len(valid)):
        if valid[i]:
            minv = rhoval[i]
            break

    # Find maximum
    maxv = 0
    for i in range(len(valid)):
        if valid[i]:
            maxv = rhoval[i]

    return minv, maxv

@numba.njit
def bernoulli2_rep(EX, EY, rho):
    """ Change the representation of 2-point Bernoulli basis to the 2-hypercube basis.

    Args:
        EX:  the expectation value of X
        EY:  the expectation value of Y
        rho: the correlation coefficient

    Returns:
        P:   numpy array with 4 probability elements
    """

    # Change the representation to a multinomial (hypercube) basis
    p3 = rho*np.sqrt(EX*EY*(EX - 1)*(EY - 1)) + EX*EY
    p2 = EX - p3
    p1 = EY - p3
    p0 = 1 - (p1 + p2 + p3)

    P = np.array([p0, p1, p2, p3])

    # For speed, we do not test here if we are inside physically
    # possible phase-space [that check is done outside this function]

    return P

@numba.njit
def bernoulli2_rand(n, EX, EY, rho=0):
    """ Generate 2-dimensional Bernoulli random numbers Z = (X,Y).
    with a non-zero correlation coefficient rho(X,Y) in [-1,1]

    Note! Test the input parameters first with bernoulli2_is_valid() function.
    
    Args:
        n   : Number of experiments
        EX  : Mean <X>  in [0, 1]
        EY  : Mean <Y>  in [0, 1]
        rho : Corr[X,Y] in [-1,1]

    Returns:
        v   : Bernoulli random 2-vectors

    Examples:
        $  v = bernoulli2_rand(n=1000000, EX=0.2, EY=0.4, rho=0.2)
        $  print(f'<X> = {np.mean(v[:,0])}, <Y> = {np.mean(v[:,1])}')
        $  print(f'COR = {np.corrcoef(v[:,0], v[:,1])}')
    """

    # Change the representation
    P = bernoulli2_rep(EX, EY, rho)

    # Cast numbers via the multinomial distribution
    m = np.random.multinomial(n, P)

    # Generate Bernoulli 2-vectors
    B = np.array([[0,0], [0,1], [1,0], [1,1]])

    # Random order
    order = np.arange(n)
    np.random.shuffle(order) # in-place
    k = 0
    
    # Generate vectors in random order
    v = np.zeros((n,2))
    for c in range(4):
        for i in range(m[c]):
            v[order[k],:] = B[c,:]
            k += 1
    
    return v
