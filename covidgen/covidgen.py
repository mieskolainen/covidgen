# COVID Bernoulli MC generator core functions
#
# m.mieskolainen@imperial.ac.uk, 2020


import numpy as np
import numba
from tqdm import tqdm

from bernoulli import *
from aux import *

# Version tag
VER = 1.0


@numba.njit
def bernoulli_param(N, T, I_T, F_N):
    """ Bernoulli parameter definition (Maximum Likelihood)
    
    Args:
        N   : Total number of people in the city
        T   : Number of people in the test sample
        I_T : Number of infected people in the test sample
        F_N : Total number of fatalities in the city
    
    Returns:
        p_T : Bernoulli probabilities
        p_I : --|--
        p_F : --|--
    """

    p_T = T   / N   # Testing rate Bernoulli parameter
    p_I = I_T / T   # Infection rate Bernoulli parameter
    p_F = F_N / N   # Fatality Bernoulli parameter

    return p_T, p_I, p_F


def simulation(N, T, I_T, F_N, rho, R=1000000, constrain=1, fixT=0):
    """ Simulation wrapper function due to Numba/JIT.

                            |--> Multivariate Bernoulli ----.
                            |                               |
    Approximate hierarchy: Bernoulli -> Binomial -> Multinomial -->
                                            |
                                            |---> Poisson -> Gaussian
    """

    printbar('=')
    print(' SIMULATION INPUT')
    printbar('=')
    print('')
    
    print(f' R         = {R} \t| Number of MC runs\n')

    print(f' N         = {N} \t| Number of people in the city')
    print(f' F_N       = {F_N}  \t| Number of deaths in the city')
    print(f' T         = {T} \t| Number of people in the test sample')
    print(f' I_T       = {I_T} \t| Number of infected in the test sample')
    print('\n')
    
    print(f' rho       = {rho:5f}  \t| <Infected,Fatal> correlation coupling [-1,1]')
    print(f' constrain = {constrain}\t\t| Boundary condition: infected=0 AND fatal=1 not allowed')
    print(f' fixT      = {fixT} \t\t| Number of people tested is constant')
    print('\n')
    
    
    B,B3 = simkernel(N=N, T=T, I_T=I_T, F_N=F_N, R=R, rho=rho, constrain=constrain, fixT=fixT)


    # Return full MC statistics
    return B,B3


@numba.njit
def generator(N, p_T, p_I, p_F, rho, constrain=1, fixT=0):
    """ Event sample generator 'kernel'
    For parameters, see simulation() function
    """
    N = int(N)

    # Citizen vectors with 0,1 values
    nT  = np.zeros(N, dtype=np.int8)  # Tested
    nI  = np.zeros(N, dtype=np.int8)  # Infected
    nF  = np.zeros(N, dtype=np.int8)  # Fatal

    # Calculate coupled Bernoulli random numbers here for CPU eff.
    BUFFER = 2*N # larger than n due to constraints, which cause re-trials
    IF = bernoulli2_rand(n=BUFFER, EX=p_I, EY=p_F, rho=rho)
    
    # If the number of tested people is fixed
    if fixT == 1:
        nT[0:int(np.round(N*p_T))] = 1

    # Fixed citizen count loop
    i = 0
    k = 0
    while True:

        # T gets independent Bernoulli trials
        if (fixT == 0):
            nT[i] = 1 if (np.random.rand() < p_T) else 0

        # I and F get coupled Bernoulli trials
        nI[i], nF[i] = IF[k][0], IF[k][1]

        k += 1
        if (k == BUFFER): # Technical, generate a new fresh buffer if we run out
            IF = bernoulli2_rand(n=BUFFER, EX=p_I, EY=p_F, rho=rho)
            k  = 0

        # Filter forbidden combinations by exluding them
        # => induces (trivial) correlation between I and F
        if (constrain == 1):
            if (nI[i]==0) & (nF[i]==1):
                continue        # Try again

        # [add other constraints here ...]

        i += 1
        if i == N: break

    return nT, nI, nF


def simkernel(N, T, I_T, F_N, rho, R=1000000, constrain=1, fixT=0):
    """ Simulation kernel loop
    """
    N = int(N)
    T = int(T)
    R = int(R)

    # Compute Bernoulli parameters
    p_T, p_I, p_F = bernoulli_param(N=N, T=T, I_T=I_T, F_N=F_N)

    # Fundamental observables >>
    
    # 3-Bernoulli hypercube (multinomial) representation table
    B   = np.zeros((R, 8), dtype=np.double)
    # 3-Bernoulli expectation & correlation representation
    B3  = np.zeros((R, 7), dtype=np.double)

    # MC run loop
    for i in tqdm(range(R)):

        # Run sample generator
        nT,nI,nF = generator(N=N, p_T=p_T, p_I=p_I, p_F=p_F, rho=rho, constrain=constrain, fixT=fixT)

        # Get tables
        B[i,:]  = bernoulli3_combinations(nT,nI,nF)
        B3[i,:] = bernoulli3_parameters(nT,nI,nF)

    return B, B3
