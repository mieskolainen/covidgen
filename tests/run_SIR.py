# SIR model stochastic simulation
#
# Run few times if no output (the epidemic does not start always)
#
# m.mieskolainen@imperial.ac.uk, 2020


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Import local path
import sys
sys.path.append('./covidgen')
from aux import *
import estimators as est

import math
import random



@numba.njit
def sim(S,I,R,D,C, beta,gamma,delta,theta):

    # Output
    t_ = np.zeros(int(1e6))
    S_ = np.zeros(int(1e6))
    I_ = np.zeros(int(1e6))
    R_ = np.zeros(int(1e6))
    D_ = np.zeros(int(1e6))
    C_ = np.zeros(int(1e6))


    #I_tot = np.zeros(int(1e6))
    a     = np.zeros(4)


    # Gillespie algorithm
    k  = 0
    t  = 0
    dt = 0

    while t < T:
        if I == 0: break

        ### Stochastic SIR process probabilities

        # Infection
        a[0] = beta(t) * S*I/N
        # Resolving of the infection
        a[1] = gamma*I
        # Death due to infection
        a[2] = delta*theta*R
        # Recovery from the infection
        a[3] = (1-delta)*theta*R

        a += 1e-12 # Numerical protection

        # ----------------------------------------
        # Time step exponentially distributed random number,
        # scaled by the sum of process rates

        W  = np.sum(a)
        dt = -np.log(np.random.rand()) / W
        t  += dt

        # Determine which process happens after dt
        #try:
        pr = np.random.multinomial(1, a/W).argmax()
            #pr = np.random.choice(len(a), 1, p=a/W)
        #except:
        #    print(a)
        
        # Infection
        if pr == 0:
            if S >= 1:
                S -= 1
                I += 1
        # Resolving
        if pr == 1:
            if I >= 1:
                I -= 1
                R += 1
        # Death due to infection
        if pr == 2:
            if R >= 1:
                R -= 1
                D += 1
        # Recovery from the infection
        if pr == 3:
            if R >= 1:
                R -= 1
                C += 1

        t_[k] = t
        S_[k] = S
        I_[k] = I
        R_[k] = R
        D_[k] = D
        C_[k] = C
        k += 1

    return t_[0:k], S_[0:k], I_[0:k], R_[0:k], D_[0:k], C_[0:k]


# -------------------------------------------------------
# Dynamic parameters


# Basic infection rate function, depends on the lockdown etc policies
@numba.njit
def beta(t):
    return 0.5


T = 90.0          # Maximum elapsed time (t)
t = 0.0           # Start time
N = 12597         # Total population


gamma = 0.3       # The average time person is infectious = 1/gamma
theta = 0.1       # The average time to resolve = 1/theta
delta = 1E-2      # IFR


# Basic re-production number at t=0
R0 = beta(0) / gamma


print(f'Input parameters:')
print('')
print(f'Basic reproduction number R0 = {R0}')


# -------------------------------------------------------
# Kinematic population variables
# S[t] + I[t] + R[t] == N

I = 2          # Number of initially infected
S = N - I      # Suspectible
R = 0
D = 0
C = 0


# Run the simulation
t,S_,I_,R_,D_,C_ = sim(S,I,R,D,C, beta,gamma,delta,theta)


print(D_)


fig,ax = plt.subplots(1)
ind = np.arange(0,len(t), 10)

ax.plot(t[ind], S_[ind], label='[S]uscectible')
ax.plot(t[ind], I_[ind], label='[I]nfected')
ax.plot(t[ind], R_[ind], label='[R]esolving')
ax.plot(t[ind], D_[ind], label='[D]ead')
ax.plot(t[ind], C_[ind], label='[C]recovered')

ax.set_xlim([0,90])

ax.set_xlabel('time $t$ [days]')
ax.set_ylabel('state count at time [t]')


ax.set_ylim([0,None])
ax.legend()
#ax[0].set_aspect('equal', 'box')

plt.show()

