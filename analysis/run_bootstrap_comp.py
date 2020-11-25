# First order bootstrap via pure percentiles
# Second order bootstrap (BCA = Bias Corrected Acceleration)
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import numba
import matplotlib.pyplot as plt
from scipy.stats import norm

import sys
sys.path.append('./analysis')
sys.path.append('./covidgen')

import estimators as est
from estimators import *
import aux


# Input data counts
k1 = 7
n1 = 12597
k2 = 138
n2 = 919

# Number of bootstrap runs
B = 10000

aux.set_arr_format(2)

CL = est.q68_q95


CI_bas = est.bootstrap_binom_ratio_err(k1=k1,n1=n1,k2=k2,n2=n2, B=B,     CL=CL, type='basic')
CI_per = est.bootstrap_binom_ratio_err(k1=k1,n1=n1,k2=k2,n2=n2, B=B,     CL=CL, type='percentile')
CI_bc  = est.binom_ratio_bca_bootstrap_err(k1=k1,n1=n1,k2=k2,n2=n2, B=B, CL=CL, acceleration=False)
CI_bca = est.binom_ratio_bca_bootstrap_err(k1=k1,n1=n1,k2=k2,n2=n2, B=B, CL=CL, acceleration=True )

print('\n')
percent=100
print(f'basic       & {CI_bas[[1,2]]*percent} & {CI_bas[[0,3]]*percent} \\\\')
print(f'percentile  & {CI_per[[1,2]]*percent} & {CI_per[[0,3]]*percent} \\\\')
print(f'bc          & {CI_bc [[1,2]]*percent} & {CI_bc [[0,3]]*percent} \\\\')
print(f'bca         & {CI_bca[[1,2]]*percent} & {CI_bca[[0,3]]*percent} \\\\')
print('\n')

## -----------------------------------------------------------------------
## Efron's book test comparison
# Quotes (119/11037) / (98/11034) = 1.21 with a bootstrap range of 0.93 to 1.60

RR,B1,B2 = est.bootstrap_binomial_ratio(k1=119,n1=11037, k2=98,n2=11034, B=100000)

# "Percentile bootstrap intervals"
titlestr = f'Efron book: $\\langle R \\rangle$ = {np.mean(RR):.3} % CL68: [{np.percentile(RR,Q68[0]):.3}, {np.percentile(RR,Q68[1]):.3}] % CL95: [{np.percentile(RR,Q95[0]):.3}, {np.percentile(RR,Q95[1]):.3}] %'

print(titlestr)
print('** Book says: (119/11037) / (98/11034) = 1.21 with a bootstrap range of 0.93 to 1.60 **')
print(__name__ + ' done!')
