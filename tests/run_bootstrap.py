# Bootstrap based confidence intervals tests
# 
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import numba
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('./covidgen')

from estimators import *
from aux import *

# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)


S = 12597  # Number of people in the city
T = 919    # Number of people in the test sample
I = 138    # Number of infected in the test sample
F = 7      # Number of deaths in the city

percent = 100

#-------------------------------------------------------------------------
# Bootstrap of the ratio between two independent binomials

N_bootstrap = 100000
RR,B1,B2 = bootstrap_binomial_ratio(k1=F,n1=S, k2=I,n2=T, B=N_bootstrap)
RR *= percent

IFR = (F/S)/(I/T) * percent


# "Percentile bootstrap intervals"
titlestr = f'$\\langle IFR \\rangle$ = {np.mean(RR):.3} % CL68: [{np.percentile(RR,Q68[0]):.3}, {np.percentile(RR,Q68[1]):.3}] % CL95: [{np.percentile(RR,Q95[0]):.3}, {np.percentile(RR,Q95[1]):.3}] %'


print(titlestr)
bin_edges = np.linspace(0,1,100)

os.makedirs('./figs/bootstrap/', exist_ok = True)

fig = plt.figure()
plt.hist(RR, bins=bin_edges)
plt.xlabel('Infection Fatality Rate ($IFR$) %')
plt.ylabel('Bootstrap runs')
plt.title(f'Percentile bootstrap of two binomials ~ (F/S) / (I/T): S={S}, T={T}, I={I}, F={F}')
plt.xlim([0, 1.001])
plt.xticks(np.arange(0, max(bin_edges), 0.1))
plt.legend([titlestr])
plt.savefig('./figs/bootstrap/bootstrap.pdf', bbox_inches='tight')

fig = plt.figure()
bin_edges = np.arange(0,25,0.25)
plt.hist(B1, bins=bin_edges)
plt.xlabel('Fatal $(F)$ counts')
plt.ylabel('Bootstrap runs')
plt.title(f'Percentile bootstrap ~ (F/S): F={F}, S={S}')
#plt.xlim([0, 1.001])
#plt.xticks(np.arange(0, max(bin_edges), 0.1))
plt.legend([f'mean = {np.mean(B1):.2}, std = {np.std(B1):.2}'])
plt.savefig('./figs/bootstrap/bootstrap_B1.pdf', bbox_inches='tight')

fig = plt.figure()
plt.hist(B2, bins=160)
plt.xlabel('Infected $(I)$ counts')
plt.ylabel('Bootstrap runs')
plt.title(f'Percentile bootstrap ~ (I/T): I={I}, T={T}')
#plt.xlim([0, 1.001])
#plt.xticks(np.arange(0, max(bin_edges), 0.1))
plt.legend([f'mean = {np.mean(B2):.3}, std = {np.std(B2):.3}'])
plt.savefig('./figs/bootstrap/bootstrap_B2.pdf', bbox_inches='tight')

print(__name__ + ' done!')
