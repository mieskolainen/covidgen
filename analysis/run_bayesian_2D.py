# Bayesian 2D-analysis
# of the double binomial ratio
# 
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import matplotlib.pyplot as plt
import numba
import sys
import os

from tqdm import tqdm
from scipy.integrate import simps
from scipy.integrate import quad
from scipy.integrate import dblquad


sys.path.append('./analysis')
sys.path.append('./covidgen')

import estimators
import aux
import tools

# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)


plotfolder = './figs/bayesian'


# Note the order of matrix indices [j,i] !
#
#numba.njit
def get_P_matrix(p1val, p2val, *args):

    P  = np.zeros((len(p1val), len(p2val)))

    for i in tqdm(range(len(p1val))):
        for j in range(len(p2val)):
            P[j,i]  = estimators.binom_post_2D_alt(p1=p1val[i], p2=p2val[j], \
                k1=k1,n1=n1, k2=k2,n2=n2, B1=B1,B2=B2, alpha1=alpha1,beta1=beta1, alpha2=alpha2,beta2=beta2)
    return P

percent        = 100
TEST_INTEGRALS = False

# ------------------------------------------------------------------------
# Parameter setup

k1 = 7
n1 = 12597
k2 = 138
n2 = 919

print(f'k1/n1 = {k1/n1}')
print(f'k2/n2 = {k2/n2}')

# p1, p2 in [0,1]
p1val = np.linspace(0, np.min([10 * k1/n1, 1.0]), 2000)
p2val = np.linspace(0, np.min([10 * k2/n2, 1.0]), 2000)


# Beta prior
# (1,1)      for the flat prior
# (0.5, 0.5) for Jeffrey's prior
# (0,0)      for Haldane's prior
alpha = 0.5
beta  = 0.5

# Both the numerator and denominator the same prior
alpha1 = alpha2 = alpha
beta1  = beta2  = beta

# Pre-calculate Beta function parts for speed
B1 = estimators.betabinom_B(k=k1,n=n1, alpha=alpha1,beta=beta1)
B2 = estimators.betabinom_B(k=k2,n=n2, alpha=alpha2,beta=beta2)

args = {
    'k1': k1,
    'n1': n1, 
    'k2': k2,
    'n2': n2, 
    'B1': B1,
    'B2': B2, 
    'alpha1': alpha1,
    'beta1' : beta1,
    'alpha2': alpha2,
    'beta2' : beta2
}

args_ = (k1,n1, k2,n2, B1,B2, alpha1,beta1, alpha2,beta2)

# ------------------------------------------------------------------------
### Test normalization integrals

if TEST_INTEGRALS:
    I1   = quad(estimators.binom_post_alt, a=0, b=0.01, args=(k1,n1, B1, alpha1, beta1))
    print(f'Normalization integral 1D: {I1}')

    N    = int(1E8)
    val,err = tools.MC_integral_2D(N, estimators.binom_post_2D_alt, args_)
    print(f'Normalization integral 2D: {val:.3} +- {err:.3}')


# ------------------------------------------------------------------------
### Plot posteriori contours

fig,ax = plt.subplots(1,1)

P      = get_P_matrix(p1val, p2val, *args)
X,Y    = np.meshgrid(p1val*percent, p2val)
logP   = np.log(P + 1E-15)

cc   = plt.imshow(P, origin='lower',
           extent=(p1val[0]*percent, p1val[-1]*percent, p2val[0], p2val[-1]),
           cmap=plt.cm.gray,
           aspect='auto')
cbar = fig.colorbar(cc)

ax.set_xlim([0.15*(k1/n1)*percent, 2.4*(k1/n1)*percent])
ax.set_ylim([0.75*(k2/n2), 1.25*(k2/n2)])

CR = estimators.contour_cumsum(logP) # Create Credible Regions


# ------------------------------------------------------------------------
# Masked integral credible region test

if TEST_INTEGRALS:
    mask68 = CR < 0.68
    mask95 = CR < 0.95

    N = 10000000
    
    W,W2 = tools.MC_integral_2D_mask(N, estimators.binom_post_2D_alt, mask68, p1val, p2val, args_)
    print(f'Masked CR68 2D integral test: {W} +- {W2}')
    
    W,W2 = tools.MC_integral_2D_mask(N, estimators.binom_post_2D_alt, mask95, p1val, p2val, args_)
    print(f'Masked CR95 2D integral test: {W} +- {W2}')


# ------------------------------------------------------------------------
# Plot 2D-posteriori distribution with Credible Regions

CS = ax.contour(X, Y, CR, levels=(0.005, 0.683, 0.95))

fmt = {}
strs = ['MAP', 'CR 68', 'CR 95']
for l, s in zip(CS.levels, strs):
    fmt[l] = s

ax.clabel(CS, inline=1, fontsize=10, fmt=fmt)
ax.set_xlabel('$p_1 \\times 100$')
ax.set_ylabel('$p_2$')
#ax.set_title(f'Joint posterior: $P(p_1,p_2 \\, | \\, k_1={k1},n_1={n1},k_2={k2},n_2={n2})$')


os.makedirs(f'{plotfolder}', exist_ok = True)
plt.savefig(f'{plotfolder}/bayesian_contours.pdf', bbox_inches='tight')

print(__name__ + f' plots done under <{plotfolder}>')
