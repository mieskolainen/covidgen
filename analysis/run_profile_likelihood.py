# Profile (log)-likelihood ratio analysis
# for the double binomial ratio
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import matplotlib.pyplot as plt
import os

# Import local path
import sys

sys.path.append('./analysis')
sys.path.append('./covidgen')

from aux import *


# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)
import estimators as est

# Measurements
k1 = 7
n1 = 12579
k2 = 138
n2 = 919

r_MLE = (k1/n1) / (k2/n2)

# Find the confidence intervals
CI68, r0, LLR, chi2_68 = est.profile_LLR_binom_ratio_err(k1=k1,n1=n1, k2=k2,n2=n2, alpha=1-0.68, return_full=True)
CI95, r0, LLR, chi2_95 = est.profile_LLR_binom_ratio_err(k1=k1,n1=n1, k2=k2,n2=n2, alpha=1-0.95, return_full=True)

# Plot them
percent = 100
print(f'CI95 = [{CI95[0]*percent:.2f}, {CI95[1]*percent:.2f}]')

fig,ax = plt.subplots()
plt.plot(r0 * percent, LLR, color=(0,0,0), label=f'$\\mathrm{{IFR}}_{{MLE}} = {r_MLE*percent:.2f}$ %  CI68: [{CI68[0]*percent:.2f}, {CI68[1]*percent:.2f}] %  CI95: [{CI95[0]*percent:.2f}, {CI95[1]*percent:.2f}] %')

plt.plot(np.ones(2)*CI95[0] * percent, np.array([-2, 10]), color='r', linestyle='dashed')
plt.plot(np.ones(2)*r_MLE   * percent, np.array([-2, 10]), color=(0,0,0), linestyle='dashed')
plt.plot(np.ones(2)*CI95[1] * percent, np.array([-2, 10]), color='r', linestyle='dashed')

# add text
plt.text(CI95[1] * percent * 1.1, chi2_95 * 1.07, '$\\chi_{1,0.95}^2$', color=(0.5,0.5,0.5))

plt.plot(np.array([0, 100]), np.ones(2)*chi2_95, color=(0.5, 0.5, 0.5), linestyle='dotted')

plt.xlabel('Infection Fatality Rate ($r_0 \\times 100$) [%]')
plt.ylabel('Profile ln($L$) ratio $2[\\ln L(\\hat{r}, \\hat{p}_1) - \\ln L(r_0, p_1^*)]$')
plt.xlim([0, (2.5*r_MLE) * percent])
plt.xticks(np.linspace(0,1,11))
plt.legend(loc='upper right')
plt.ylim([-2, 10])
#plt.show()

os.makedirs('./figs/profile/', exist_ok = True)
plt.savefig('./figs/profile/profile.pdf', bbox_inches='tight')

print(__name__ + ' done!')
