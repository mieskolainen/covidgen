# Bernoulli MC simulation steering loop
#
# m.mieskolainen@imperial.ac.uk, 2020

import os
import numpy as np

R    = 10000 				   # Number of MC simulations per input
FVAL = np.linspace(0.1,21,500) # Continuum of deaths => continuum bernoulli parameter

for i in range(len(FVAL)):
	F = FVAL[i]
	os.system(f'python ./covidgen/sim.py --R {R} --F_N {F} --output F_step_{i}')

print(__name__ + ' done!')
