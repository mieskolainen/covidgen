# Analyze & plot simulation results
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import pickle

# Import local path
import sys

sys.path.append('./analysis')
sys.path.append('./covidgen')

import covidgen
from bernoulli import *
from observables import *
from aux import *

# Font style
import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)


output = 'output'

# Load events
filename = './output/output.covmc'
with open(filename, 'rb') as f:
	B,B3,args = pickle.load(f)

print(f'Simulation data loaded from {filename}')

## Compute observables
OBS = get_observables(B=B, B3=B3)

## Print statistics
print_statistics(OBS=OBS, B=B, B3=B3)

## Produce histograms
plot_histograms(OBS=OBS)

