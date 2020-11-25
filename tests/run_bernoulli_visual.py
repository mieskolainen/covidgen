# Visualize feasible domain of correlated Bernoulli random variables
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib 
matplotlib.rc('xtick', labelsize=6) 
matplotlib.rc('ytick', labelsize=6)


# Import local path
import sys
sys.path.append('./covidgen') 

# Font style
#import aux
#import matplotlib; matplotlib.rcParams.update(aux.tex_fonts)

from bernoulli import *


EXval  = np.linspace(1e-3, 1, 100)
EYval  = np.linspace(1e-3, 1, 100)
rhoval = np.linspace(-0.99, 0.99, 3*4)


fig = plt.figure()

# Loop over correlations
for k in range(len(rhoval)):

	# Phase-space matrix
	Z = np.zeros((len(EXval), len(EYval)))

	# Loop over mean values
	for i in range(len(EXval)):
		for j in range(len(EYval)):
			Z[i,j] = bernoulli2_is_valid(EXval[i], EYval[j], rhoval[k])

	plt.subplot(3, 4, k+1)
	plt.imshow(Z, cmap=plt.cm.Greys_r)
	plt.xticks([0, 0.5, 1.0])
	x_label_list = ['0', '0.25', '0.5', '0.75', '1']

	ax = plt.gca()
	ax.set_xticks([0, 25, 50, 75, 99])
	ax.set_xticklabels(x_label_list)

	ax.set_yticks([0, 25, 50, 75, 99])
	ax.set_yticklabels(x_label_list)

	plt.xlabel('$\\langle X \\rangle$')
	plt.ylabel('$\\langle Y \\rangle$')
	plt.gca().invert_yaxis()
	plt.title(f'$\\rho={rhoval[k]:.2}$')

# Save
fig.tight_layout(pad=0.9)
os.makedirs('./figs/', exist_ok = True)
plt.savefig('./figs/bernoulli_2D.pdf', bbox_inches='tight')

print(__name__ + ' done!')
