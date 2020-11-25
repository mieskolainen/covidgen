# COVID statistics event generator (simulator) based on Monte Carlo sampling
# with 3-dimensional Bernoulli distributions
# 
# 
# Run with:                python main.py
# Get help with:           python main.py --help
#
# Install packages with:   pip install numpy numba matplotlib
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import numba
import time
import argparse
import pickle
import os

# Import local path
import sys

sys.path.append('./analysis')
sys.path.append('./covidgen')

import covidgen
from bernoulli import *
from observables import *
from aux import *



def main():
	"""
	Main steering function.
	"""

	### Common simulation parameters
	print(f'COVIDGEN v{covidgen.VER}')
	parser = argparse.ArgumentParser(description=f'COVIDGEN v{covidgen.VER}')

	parser.add_argument('--R',      default=100000,   help='Number of MC runs')
	parser.add_argument('--N',      default=12597,    help='Number of people in the city')
	parser.add_argument('--F_N',    default=7,        help='Number of deaths in the city')
	parser.add_argument('--T',      default=919,      help='Number of people in the test sample')
	parser.add_argument('--I_T',    default=138,      help='Number of infected in the test sample')


	parser.add_argument('--rho',    default='max',    help='<Infected,Fatal> correlation coupling mode [''max'',''min'',''0'']')
	parser.add_argument('--C',      default=1,        help='Boundary conditions on')
	parser.add_argument('--fixT',   default=1,        help='Number of people tested is constant')
	
	
	parser.add_argument('--seed',   default=1234,     help='Random number seed (int32)')
	parser.add_argument('--output', default='output', help='Output filename')


	## Parse CLI arguments
	args = parser.parse_args()

	R         = int(args.R)

	N         = int(args.N)
	T         = int(args.T)
	I_T       = float(args.I_T) # Keep it float to be able to do uncertainty scans
	F_N       = float(args.F_N) # Keep it float to be able to do uncertainty scans
	rhomode   = str(args.rho)
	constrain = int(args.C)
	fixT      = int(args.fixT)


	if R < 1: raise Exception('Error: R < 1 (number of MC runs)')

	### Seed the random number generator
	seed = int(args.seed)
	np.random.seed(seed)
	print(f'rngseed = {seed}')
	
	t0 = time.time()

	# ------------------------------------------------------------------------
	### Check the input feasibility domain before the event generation

	p_T, p_I, p_F  = covidgen.bernoulli_param(N=N, T=T, I_T=I_T, F_N=F_N)
	minrho, maxrho = bernoulli2_rhorange(EX=p_I, EY=p_F)

	print('\n')
	print(f'Computed Maximum Likelihood Bernoulli parameters:')
	print(f' p_T = {p_T:.9f}')
	print(f' p_I = {p_I:.9f}')
	print(f' p_F = {p_F:.9f}')
	print('\n')

	print(f'Probability conservation limited coupling range:')
	print(f' rho = [{minrho:.9f}, {maxrho:.9f}]')

	# Set coupling
	EPS = 1E-6 # Numerical protection
	if   rhomode == 'min':
		rho = np.min([0.0, minrho + np.abs(minrho)*EPS])
	elif rhomode == 'max':
		rho = np.max([0.0, maxrho - np.abs(minrho)*EPS])
	elif rhomode == '0':
		rho = 0.0
	else:
		raise Exception(f'Unknown rho mode parameter = {rhomode}')

	print('\n')
	print(f'Rho coupling mode = {rhomode}')

	# ------------------------------------------------------------------------
	### Simulation and analysis of observables

	B   = dict()
	B3  = dict()
	OBS = dict()

	t = time.time()

	print(f'\nSimulation running ... \n')

	## Run simulation
	try:
		B,B3 = covidgen.simulation(N=N, T=T, I_T=I_T, F_N=F_N, R=R, rho=rho, constrain=constrain, fixT=fixT)
	except:
		print("Unexpected error in the simulation:", sys.exc_info()[0])
		exit()

	print(f'Simulation took {time.time() - t:.1f} sec\n')
	printbar('=')
	print(' SIMULATION OUTPUT')
	printbar('='); print('\n')
			
	## Save events
	os.makedirs('./output/', exist_ok = True)

	filename = './output/' + args.output + '.covmc'
	with open(filename, 'wb') as f:
		pickle.dump([B, B3, args], f)
	print(f'Simulation data saved to {filename}')
	
	print('\n')
	print(f'Full process took {time.time() - t0:.1f} sec\n')


if __name__ == '__main__' :
	main()
