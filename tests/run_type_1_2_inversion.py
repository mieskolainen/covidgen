# Type I and II test error inversion
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import numba

# Import local path
import sys
sys.path.append('./covidgen')

from aux import *
import estimators
percent = 100

# ------------------------------------------------------------------------
# Global prevalance test data (#corrected counts, #tests)

k = np.array([13, 35, 50, 12, 13, 138, 84, 171, 33, 18, 26])
N = np.array([388, 863, 3330, 1224, 2283, 919, 775, 2482, 1742, 707, 824])

# ------------------------------------------------------------------------
# Imperial Report 34 collection

## Serology Sensitivity data
v_arr = np.array([85.14, 82.09, 78.4, 96.04, 98.28, 81.84, 99.36, 91.16, 90.74, 89.39]) / 100

## Specificity data
s_arr = np.array([99.72, 99.25, 99.44, 99.7, 99.65, 98.79, 98.89, 100.0, 99.89, 98.73]) / 100

# ------------------------------------------------------------------------


## Compute the global mean
v = np.mean(v_arr)
s = np.mean(s_arr)


# Error estimate on the mean values
dv = np.std(v_arr) / np.sqrt(len(v_arr))
ds = np.std(s_arr) / np.sqrt(len(s_arr))


print('----------------')
print(f'mean specificity s: {s:0.6} +- {ds:0.6e}  [{ds/s*100:0.3e}] %')
print(f'mean sensitivity v: {v:0.6} +- {dv:0.6e}  [{dv/v*100:0.3e}] %')
print('----------------')
print('\n')

percent = 100

for i in range(len(k)):	
	
	# Test by assuming k is corrected counts
	dp_new, dp_orig = estimators.renormalize_test12_error_corrected_input(k=k[i], N=N[i], s=s,v=v, ds=ds, dv=dv)

	# Test by assuming k is raw counts
	dp_new, dp_orig = estimators.renormalize_test12_error_raw_input(k=k[i], N=N[i], s=s,v=v, ds=ds, dv=dv)

	print('----------------')
	print('\n\n')



