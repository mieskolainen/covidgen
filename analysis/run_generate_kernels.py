# Generate time convolution kernels
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import sys
import os
import pickle

sys.path.append('./analysis')
sys.path.append('./covidgen')

import tools
import functions
import cstats

def covid_kernels_init(t):

    ### https://www.medrxiv.org/content/10.1101/2020.06.10.20127423v1

    # Qifang Bi et al. "Epidemiology and transmission of COVID-19 in 391 cases and
    # 1286 of their close contacts in Shenzhen, China: a retrospective cohort study".
    # The Lancet Infectious Diseases (2020).
    # https://www.medrxiv.org/content/10.1101/2020.03.03.20028423v3

    # Jeremie Scire et al. "Reproductive number of the COVID-19 epidemic in Switzerland
    # with a focus on the Cantons of Basel-Stadt and Basel-Landschaft".
    # Swiss Medical Weekly 150.19-20 (2020).
    # https://smw.ch/article/doi/smw.2020.20271
    
    # Silvia Stringhini et al. "Seroprevalence of anti-SARS-CoV-2 IgG antibodies
    # in Geneva, Switzer-land (SEROCoV-POP): a population-based study".
    # The Lancet (2020).
    # https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)31304-0/fulltext

    ### Distribution mean values [units of days]
    mu = {
        'I2O' : 5.94,  # Incubation (Exposure to onset)
        'O2C' : 5.60,  # Symptom onset to reporting
        'O2S' : 11.2,  # Symptom onset to seroconversion
        'C2F' : 11.9   # Reporting to death
    }

    ### Distribution STD values [units of days]
    sigma = {
        'I2O' : 4.31,  # Incubation (Exposure to onset)
        'O2C' : 4.20,  # Symptom onset to reporting
        'O2S' : 4.40,  # Symptom onset to seroconversion
        'C2F' : 12.7   # Reporting to death
    }

    # Crude 1 sigma uncertainty [units of days]
    # (approximately relative 20 % for each)
    mu_std = {
        'I2O' : 1,
        'O2C' : 1,
        'O2S' : 2,
        'C2F' : 2
    }
    
    # Crude 1 sigma uncertainty [units of days]
    # (approximately relative 20 % for each)
    sigma_std = {
        'I2O' : 1,
        'O2C' : 1,
        'O2S' : 1,
        'C2F' : 2.5
    }

    W_a = {}
    W_k = {}

    # Conversion to Weibull parameters
    for key in mu.keys():
        W_a[key], W_k[key] = tools.get_weibull_param(mu[key], sigma[key])

    # Generate kernels with mean values
    K = cstats.covid_kernels(t, mu=mu, sigma=sigma, mu_std=None, sigma_std=None)

    # Do the approximate conversion to Weibull
    for key in ['C', 'S', 'F']:
        mu[key], sigma[key] = tools.get_f_mean_sigma(t, K[key])
        W_a[key], W_k[key]  = tools.get_weibull_param(mu[key], sigma[key], 'moments')

    param = {
        't'        : t,
        'K'        : K,
        'mu'       : mu,
        'sigma'    : sigma,
        'mu_std'   : mu_std,
        'sigma_std': sigma_std,
        'W_a'      : W_a,
        'W_k'      : W_k
    }

    return param


# Save output
os.makedirs('./output/', exist_ok = True)


### Fine discretization
filename = './output/kernels_fine.pkl'
with open(filename, 'wb') as f:

    lastday = 199
    interpolation = 10
    t = np.linspace(0, lastday, (lastday+1)*interpolation)
    param = covid_kernels_init(t)
    param['interpolation'] = interpolation
    pickle.dump(param, f)

print(f'Time delay kernels saved to {filename}')


### Daily discretization
filename = './output/kernels_daily.pkl'
with open(filename, 'wb') as f:

    lastday = 199
    interpolation = 1
    t = np.linspace(0, lastday, (lastday+1)*interpolation)
    param = covid_kernels_init(t)
    param['interpolation'] = interpolation
    pickle.dump(param, f)

print(f'Time delay kernels saved to {filename}')


### Fine discretization and max day 100
filename = './output/kernels_fine_max_100.pkl'
with open(filename, 'wb') as f:

    lastday = 100
    interpolation = 10
    t = np.linspace(0, lastday, (lastday+1)*interpolation)
    param = covid_kernels_init(t)
    param['interpolation'] = interpolation
    pickle.dump(param, f)

print(f'Time delay kernels saved to {filename}')

print(__name__ + ' done!')
