# Datasets to use in the IFR analysis

import sys
sys.path.append('./dataconfig')

import datasets
import datasets_sero

'''
def default():
    
    CSETS = {
        'FIN' : datasets.FIN,
        'LAC' : datasets.LAC  
    }
    
    CSETS_sero = {
        'FIN' : datasets_sero.FIN,
        'LAC' : datasets_sero.LAC
    }
    
    return CSETS, CSETS_sero
'''

def default():
    
    CSETS = {
        'FIN' : datasets.FIN,
        'LAC' : datasets.LAC,
        'SCC' : datasets.SCC,
        'SFR' : datasets.SFR,
        'ISL' : datasets.ISL,
        'GAN' : datasets.GAN,
        'GVA' : datasets.GVA,
        'NYC' : datasets.NYC,
        'MIA' : datasets.MIA,
        'STK' : datasets.STK,
        'PHI' : datasets.PHI,   
    }
    
    CSETS_sero = {
        'FIN' : datasets_sero.FIN,
        'LAC' : datasets_sero.LAC,
        'SCC' : datasets_sero.SCC,
        'SFR' : datasets_sero.SFR,
        'ISL' : datasets_sero.ISL,
        'GAN' : datasets_sero.GAN,
        'GVA' : datasets_sero.GVA,
        'NYC' : datasets_sero.NYC,
        'MIA' : datasets_sero.MIA,
        'STK' : datasets_sero.STK,
        'PHI' : datasets_sero.PHI,
    }
    
    return CSETS, CSETS_sero
