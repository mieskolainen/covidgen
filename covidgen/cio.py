# COVID dataset input readers
#
# m.mieskolainen@imperial.ac.uk, 2020

import sys
import numpy as np
from datetime import datetime,timedelta
from termcolor import colored
import os
import pandas as pd

#from datetime import datetime
#a = datetime.strptime(dt[0], '%Y-%m-%d')

def todiff(series):
    """
    Turn cumulative series into differential
    """
    series = np.diff(series, prepend=0)

    # Fix possible NaN
    series[~np.isfinite(series)] = 0
    
    # Fix possible errors in data (cumulative were not monotonic)
    ind = series < 0
    if np.sum(series[ind]) != 0:
        print(colored(f'{__name__}.todiff: fixing non-monotonic input (negative dx set to 0)', 'red'))
        print(series)
        series[ind] = 0
    return series


def data_processor(meta):
    """ 
    Dataset processor wrapper
    """
    evalstr = f"{meta['function']}(meta)"
    print(evalstr)
    try:
        d = eval(evalstr)
        return d
    except:
        print(__name__ + f".data_processor: {colored('Failed to process','yellow')} {meta['isocode']}")
        print(f'Error: {sys.exc_info()[0]} {sys.exc_info()[1]}')


def get_isocodes():

    isodata   = pd.read_csv('./data/iso.csv', comment='#')
    code      = np.array(isodata['code'])
    return code


def get_european_isocodes():

    isodata   = pd.read_csv('./data/iso.csv', comment='#')
    code      = np.array(isodata['code'])
    continent = np.array(isodata['continent'])
    return code[continent == 4] # Europe only


def data_reader_swiss(meta):
    """
    Swiss data format reader
    """
    # --------------------------------------------------------------------
    # DEATHS

    df = pd.read_csv('./data/' + meta['filename_deaths'], comment='#')
    df = df.sort_index(ascending=meta['ascending'], axis=0)

    d  = {}
    d['dt']     = np.array(df["Date"])    

    # Turn cumulative into daily
    d['deaths'] = todiff(df[meta['region']])

    # --------------------------------------------------------------------
    # Cases

    df = pd.read_csv('./data/' + meta['filename_cases'], comment='#')
    df = df.sort_index(ascending=meta['ascending'], axis=0)

    # Turn cumulative into daily
    d['cases']  = todiff(df[meta['region']])


    # --------------------------------------------------------------------
    # Tests

    df = pd.read_csv('./data/' + meta['filename_tested'], comment='#')
    df = df.sort_index(ascending=meta['ascending'], axis=0)

    # Turn cumulative into daily
    d['tests']  = todiff(df[meta['region']])

    # --------------------------------------------------------------------
    d['population'] = meta['population']
    d['isocode']    = meta['isocode']

    # --------------------------------------------------------------------

    if (len(d['deaths']) != len(d['cases'])):
        raise Exception(__name__ + '.data_reader_swiss: len(deaths) != len(cases)')
    if (len(d['cases'])  != len(d['tests'])):
        raise Exception(__name__ + '.data_reader_swiss: len(cases) != len(tests)')

    return d


def data_reader_sweden(meta):

    d = {}
    d['isocode']    = meta['isocode']
    d['population'] = meta['population']

    df              = pd.read_csv('./data/' + meta['filename_cases'], comment='#')
    df              = df.loc[df["Region"] == meta['region']]

    # --------------------------------------------------------------------
    # Iterating the columns, find date columns

    dt=list()
    for col in df.columns: 
        if "2020-" in col:
            dt.append(col)
    d['dt'] = dt

    # --------------------------------------------------------------------
    # Cases

    d['cases'] = np.array(df[dt])[0]
    
    # --------------------------------------------------------------------
    # Deaths

    df = pd.read_csv('./data/' + meta['filename_deaths'], comment='#')
    df = df.loc[df["Region"] == meta['region']]

    d['deaths'] = np.array(df[dt])[0]

    # --------------------------------------------------------------------
    # Tests

    # ** NOT AVAILABLE **
    d['tests']   = np.zeros(len(dt))*np.nan

    return d

def data_reader_usa(meta):

    d = {}
    d['population'] = meta['population']
    d['isocode']    = meta['isocode']
    
    # --------------------------------------------------------------------
    # Deaths
    
    df          = pd.read_csv('./data/' + meta['filename'], comment='#')
    df          = df.loc[df["county"] == meta['region']]
    
    d['dt']     = np.array(df['date'])
    d['deaths'] = todiff(df['deaths'])

    # --------------------------------------------------------------------
    # Cases
    d['cases']  = todiff(df['cases'])

    # --------------------------------------------------------------------
    # Tests
    d['tests']  = np.zeros(len(d['dt']))*np.nan

    return d

def data_reader_heinsberg(meta):

    d = {}
    d['population'] = meta['population']
    d['isocode']    = meta['isocode']
    
    # Cases data
    #df              = pd.read_csv('./data/' + meta['filename_cases'], comment='#')
    #data            = df.loc[df["county"] == meta['region']]

    # --------------------------------------------------------------------
    # Deaths
    
    df          = pd.read_csv('./data/' + meta['filename_deaths'], comment='#')

    d['dt']     = np.array(df['date'])
    d['deaths'] = np.array(df['deaths'])

    # --------------------------------------------------------------------
    # Cases

    d['cases']  = np.zeros(len(d['dt']))*np.nan

    # --------------------------------------------------------------------
    # Tests

    d['tests']  = np.zeros(len(d['dt']))*np.nan

    return d

def data_reader_florida(meta):

    d = {}
    d['population'] = meta['population']
    d['isocode']    = meta['isocode']
    
    # Cases data
    #df              = pd.read_csv('./data/' + meta['filename_cases'], comment='#')
    #data            = df.loc[df["county"] == meta['region']]

    # --------------------------------------------------------------------
    # Deaths
    
    df          = pd.read_csv('./data/' + meta['filename_deaths'], comment='#')

    d['dt']     = np.array(df['date'])
    d['deaths'] = np.array(df['deaths'])

    # --------------------------------------------------------------------
    # Cases

    d['cases']  = np.zeros(len(d['dt']))*np.nan #np.array(data["frequency"])

    # --------------------------------------------------------------------
    # Tests

    d['tests']  = np.zeros(len(d['dt']))*np.nan

    return d


def data_reader_LA(meta):
    """
    LA County data format reader
    """
    df = pd.read_csv('./data/' + meta['filename'], comment='#')
    df = df.sort_index(ascending=meta['ascending'], axis=0)

    d  = {}
    d['dt']         = np.array(df["date_dt"])
    d['cases']      = np.array(df["new_case"])
    d['deaths']     = np.array(df["new_deaths"])
    d['tests']      = np.array(df['new_persons_tested'])
    d['population'] = meta['population']
    d['isocode']    = meta['isocode']

    return d


def data_reader_OWID(meta):
    """
    World-in-data format reader
    """
    df = pd.read_csv('./data/' + meta['filename'],  comment='#')
    df = df.sort_index(ascending=meta['ascending'], axis=0)

    # Take the isocode
    data            = df.loc[df["iso_code"] == meta['isocode']]

    d  = {}
    d['dt']         = np.array(data["date"])
    d['cases']      = np.array(data["new_cases"])
    d['deaths']     = np.array(data["new_deaths"])
    d['tests']      = np.array(data["new_tests_smoothed"])
    d['population'] = np.array(data["population"])[0]
    d['isocode']    = meta['isocode']

    return d


def choose_timeline(data, target_key='deaths', first_date='2020-01-01', last_date=None):
    """
    Choose a particular timeline of a time-series
    """
    print(__name__ + f'.choose_timeline: Trying timeline: [{first_date}, {last_date}] with key = {target_key}')

    if data is None:
        raise Exception(__name__ + f'.choose_timeline: input data is None')

    firstind = np.where(np.isfinite(data[target_key]))
    firstind = firstind[0][0]

    try:
        firstind_alt = np.where(data['dt'] == first_date)[0][0]
        if firstind_alt > firstind:
            firstind = firstind_alt
    except:
        print(__name__ + f'.choose_timeline: {colored("Not able to pick the first date:","yellow")} {first_date}')
        firstind = firstind

    lastind      = np.where(np.isfinite(data[target_key]))
    lastind      = lastind[0][-1]

    # Apply last date
    if last_date is not None:
        try:
            lastday = np.where(data['dt'] == last_date)[0][0]
            lastind = np.min([lastind, lastday])
        except:
            print(__name__ + f'.choose_timeline: {colored("Not able to pick the last date:","yellow")} {last_date}')
            lastind = lastind

    ### Select time-series
    data['deaths']  = data['deaths'][firstind:lastind+1]
    data['cases']   = data['cases'] [firstind:lastind+1]
    data['tests']   = data['tests'] [firstind:lastind+1]
    data['dt']      = data['dt']    [firstind:lastind+1]
    
    print(__name__ + f".choose_timeline: Timeline obtained: [{data['dt'][0]}, {data['dt'][-1]}]")

    return data
