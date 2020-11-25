# COVID-19 datasets
#
# m.mieskolainen@imperial.ac.uk, 2020

'''
# Miami-Dade County
# https://rwilli5.github.io/MiamiCovidProject
MIA = {
    'filename_cases'  : '',
    'filename_deaths' : 'covid-19-miami-dade_deaths-by-day.csv',
    'filename_tested' : '',
    'region'          : 'Dade', # Miami-Dade
    'isocode'         : 'MIA',
    'function'        : 'data_reader_florida',
    'ascending'       : True,
    'population'      : 2716940, # Wikipedia
}
'''

# Miami-Dade County
# https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv
MIA = {
    'filename'        : 'us-counties.csv',
    'region'          : 'Miami-Dade',
    'isocode'         : 'MIA',
    'function'        : 'data_reader_usa',
    'ascending'       : True,
    'population'      : 2716940, # Wikipedia
}

# Santa-Clara County
# https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv
SCC = {
    'filename'        : 'us-counties.csv',
    'region'          : 'Santa Clara',
    'isocode'         : 'SCC',
    'function'        : 'data_reader_usa',
    'ascending'       : True,
    'population'      : 1928000, # Wikipedia
}

# New York City Metropolitan county
# https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv
NYC = {
    'filename'        : 'us-counties.csv',
    'region'          : 'New York City',
    'isocode'         : 'NYC',
    'function'        : 'data_reader_usa',
    'ascending'       : True,
    'population'      : 19979477, # Wikipedia
}

# San Francisco (Bay Area) county
# https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv
SFR = {
    'filename'        : 'us-counties.csv',
    'region'          : 'San Francisco',
    'isocode'         : 'SFR',
    'function'        : 'data_reader_usa',
    'ascending'       : True,
    'population'      : 883305, # Wikipedia
}

# Philadelphia metro (Pennsylvania)
# https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv
PHI = {
    'filename'        : 'us-counties.csv',
    'region'          : 'Philadelphia',
    'isocode'         : 'PHI',
    'function'        : 'data_reader_usa',
    'ascending'       : True,
    'population'      : 1584000, # Wikipedia
}

'''
# Los Angeles
# https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv
LAC = {
    'filename'        : 'us-counties.csv',
    'region'          : 'Los Angeles',
    'isocode'         : 'LAC',
    'function'        : 'data_reader_usa',
    'ascending'       : True,
    'population'      : 10039107, # Wikipedia
}
'''

# Los Angeles County
# https://covid19.lacounty.gov
LAC = {
    'filename'  : 'date_table_LA_county.csv', # tests,cases,deaths
    'isocode'   : 'LAC',
    'region'    : 'Los Angeles',
    'function'  : 'data_reader_LA',
    'ascending' : False,
    'population': 10039107, # Wikipedia
}


# Stockholm metropolitan county
# https://www.kaggle.com/jannesggg/sweden-covid19-dataset
STK = {
    'filename_cases'  : 'datasets_583052_1362315_time_series_confimed-confirmed.csv',
    'filename_deaths' : 'datasets_583052_1362315_time_series_deaths-deaths.csv',
    'filename_tested' : '',
    'region'          : 'Region Stockholm',
    'isocode'         : 'STK',
    'function'        : 'data_reader_sweden',
    'ascending'       : True,
    'population'      : 2370000, # Stockholm county
}

# Geneva Region
# OpenZH
GVA = {
    'filename_cases'  : 'covid19_cases_switzerland_openzh.csv',
    'filename_deaths' : 'covid19_fatalities_switzerland_openzh.csv',
    'filename_tested' : 'covid19_tested_switzerland_openzh.csv',
    'isocode'         : 'GVA',
    'region'          : 'GE',
    'function'        : 'data_reader_swiss',
    'ascending'       : True,
    'population'      : 499480, # Wikipedia
}

# Finland
# OWID
FIN = {
    'filename'  : 'owid-covid-data.csv',  # tests,cases,deaths
    'isocode'   : 'FIN',
    'region'    : 'Finland',
    'function'  : 'data_reader_OWID',
    'ascending' : True,
    'population': 5528737, # Wikipedia
}

# Gangelt of Heinsberg
# Manually constructed from the paper
GAN = {
    'filename_deaths' : 'covid-19-heinsberg-deaths-by-day.csv',  # tests,cases,deaths
    'isocode'   : 'GAN',
    'region'    : 'Gangelt',
    'function'  : 'data_reader_heinsberg',
    'ascending' : True,
    'population': 12597, # Paper
}

# Iceland
# OWID
ISL = {
    'filename'  : 'owid-covid-data.csv',  # tests,cases,deaths
    'isocode'   : 'ISL',
    'region'    : 'Iceland',
    'function'  : 'data_reader_OWID',
    'ascending' : True,
    'population': 364134, # Wikipedia
}

# Switzerland
# OWID
CHE = {
    'filename'  : 'owid-covid-data.csv',  # tests,cases,deaths
    'isocode'   : 'CHE',
    'region'    : 'Switzerland',
    'function'  : 'data_reader_OWID',
    'ascending' : True,
    'population': None, # Read-from-OWID-file
}

# Template for OWID data
TEMPLATE = {
    'filename'  : 'owid-covid-data.csv',  # tests,cases,deaths
    'isocode'   : None,
    'region'    : None,
    'function'  : 'data_reader_OWID',
    'ascending' : True,
    'population': None, # Read-from-OWID-file
}
