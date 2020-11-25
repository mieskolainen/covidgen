# COVID-19 seroprevalance datasets
#
# Under datasets.py, one should have the corresponding fatality datacards.
# 
# Usage instructions:
#
#   If 'corrected' : False, then type I/II test error corrections done according
#                    to the sensitivity and specificity as specified.
# 
# m.mieskolainen@imperial.ac.uk, 2020


# South Florida County
# https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2768834
MIA = {
    'corrected' : True,
    
	'positive'  : 33, # From 1.9 % (age, type I/II errors and others adjusted)
	'tested'    : 1742,
	'age'       : 'all',
    'test_date' : ['2020-04-06', '2020-04-10'], # year-month-day
    'test_type' : 'IgG',
    'test_tex'  : '10.1001/jamainternmed.2020.4130',
    
    # Average default values from Imperial report 34
    'specificity'       : 0.99406,
    'specificity_error' : 0.00140, # abs error, std/sqrt(N)

    'sensitivity'       : 0.89244,
    'sensitivity_error' : 0.02183,  # abs error, std/sqrt(N)

    # Delay kernels
    'kernel_path' : './output/kernels_daily.pkl'
}


# New York City
# https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2768834
NYC = {
    'corrected' : True,

	'positive'  : 171, # From 6.9 % (age, type I/II errors and others adjusted)
	'tested'    : 2482,
	'age'       : 'all',
    'test_date' : ['2020-03-23', '2020-04-01'], # year-month-day
    'test_type' : 'IgG',
    'test_tex'  : '10.1001/jamainternmed.2020.4130',

    # Average default values from Imperial report 34
    'specificity'       : 0.99406,
    'specificity_error' : 0.00140, # abs error, std/sqrt(N)

    'sensitivity'       : 0.89244,
    'sensitivity_error' : 0.02183,  # abs error, std/sqrt(N)

    # Delay kernels
    'kernel_path' : './output/kernels_daily.pkl'
}

# San Francisco (Bay Area)
# https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2768834
SFR = {
    'corrected' : True,

	'positive'  : 12, # From 1.0 % (age, type I/II errors and others adjusted)
	'tested'    : 1224,
	'age'       : 'all',
    'test_date' : ['2020-04-23', '2020-04-27'], # year-month-day
    'test_type' : 'IgG',
    'test_tex'  : '10.1001/jamainternmed.2020.4130',

    # Average default values from Imperial report 34
    'specificity'       : 0.99406,
    'specificity_error' : 0.00140, # abs error, std/sqrt(N)

    'sensitivity'       : 0.89244,
    'sensitivity_error' : 0.02183,  # abs error, std/sqrt(N)


    # Delay kernels
    'kernel_path' : './output/kernels_daily.pkl'
}

# Philadelphia
# https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2768834
PHI = {
    'corrected' : True,

	'positive'  : 26, # From 3.2 % (age, type I/II errors and others adjusted)
	'tested'    : 824,
	'age'       : 'all',
    'test_date' : ['2020-04-13', '2020-04-25'], # year-month-day
    'test_type' : 'IgG',
    'test_tex'  : '10.1001/jamainternmed.2020.4130',

    # Average default values from Imperial report 34
    'specificity'       : 0.99406,
    'specificity_error' : 0.00140, # abs error, std/sqrt(N)

    'sensitivity'       : 0.89244,
    'sensitivity_error' : 0.02183,  # abs error, std/sqrt(N)


    # Delay kernels
    'kernel_path' : './output/kernels_daily.pkl'
}

# Santa Clara County
# https://www.medrxiv.org/content/10.1101/2020.04.14.20062463v2
SCC = {
    'corrected' : True,

	#'positive'  : 93, # 2.8 % (age, type I/II errors and others adjusted)
	'positive'  : 50, # Unadjusted
	'tested'    : 3330,
	'age'       : 'all',
    'test_date' : ['2020-04-03', '2020-04-04'], # year-month-day
    'test_type' : 'IgG/IgM',
    'test_tex'  : 'Bendavid2020.04.14.20062463',

    # Average default values from Imperial report 34
    'specificity'       : 0.99406,
    'specificity_error' : 0.00140, # abs error, std/sqrt(N)

    'sensitivity'       : 0.89244,
    'sensitivity_error' : 0.02183,  # abs error, std/sqrt(N)


    # Delay kernels
    'kernel_path' : './output/kernels_daily.pkl'
}

# Stockholm
# https://www.folkhalsomyndigheten.se/contentassets/53c0dc391be54f5d959ead9131edb771/infection-fatality-rate-covid-19-stockholm-technical-report.pdf
STK = {
    'corrected' : True,

	'positive'  : 18,
	'tested'    : 707,
	'age'       : 'all',
    'test_date' : ['2020-03-26', '2020-04-02'], # year-month-day
    'test_type' : 'PCR',
    'test_tex'  : 'stockholmstudy',

    # Average default values from Imperial report 34
    'specificity'       : 0.99406,
    'specificity_error' : 0.00140, # abs error, std/sqrt(N)

    'sensitivity'       : 0.89244,
    'sensitivity_error' : 0.02183,  # abs error, std/sqrt(N)

    # Delay kernels
    'kernel_path' : './output/kernels_daily.pkl'
}

# Geneva
# https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)31304-0/fulltext
GVA = {
    'corrected' : True,

	'positive'  : 84, # 10.8 %, last week result of the several week study
	'tested'    : 775,
	'age'       : 'all',
    'test_date' : ['2020-05-04', '2020-05-09'], # year-month-day
    'test_type' : 'IgG',
    'test_tex'  : 'Stringhini2020.05.02.20088898',

    # Average default values from Imperial report 34
    'specificity'       : 0.99406,
    'specificity_error' : 0.00140, # abs error, std/sqrt(N)

    'sensitivity'       : 0.89244,
    'sensitivity_error' : 0.02183,  # abs error, std/sqrt(N)

    # Delay kernels
    'kernel_path' : './output/kernels_daily.pkl'
}

# Los Angeles County
# https://jamanetwork.com/journals/jama/fullarticle/2766367
LAC = {
    'corrected' : True,

	'positive'  : 35,
	'tested'    : 863,
	'age'       : 'all',
    'test_date' : ['2020-04-10', '2020-04-11'], # year-month-day
    'test_type' : 'IgG/IgM',
    'test_tex'  : '10.1001/jama.2020.8279',

    # Average default values from Imperial report 34
    'specificity'       : 0.99406,
    'specificity_error' : 0.00140, # abs error, std/sqrt(N)

    'sensitivity'       : 0.89244,
    'sensitivity_error' : 0.02183,  # abs error, std/sqrt(N)

    # Delay kernels
    'kernel_path' : './output/kernels_daily.pkl'
}

# Finland
# https://www.thl.fi/roko/cov-vaestoserologia/sero_report_weekly_en.html
# Note: maximum age 69 in this data
# Weeks 23+24 data
FIN = {
    'corrected' : True,

	'positive'  : 13,  # 8 + 5
	'tested'    : 388, # 214 + 174
	'age'       : '0-69',
    'test_date' : ['2020-06-01', '2020-06-14'], # year-month-day
    'test_type' : 'IgG',
    'test_tex'  : 'finnish_thl',

    # Average default values from Imperial report 34
    'specificity'       : 0.99406,
    'specificity_error' : 0.00140, # abs error, std/sqrt(N)

    'sensitivity'       : 0.89244,
    'sensitivity_error' : 0.02183,  # abs error, std/sqrt(N)


    # Delay kernels
    'kernel_path' : './output/kernels_daily.pkl'
}

# Iceland
# https://www.nejm.org/doi/full/10.1056/NEJMoa2006100
ISL = {
    'corrected' : True,

	'positive'  : 13,
	'tested'    : 2283,
	'age'       : 'all',
    'test_date' : ['2020-04-04', '2020-04-04'], # year-month-day
    'test_type' : 'PCR',
    'test_tex'  : 'gudbjartsson2020spread',

    # Average default values from Imperial report 34
    'specificity'       : 0.99406,
    'specificity_error' : 0.00140, # abs error, std/sqrt(N)

    'sensitivity'       : 0.89244,
    'sensitivity_error' : 0.02183,  # abs error, std/sqrt(N)

    # Delay kernels
    'kernel_path' : './output/kernels_daily.pkl'
}

# Gangelt of Heinsberg
# https://www.medrxiv.org/content/10.1101/2020.05.04.20090076v2
GAN = {
    'corrected' : True,
    
	'positive'  : 138,
	'tested'    : 919,
	'age'       : 'all',
    'test_date' : ['2020-03-31', '2020-04-06'], # year-month-day"
    'test_type' : 'IgG/IgA',
    'test_tex'  : 'Streeck2020.05.04.20090076',

    # Average default values from Imperial report 34
    'specificity'       : 0.99406,
    'specificity_error' : 0.00140, # abs error, std/sqrt(N)

    'sensitivity'       : 0.89244,
    'sensitivity_error' : 0.02183,  # abs error, std/sqrt(N)

    # Delay kernels
    'kernel_path' : './output/kernels_daily.pkl'
}
