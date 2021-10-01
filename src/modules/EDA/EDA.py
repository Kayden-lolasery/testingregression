from logs import logDecorator as lD 
import jsonref, pprint
import numpy as np
import pandas as pd
import sys, os, pathlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas_profiling  as pdp

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.EDA.EDA'
configEDA = jsonref.load(open('../config/modules/EDA.json'))

cleaned = pd.read_csv(pathlib.Path(
        r"/Users/kiathaolim93/testingregression/data/intermediate/..")\
        .parent.resolve()/"cleaned.csv")
@lD.log(logBase + '.descriptive')
def descriptive(logger):
    '''
    This function stores the descriptive statistics (.describe())
    about the dataset as a .txt file
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    '''
    print('We are in EDA')
    global cleaned
    print('Fetching statistical description of the data')
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    # script_dir = os.path.dirname(__file__)
    # rel_path = configEDA["rel_path"]
    # abs_file_path = os.path.join(script_dir, rel_path)
    f = open(pathlib.Path(
        r"/Users/kiathaolim93/testingregression/results/..")\
        .parent.resolve()/"descriptive.txt", "w")
    print(round(cleaned.describe(),2))
    f.close()
    print("saved")
    return

@lD.log(logBase + '.html')
def html(logger):
    """Get a HTML summary of the data
    
    Parameters
    ----------
    logger : logging.Logger
        The logger used for logging error information
    """
    global cleaned
    print("initiating pandas_profiling")
    profile = pdp.ProfileReport(cleaned)
    print("storing")
    profile.to_file(output_file = pathlib.Path(
        r"/Users/kiathaolim93/testingregression/results/html_files/..")\
        .parent.resolve()/"pp_data_summary.html",)
    return


@lD.log(logBase + '.main')
def main(logger, resultsDict):
    '''main function for module1
    
    This function finishes all the tasks for the
    main function. This is a way in which a 
    particular module is going to be executed. 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    resultsDict: {dict}
        A dintionary containing information about the 
        command line arguments. These can be used for
        overwriting command line arguments as needed.
    '''

    print('='*75)
    print('Main function of EDA')
    print('='*75)
    print('We get a copy of the result dictionary over here ...')
    pprint.pprint(resultsDict)

    descriptive()
    html()

    print('Getting out of EDA')
    print('-'*75)

    return

