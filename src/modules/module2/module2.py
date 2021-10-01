from logs import logDecorator as lD 
import jsonref, pprint
from lib.databaseIO import pgIO
import os
import pandas as pd

#< 1000 patient csv or pickle, >1000 patient save in DB
config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.module2.module2'

schema = "r21r1_dcdm"
table  = "person"
columns = ["person_id","marital_status"]


@lD.log(logBase + '.test1')
def test1(logger):
    query = """
    SELECT person_id, marital_status
        FROM r21r1_dcdm.person """
    df = pd.DataFrame(pgIO.getAllData(query), 
        columns =["person_id", "marital_status"])
    return df


@lD.log(logBase + '.doSomething')
def doSomething(logger):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    '''

    print('We are in module 1')

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

    print('='*30)
    print('Main function of module 2')
    print('='*30)
    print('We get a copy of the result dictionary over here ...')
    # pprint.pprint(resultsDict)

    df = test1()
    print(df)

    print('Getting out of Module 2')
    print('-'*30)

    return

