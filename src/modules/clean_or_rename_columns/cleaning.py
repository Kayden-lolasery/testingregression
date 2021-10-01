from logs import logDecorator as lD 
import jsonref, pprint
import pandas as pd 
import pathlib
from sklearn.model_selection import train_test_split

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.clean_or_rename_columns.cleaning'

input_config = jsonref.load(open('../config/modules/cleaning.json'))


cleaning = pd.read_csv(pathlib.Path(
        r"/Users/kiathaolim93/testingregression/data/raw_data/..")\
        .parent.resolve()/"dataset_rossi.csv")

@lD.log(logBase + '.readDataSet')
def readDataSet(logger):
    """Summary: read the data
    
    Args:
        logger {logging.Logger}: The logger used for logging error information
    
    Returns:
        Dataframe: cleaned dataframe
    """
    print('We are cleaning!')
    global cleaning

    # unanamed column populates.. lets drop it
    if "Unnamed: 0" in cleaning.columns:
        print("dropping columns")
        cleaning.drop(columns=input_config["drop"],axis=1,inplace= True)
        

    #rename columns to however i like.
    print("renaming columns")
    cleaning.rename(columns=input_config["rename"],inplace=True)

    #re-arrange columns where binary columns are at the end
    #(can probably use if logic for unique values ==2, <2 or >2 for a more generalisable code)
    print("rearranging columns")
    cleaning = cleaning[input_config["rearrange"]]

    #save file to intermediate data folder
    print("saving file")
    cleaning.to_csv(pathlib.Path(
        r"/Users/kiathaolim93/testingregression/data/intermediate/..")\
        .parent.resolve()/"cleaned.csv")
    return

@lD.log(logBase + '.traintestsplit')
def traintestsplit(logger):
    """ splits data into train and test splits
    Args:
        logger {logging.Logger}: The logger used for logging error information
        clean_data (dataframe): cleaned_data
    
    Returns:
        TYPE: traintest split sets
    """
    print("splitting")
    X_train, X_test, y_train, y_test = train_test_split(cleaning.drop(
        columns     =input_config["train_test_split"]["columns"],
        axis        =input_config["train_test_split"]["axis"]),

        cleaning[input_config["train_test_split"]["columns"]], 
        test_size   =input_config["train_test_split"]["test_size"], 
        random_state=input_config["train_test_split"]["random_state"])
    print("saving train and test splits")
    for i,j in zip([X_train, X_test, y_train, y_test],
              input_config["train_test_split"]["variable_names"]):
        i.to_csv(pathlib.Path.joinpath(pathlib.Path(
        r"/Users/kiathaolim93/testingregression/data/final/..")\
        .parent.resolve(),j+".csv"))
    return X_train, X_test, y_train, y_test

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
        A dictionary containing information about the 
        command line arguments. These can be used for
        overwriting command line arguments as needed.
    '''

    print('='*75)
    print('Main function of cleaning: read the data')
    print('='*75)
    print('We get a copy of the result dictionary over here ...')
    pprint.pprint(resultsDict)
    print('='*75)
    readDataSet()
    traintestsplit()

    print('Getting out oof cleaning')
    print('-'*75)
    return

