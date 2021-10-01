from logs import logDecorator as lD 
import pandas as pd
import numpy as np
import jsonref, pprint, os, pickle, pathlib 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.ML.ML'
input_config = jsonref.load(open('../config/modules/ML.json'))

cleaned = pd.read_csv(pathlib.Path(
        r"/Users/kiathaolim93/testingregression/data/intermediate/..")\
        .parent.resolve()/"cleaned.csv").drop(columns=["Unnamed: 0"])
path = []
for file in os.listdir("../data/final/"):
    placeholder = os.path.join(pathlib.Path(
        r"/Users/kiathaolim93/testingregression/data/final/"), file+".csv")
    if file == "X_train":
        path.append(placeholder)
for file in os.listdir("../data/final/"):
    placeholder = os.path.join(pathlib.Path(
        r"/Users/kiathaolim93/testingregression/data/final/"), file+".csv")
    if file == "X_test":
        path.append(placeholder)
for file in os.listdir("../data/final/"):
    placeholder = os.path.join(pathlib.Path(
        r"/Users/kiathaolim93/testingregression/data/final/"), file+".csv")
    if file == "y_train":
        path.append(placeholder)
for file in os.listdir("../data/final/"):
    placeholder = os.path.join(pathlib.Path(
        r"/Users/kiathaolim93/testingregression/data/final/"), file+".csv")
    if file == "y_test":
        path.append(placeholder)

print(path)
dictionary = pd.DataFrame(path)
print(dictionary)
dictionary["variable"] = ["X_train", "X_test", "y_train", "y_test"]
dictionary.rename(columns={0:"Path","variable":"variables"},inplace= True)
dictionary[["variables","Path"]]
dictionary1 = dict(dictionary[["variables","Path"]].values)
X_train = 0
X_test = 0
y_train = 0
y_test = 0

for i,j in dictionary1.items():
    print(i,j)
    globals()[i] = pd.read_csv(j).drop(columns=["Unnamed: 0"])
print("hmm")

@lD.log(logBase + '.LR')
def LR(logger):
    """run linear regression
    Parameters
    ----------
    logger : logging.Logger
        The logger used for logging error information
    X_train : 
        Predictive features in train
    X_test : Predictive features in test
        Predictive features in test, if no test, unable to proceed
    y_train : Predictive features in train
        PredicTED features in train
    y_test : Predictive features in test
        PredicTED features in test, if no test, unable to proceed
    """
    # Create linear regression object
    reg = LinearRegression() 

    # Train the model using the training sets
    reg.fit(X_train, y_train["age"])
    # pickle and store model under ../data/final
    pickle.dump(reg,open(input_config["pickle_filename"],"wb"))

    # Make predictions using the testing set
    global y_pred
    y_pred = reg.predict(X_test)
    #save as csv file
    np.savetxt(input_config["savetext"], y_pred, delimiter=",")

    # The coefficients between predictive & predicted feature
    print('Coefficients: \n', reg.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
    return

@lD.log(logBase + '.main')
def graphs(logger):
    """draw graphs based on ML result
    
    Parameters
    ----------
    y_pred : Array
        Predicted result of age given predictive feature
    """
    # Plot outputs
    print("drawing graphs for results")
    a = 3  # number of rows
    b = 3  # number of columns
    c = 1  # initialize plot counter
    fig = plt.figure(figsize=(14,10))
    for i in cleaned.drop(columns="age",axis=1).columns: #8 columns
      plt.subplot(a, b, c)
      # print(np.shape(X_test[i]),np.shape(y_test))
      # print(type(y_pred),np.shape(y_pred))
      plt.scatter(X_test[i], y_test["age"], alpha=.3, label='ground truth',c="blue")
      plt.plot(X_test[i], y_pred, alpha=.3, label='predictions',   c ="red")
      plt.xlabel(i)
      plt.ylabel('age')
      plt.title("predictions for "+ str(i))
      plt.legend()
      plt.xticks(())
      plt.yticks(())  
      c += 1

    plt.show()
    # script_dir = os.path.dirname(__file__) #get directory
    rel_path = input_config["rel_path"]
    # abs_file_path = os.path.join(script_dir, rel_path)
    fig.savefig(rel_path) #save in results

@lD.log(logBase + '.main')
def main(logger, resultsDict):
    '''main function for ML
    
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
    print('Main function of ML')
    print('='*75)
    print('We get a copy of the result dictionary over here ...')
    pprint.pprint(resultsDict)
    LR()
    print('='*75)
    graphs()

    print('Getting out of ML')
    print('='*75)

    return

