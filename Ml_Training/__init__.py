import pandas as pd
from Ml_Training.src import execute_all




def __init__:
    """
    The main function is : Ml_Training.src.train.execute_all(model_repo, df_x, df_y, 'vine_model', savemodel=True)
    The function returns a dictionary with the model name, the best parameters and the f1 scores based on the
    model_repo dictionary and the name of the saved model
        df_x: the dataframe without the class(column) we are to predict
        df_y: the part/class/column of the dataframe we predict
        filename: the filename we want to save the trained model
    savemodel:Bool - if True the function saves the model, if False it does not save

    """
    execute_all(model_repo, df_x, df_y, filename, savemodel=True)
