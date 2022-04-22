import pandas as pd
import numpy as np
import typing as t
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from Ml_Training.src.data import split_data
from Ml_Training.output.save import save_model
from Ml_Training.src.validation import generate_validation, grid_search, select_best


def train_model(model: t.Optional[SVC], X, y) -> t.Optional[SVC]:
    """
    The function trains the specified model.
    model: the model you want to fit
    """
    return model.fit(X, np.ravel(y))


def execute_all(model_repo: t.Dict[str, t.Any],
                df_x: pd.DataFrame, df_y: pd.DataFrame, filename="savedmodel", savemodel=False) -> object:
    """
    The function returns a dictionary with the model name, the best parameters and the f1 scores based on the
    model_repo dictionary and the name of the saved model
        df_x: the dataframe without the class(column) we are to predict
        df_y: the part/class/column of the dataframe we predict
        filename: the filename we want to save the trained model
        savemodel:Bool - if True the function saves the model, if False it does not save
        :rtype: object
    """
    best_f1 = 0.0
    best_model_name = 'none_'
    metrics = dict()
    X_train, X_test, y_train, y_test = split_data(df_x, df_y)
    for name, model in model_repo.items():
        pipeline = make_pipeline(StandardScaler(), model[0])
        gs = grid_search(pipeline, model[1])
        mdl = train_model(gs, X_train, y_train)
        met = generate_validation(mdl, X_test, y_test)  # -> dictionary of metrics
        best_model_name, best_f1, change = select_best(best_f1, met["f1"], name, best_model_name)
        if change:
            best_mdl = mdl
        metrics[name] = met

        print('For model', name, "the best parameters: \n", mdl.best_params_)  # include the best parameters
    print(10 * '-', '\nBest Model:', best_model_name, 'with f1 score:', best_f1)
    if savemodel:
        best_saved_model = save_model(best_mdl, filename, str(best_model_name)[:3])
        return metrics, best_saved_model
    else:
        return metrics
