import pandas as pd
import numpy as np
import typing as t
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from Ml_Training.src.data import split_data
from Ml_Training.src.model import get_model_attr, read_configs
from Ml_Training.src.modelio import save_model


from sklearn.datasets import load_iris

from Ml_Training.src.pathconfig import PathConfig
from Ml_Training.src.validation import grid_search, generate_validation, select_best


def train_model(model: t.Optional[SVC], X, y) -> t.Any:
    """
    The function trains the specified model.
    :param model: the model you want to fit
    :param X: the features
    :param y: the target
    :return: the fitted model
    """
    return model.fit(X, np.ravel(y))


PATHFINDER = PathConfig()


def execute_all(
    models: t.List[str],
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    config: t.Union[dict, list],
    filename: t.Optional[str] = None,
    savemodel=False
    ) -> object:
    """
    The function returns a dictionary with the model name, the best parameters and the f1 scores based on the
    model dictionary and the name of the saved model
    prints out the best parameters of each classifier and the best classifier and its hyperparameters in the end
    :param models: the list of desired models
    :param df_x: the features
    :param df_y: the target
    :param config: the config dict of the {model , {hyperparameters}}
    :param filename: the filename on we want to save the trained model
    :param savemodel: Bool - if True the function saves the model, if False it does not save
    :return: the dict of metrics
    """
    defalt_filename = "savemodel.pickle"
    if filename is None:
        filename = defalt_filename
    best_f1 = 0.0
    best_model_name = 'none_'
    metrics = dict()
    X_train, X_test, y_train, y_test = split_data(df_x, df_y)
    for model in models:
        classifier, model_attr = get_model_attr(model, config)
        pipeline = make_pipeline(StandardScaler(), classifier)
        gs = grid_search(pipeline, model_attr)
        mdl = train_model(gs, X_train, y_train)
        met = generate_validation(mdl, X_test, y_test)  # -> dictionary of metrics
        best_model_name, best_f1, change = select_best(best_f1, met["f1"], model, best_model_name)
        if change:
            best_mdl = mdl
        metrics[model] = met

        print('For model', model, "the best parameters: \n", mdl.best_params_)  # include the best parameters
    print(10 * '-', '\nBest Model:', best_model_name, 'with f1 score:', best_f1)
    if savemodel:
        save_model(best_mdl, filename, str(best_model_name)[:3])
        return metrics
    else:
        return metrics


def main():
    #model_list = ['random_forest', 'logistic', 'svc_lin', 'svc_rbf', 'dtree', 'knn']
    model_list = ['random_forest']
    #model_list = ['dtree']
    # Loading data
    irisData = load_iris()
    # Create feature and target arrays
    df_x = pd.DataFrame(irisData.data)
    df_y = pd.DataFrame(irisData.target)
    #print(df_x)
    #print(df_y)
    metrics = execute_all(model_list, df_x, df_y, filename='vine_model',
                          config=read_configs(PATHFINDER.model_config), savemodel=False)


if __name__ == '__main__':
    #print(get_model_attr('dtree', read_configs(PATHFINDER.model_config)))
    #read_configs(PATHFINDER.model_config)
    main()
