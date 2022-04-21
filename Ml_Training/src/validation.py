import pickle
import typing as t
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


def generate_validation(model, x, y) -> t.Dict[str, t.Any]:
    """
    The function returns the f1 score for the model
    model: the model you want to score
    """
    y_hat = model.predict(x)
    f1 = f1_score(y_hat, y, average='weighted')
    return {"f1": f1}

def select_best(best_val: float, metric: float, model_name: str, best_model: str) -> any:
    """
    The function returns the best model with the best metrics so far and marks if change was done (change = True) from the baseline
    """
    if float(metric) > float(best_val):
        best_val = metric
        best_model = model_name
        change = True
    else:
        change = False
    return best_model, best_val, change

def grid_search(pipeline: t.Any, parameter_grid: t.Dict[str, t.List]) -> t.Any:
    """
    The function returns a grid search object with the given pipeline.
    """
    gs = GridSearchCV(estimator=pipeline,
                     param_grid = parameter_grid,
                     scoring='accuracy',
                     cv=10,
                     refit=True,
                     n_jobs=1)
    return gs


from typing import NoReturn

def load_n_score_model(X_re_test, y_re_test, filename: str) -> NoReturn:
    """
    The function loads the specified saved model (filename) and prints the score it achieves
    """
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_re_test, y_re_test)
    print(result)