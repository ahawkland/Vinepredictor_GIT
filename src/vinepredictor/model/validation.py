import typing as t

from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


def generate_validation(model, x, y) -> t.Dict[str, t.Any]:
    """
    The function returns the f1 score for the model.
    :param model: the model you want to score
    :param x: features
    :param y: target
    :return: a dict {model name : f1 score}
    """
    y_hat = model.predict(x)
    f1 = f1_score(y_hat, y, average='weighted')
    return {"f1": f1}


def select_best(best_val: float, metric: float, model_name: str, best_model: str) -> tuple:
    """
    The function returns the best model with the best metrics so far and marks if change was done (change = True)
    from the baseline
    :param best_val: the best value so far
    :param metric: the current metric of this epoch
    :param model_name: the name of the model
    :param best_model: the best model so far
    :return: a tuple (the best model after this epoch, the best value after this epoch, bool: if there was a change in
    the best values)
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
    :param pipeline: a defined pipeline for grid search
    :param parameter_grid: the dict{model name : parameters} of parameters we want to test
    :return: the grid search
    """
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring='accuracy',
        cv=10,
        refit=True,
        n_jobs=1
    )
    return gs


def score_model(loaded_model, X, y):
    """
    The function scores the loaded model and prints out the score
    :param loaded_model: the pickle file which is opened
    :param X: features
    :param y: target
    :return: None
    """
    print(loaded_model.score(X, y))


def main():
    pass


if __name__ == '__main__':
    main()