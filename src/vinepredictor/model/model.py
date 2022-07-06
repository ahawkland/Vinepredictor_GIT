# Model Repository for all models
import warnings
from typing import Any, Optional

import yaml
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from Ml_Training.src.pathconfig import PathConfig

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def read_configs(path: Path) -> dict:
    """
    reads a yaml file on the imputed Path
    :param path: the path where the yaml file is
    :return: a dictionary from the yaml file
    """
    with open(path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def get_model_attr(model: str, config: dict) -> tuple[Any, Optional[Any]]:
    """
    The function returns the configuration (a dict of hyperparameters) of a model
    :param model: the targeted model
    :param config: the config file where the hyperparameters are listed by the models
    :return: a tuple (model classifier function,  the dictionary of hyperparameters)
    """
    model = model.lower()
    model_config_repo = config.keys()
    if model not in model_config_repo:
        model_str = ", ".join(model_config_repo)
        raise ValueError(f"{model} is not a valid model type. only available ones are {model_str}")
    model_repo = {
        "random_forest": RandomForestClassifier(criterion='gini', random_state=1),
        "logistic": LogisticRegression(),
        "svc_lin": SVC(),
        "svc_rbf": SVC(),
        "dtree": tree.DecisionTreeClassifier(),
        "knn": KNeighborsClassifier(),
        "xgb": XGBClassifier()
    }

    return model_repo.get(model), config.get(model)


def main():
    PATHFINDER = PathConfig()
    #print(read_configs(PATHFINDER.model_config))
    print(get_model_attr('logistic', read_configs(PATHFINDER.model_config)))


if __name__ == '__main__':
    main()
