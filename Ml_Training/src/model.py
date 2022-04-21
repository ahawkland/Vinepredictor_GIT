# Model Repository for all models
import warnings
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

filename = 'vine_model'


class ModelRepository:  # todo explore class possibilities
    """A pre-defined Repository of Ml models: RandomForestClassifier, SVC, LogisticRegression"""
    svc = SVC()
    logistic = LogisticRegression()
    rfc = RandomForestClassifier(criterion='gini', random_state=1)
    MODEL_SPACE = {"random forest": [rfc, [{
        'randomforestclassifier__max_depth': [5, 6, 7, 8, 9, 10],
        'randomforestclassifier__max_features': [2, 3]  # todo set this to incl. features until the max of the given df
    }]], "logistic": [logistic, [{
        'logisticregression__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
    }]], "svc": [svc, [{
        'svc__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
        'svc__kernel': ['linear']
    },
        {
            'svc__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
            'svc__gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
            'svc__kernel': ['rbf']
        }]]}

    def get_model_space(self):
        """
        The default model dictionary is:
        {"random forest": [rfc, [{
        'randomforestclassifier__max_depth': [5, 6, 7, 8, 9, 10],
        'randomforestclassifier__max_features': [2, 3]
    }]], "logistic": [logistic, [{
        'logisticregression__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
    }]], "svc": [svc, [{
        'svc__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
        'svc__kernel': ['linear']
    },
        {
            'svc__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
            'svc__gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
            'svc__kernel': ['rbf']
        }]]}
        :return: the default model dictionary
        """
        return self.MODEL_SPACE
