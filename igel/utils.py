from sklearn.linear_model import (LinearRegression, LogisticRegression)
from sklearn.ensemble import (RandomForestClassifier,
                              RandomForestRegressor,
                              ExtraTreesRegressor,
                              ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
import yaml

models_map = {
    "regression": {
        "linear": LinearRegression,
        "tree": DecisionTreeRegressor,
        "forest": RandomForestRegressor,
        "extra trees": ExtraTreesRegressor,
        "svm": SVR,
        "neighbor": KNeighborsRegressor,
        "nn": MLPRegressor
    },
    "classification": {
        "linear": LogisticRegression,
        "tree": DecisionTreeClassifier,
        "forest": RandomForestClassifier,
        "extra trees": ExtraTreesClassifier,
        "svm": SVC,
        "neighbor": KNeighborsClassifier,
        "nn": MLPClassifier
    }

}


def read_yaml(f):
    with open(f, 'r') as stream:
        try:
            res = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        else:
            return res
