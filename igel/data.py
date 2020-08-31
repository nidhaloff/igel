from sklearn.linear_model import (LinearRegression, LogisticRegression)
from sklearn.ensemble import (RandomForestClassifier,
                              RandomForestRegressor,
                              ExtraTreesRegressor,
                              ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier


models_dict = {
    "regression": {
        "linear": LinearRegression,
        "decision tree": DecisionTreeRegressor,
        "random forest": RandomForestRegressor,
        "extra trees": ExtraTreesRegressor,
        "svm": SVR,
        "nearest neighbor": KNeighborsRegressor,
        "neural network": MLPRegressor
    },
    "classification": {
        "logistic": LogisticRegression,
        "decision tree": DecisionTreeClassifier,
        "random forest": RandomForestClassifier,
        "extra trees": ExtraTreesClassifier,
        "svm": SVC,
        "nearest neighbor": KNeighborsClassifier,
        "neural network": MLPClassifier
    }

}
