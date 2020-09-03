from sklearn.linear_model import (LinearRegression, LogisticRegression)
from sklearn.ensemble import (RandomForestClassifier,
                              RandomForestRegressor,
                              ExtraTreesRegressor,
                              ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             mean_squared_log_error,
                             median_absolute_error,
                             accuracy_score,
                             f1_score,
                             r2_score,
                             precision_score,
                             recall_score)

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

metrics_dict = {
    "regression": (
                    mean_squared_error, mean_absolute_error, mean_squared_log_error, median_absolute_error, r2_score
     ),
    "classification": (
                    accuracy_score, f1_score, precision_score, recall_score
    )
}


def evaluate_model(model_type, y_pred, y_true):
    if model_type not in metrics_dict.keys():
        raise Exception("model type needs to be regression or classification")
    metrics = metrics_dict.get(model_type, None)
    eval_res = {}
    if metrics:
        for metric in metrics:
            eval_res[metric.__name__] = metric(y_pred=y_pred, y_true=y_true)

    return eval_res
