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
from sklearn.utils.multiclass import type_of_target
import logging
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

models_dict = {
    "regression": {
        "linear regression": LinearRegression,
        "decision tree": DecisionTreeRegressor,
        "random forest": RandomForestRegressor,
        "extra trees": ExtraTreesRegressor,
        "svm": SVR,
        "nearest neighbor": KNeighborsRegressor,
        "neural network": MLPRegressor
    },
    "classification": {
        "logistic regression": LogisticRegression,
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


def evaluate_model(model_type, y_pred, y_true, **kwargs):
    if model_type not in metrics_dict.keys():
        raise Exception("model type needs to be regression or classification")
    metrics = metrics_dict.get(model_type, None)
    eval_res = {}
    logger.info(f"shape of y_pred: {y_pred.shape} | shape of y_true: {y_pred.shape}")
    if metrics:
        for metric in metrics:
            logger.info(f"Calculating {metric.__name__} .....")

            # if type_of_target(y_true) == 'multiclass' and metric.__name__ in ('precision_score',
            #                                                                   'recall_score',
            #                                                                   'f1_score'):
            #
            #     eval_res[metric.__name__] = metric(y_pred=y_pred, y_true=y_true, average='micro')
            # else:
            eval_res[metric.__name__] = metric(y_pred=y_pred, y_true=y_true, **kwargs)

    return eval_res


if __name__ == '__main__':

    print(LinearRegression().fit.__code__.co_varnames)
