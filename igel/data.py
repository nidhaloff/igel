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
        "linear regression": {"class": LinearRegression,
                              "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"
                              },

        "decision tree": {"class": DecisionTreeRegressor,
                          "link": "https://scikit-learn.org/stable/modules/generatedsklearn.tree.DecisionTreeRegressor.html?highlight=decision%20tree%20regressor#sklearn.tree.DecisionTreeRegressor"
                          },
        "random forest": {"class": RandomForestRegressor,
                          "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=random%20forest#sklearn.ensemble.RandomForestRegressor"},

        "extra trees": {"class": ExtraTreesRegressor,
                        "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html?highlight=extra%20trees#sklearn.ensemble.ExtraTreesRegressor"},
        "svm": {"class": SVR,
                "link": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html?highlight=svr#sklearn.svm.SVR"},
        "nearest neighbor": {"class": KNeighborsRegressor,
                             "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html?highlight=neighbor#sklearn.neighbors.KNeighborsRegressor"},
        "neural network": {"class": MLPRegressor,
                           "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html?highlight=mlp#sklearn.neural_network.MLPRegressor"}
    },
    "classification": {
        "logistic regression": {"class": LogisticRegression,
                                "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=regression#sklearn.linear_model.LogisticRegression"},
        "decision tree": {"class": DecisionTreeClassifier,
                          "link": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decision%20tree#sklearn.tree.DecisionTreeClassifier"},
        "random forest": {"class": RandomForestClassifier,
                          "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest#sklearn.ensemble.RandomForestClassifier"},
        "extra trees": {"class": ExtraTreesClassifier,
                        "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html?highlight=extra%20trees#sklearn.ensemble.ExtraTreesClassifier"},
        "svm": {"class": SVC,
                "link": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=svc#sklearn.svm.SVC"},
        "nearest neighbor": {"class": KNeighborsClassifier,
                             "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?highlight=neighbor#sklearn.neighbors.KNeighborsClassifier"},
        "neural network": {"class": MLPClassifier,
                           "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html?highlight=mlp#sklearn.neural_network.MLPClassifier"}
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


def evaluate_model(model, model_type, x_test, y_pred, y_true, **kwargs):

    if y_pred.ndim > 1:
        if y_true.shape[1] > 1 or y_pred.shape[1] > 1:
            logger.info(f"Multitarget {model_type} Evaluation: calculating {model_type} score")
            return {f"{model_type} score": model.score(x_test, y_true)}

    if model_type not in metrics_dict.keys():
        raise Exception("model type needs to be regression or classification")
    metrics = metrics_dict.get(model_type, None)
    eval_res = {}
    if metrics:
        for metric in metrics:
            logger.info(f"Calculating {metric.__name__} .....")
            logger.info(f"type of target: {type_of_target(y_true)}")
            if type_of_target(y_true) in ('multiclass-multioutput',
                                          'multilabel-indicator',
                                          'multiclass') and metric.__name__ in ('precision_score',
                                                                                'accuracy_score',
                                                                                'recall_score',
                                                                                'f1_score'):
                if metric.__name__ == 'accuracy_score':
                    eval_res[metric.__name__] = metric(y_pred=y_pred,
                                                       y_true=y_true)
                else:
                    eval_res[metric.__name__] = metric(y_pred=y_pred,
                                                       y_true=y_true,
                                                       average='micro')

            else:
                eval_res[metric.__name__] = metric(y_pred=y_pred, y_true=y_true, **kwargs)

    return eval_res


if __name__ == '__main__':
    print(LinearRegression().fit.__code__.co_varnames)
