from sklearn.linear_model import (LinearRegression,
                                  LogisticRegression,
                                  Ridge,
                                  RANSACRegressor,
                                  RidgeClassifier,
                                  RidgeClassifierCV,
                                  RidgeCV,
                                  BayesianRidge,
                                  SGDRegressor,
                                  GammaRegressor,
                                  LogisticRegressionCV,
                                  TheilSenRegressor,
                                  PoissonRegressor,
                                  TweedieRegressor,
                                  ARDRegression,
                                  SGDClassifier,
                                  HuberRegressor,
                                  Lasso,
                                  LassoCV,
                                  LassoLars,
                                  LassoLarsCV,
                                  PassiveAggressiveClassifier,
                                  ElasticNet,
                                  ElasticNetCV,
                                  Perceptron)

from sklearn.ensemble import (RandomForestClassifier,
                              RandomForestRegressor,
                              ExtraTreesRegressor,
                              ExtraTreesClassifier)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier, BernoulliRBM
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

        "linear regression": {
            "class": LinearRegression,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html",
            "sgd_class": SGDRegressor
        },

        "lasso regression": {
            "class": Lasso,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html?\
                    highlight=lasso#sklearn.linear_model.Lasso",
            "cv_class": LassoCV
        },

        "lassolars regression": {
            "class": LassoLars,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html?\
                    highlight=lasso#sklearn.linear_model.LassoLars",
            "cv_class": LassoLarsCV
        },

        "bayesian regression": {
            "class": BayesianRidge,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html?\
                    highlight=ridge#sklearn.linear_model.BayesianRidge"
        },

        "huber regression": {
            "class": HuberRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html?\
                    highlight=huber#sklearn.linear_model.HuberRegressor"
        },

        "ridge": {
            "class": Ridge,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html\
                    #sklearn.linear_model.Ridge",
            "cv_class": RidgeCV
        },

        "poisson regression": {
            "class": PoissonRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html?\
                    highlight=poisson#sklearn.linear_model.PoissonRegressor"
        },

        "ARD regression": {
            "class": ARDRegression,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html?\
                    highlight=ard#sklearn.linear_model.ARDRegression"
        },

        "tweedie regression": {
            "class": TweedieRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html?\
                    highlight=tweedie#sklearn.linear_model.TweedieRegressor"
        },

        "TheilSen regression": {
            "class": TheilSenRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html?\
                    highlight=theilsenregressor#sklearn.linear_model.TheilSenRegressor"
        },

        "gamma regression": {
            "class": GammaRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html?\
                    highlight=gamma%20regressor#sklearn.linear_model.GammaRegressor"
        },
        "RANSAC regression": {
            "class": RANSACRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html"
        },

        "decision tree": {
            "class": DecisionTreeRegressor,
            "link": "https://scikit-learn.org/stable/modules/generatedsklearn.tree.DecisionTreeRegressor.html?\
                    highlight=decision%20tree%20regressor#sklearn.tree.DecisionTreeRegressor"
        },

        "random forest": {
            "class": RandomForestRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?\
                    highlight=random%20forest#sklearn.ensemble.RandomForestRegressor"},

        "extra trees": {
            "class": ExtraTreesRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html?\
                    highlight=extra%20trees#sklearn.ensemble.ExtraTreesRegressor"},

        "svm": {
            "class": SVR,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html?\
                    highlight=svr#sklearn.svm.SVR"},

        "nearest neighbor": {
            "class": KNeighborsRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html?\
                    highlight=neighbor#sklearn.neighbors.KNeighborsRegressor"},

        "neural network": {
            "class": MLPRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html?\
                    highlight=mlp#sklearn.neural_network.MLPRegressor"
        },

        "elasticnet": {
            "class": ElasticNet,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html?\
                    highlight=elasticnet#sklearn.linear_model.ElasticNet",
            "cv_class": ElasticNetCV
        },

        "BernoulliRBM": {
            "class": BernoulliRBM,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html#\
                    sklearn.neural_network.BernoulliRBM"
        }
    },

    "classification": {

        "logistic regression": {

            "class": LogisticRegression,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?\
                    highlight=regression#sklearn.linear_model.LogisticRegression",
            "cv_class": LogisticRegressionCV,
            "sgd_class": SGDClassifier
        },

        "ridge": {
            "class": RidgeClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html?\
                    highlight=ridgeclassifier#sklearn.linear_model.RidgeClassifier",
            "cv_class": RidgeClassifierCV
        },

        "decision tree": {
            "class": DecisionTreeClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?\
                    highlight=decision%20tree#sklearn.tree.DecisionTreeClassifier"},

        "random forest": {
            "class": RandomForestClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?\
                    highlight=random%20forest#sklearn.ensemble.RandomForestClassifier"},

        "extra trees": {
            "class": ExtraTreesClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html?\
                    highlight=extra%20trees#sklearn.ensemble.ExtraTreesClassifier"},

        "svm": {
            "class": SVC,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?\
                    highlight=svc#sklearn.svm.SVC"},

        "nearest neighbor": {
            "class": KNeighborsClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?\
                    highlight=neighbor#sklearn.neighbors.KNeighborsClassifier"},

        "neural network": {
            "class": MLPClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html?\
                    highlight=mlp#sklearn.neural_network.MLPClassifier"
        },

        "passive agressive classifier": {
            "class": PassiveAggressiveClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html?\
                    highlight=passiveaggressiveclassifier#sklearn.linear_model.PassiveAggressiveClassifier"
        },

        "perceptron": {
            "class": Perceptron,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#\
                    sklearn.linear_model.Perceptron"
        },

        "BernoulliRBM": {
                    "class": BernoulliRBM,
                    "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html#\
                            sklearn.neural_network.BernoulliRBM"
                }
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
