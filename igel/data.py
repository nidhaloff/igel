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
                              ExtraTreesClassifier,
                              AdaBoostClassifier,
                              AdaBoostRegressor,
                              BaggingClassifier,
                              BaggingRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              StackingClassifier,
                              StackingRegressor,
                              VotingClassifier,
                              VotingRegressor)

from sklearn.naive_bayes import (BernoulliNB,
                                 CategoricalNB,
                                 ComplementNB,
                                 GaussianNB,
                                 MultinomialNB)

from sklearn.cluster import (KMeans,
                             AffinityPropagation,
                             AgglomerativeClustering,
                             Birch,
                             DBSCAN,
                             FeatureAgglomeration,
                             MiniBatchKMeans,
                             MeanShift,
                             OPTICS,
                             SpectralBiclustering,
                             SpectralClustering,
                             SpectralCoclustering)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR
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

        "LinearRegression": {
            "class": LinearRegression,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html",

        },

        "SGDRegressor": {
            "class": SGDRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html?"
                    "highlight=sgd#sklearn.linear_model.SGDRegressor",

        },

        "Lasso": {
            "class": Lasso,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html?"
                    "highlight=lasso#sklearn.linear_model.Lasso",
            "cv_class": LassoCV
        },

        "LassoLars": {
            "class": LassoLars,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html?"
                    "highlight=lasso#sklearn.linear_model.LassoLars",
            "cv_class": LassoLarsCV
        },

        "BayesianRegression": {
            "class": BayesianRidge,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html?"
                    "highlight=ridge#sklearn.linear_model.BayesianRidge"
        },

        "HuberRegression": {
            "class": HuberRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html?"
                    "highlight=huber#sklearn.linear_model.HuberRegressor"
        },

        "Ridge": {
            "class": Ridge,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html"
                    "#sklearn.linear_model.Ridge",
            "cv_class": RidgeCV
        },

        "PoissonRegression": {
            "class": PoissonRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html?"
                    "highlight=poisson#sklearn.linear_model.PoissonRegressor"
        },

        "ARDRegression": {
            "class": ARDRegression,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html?"
                    "highlight=ard#sklearn.linear_model.ARDRegression"
        },

        "TweedieRegression": {
            "class": TweedieRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html?"
                    "highlight=tweedie#sklearn.linear_model.TweedieRegressor"
        },

        "TheilSenRegression": {
            "class": TheilSenRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html?"
                    "highlight=theilsenregressor#sklearn.linear_model.TheilSenRegressor"
        },

        "GammaRegression": {
            "class": GammaRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html?"
                    "highlight=gamma%20regressor#sklearn.linear_model.GammaRegressor"
        },

        "RANSACRegression": {
            "class": RANSACRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html"
        },

        "DecisionTree": {
            "class": DecisionTreeRegressor,
            "link": "https://scikit-learn.org/stable/modules/generatedsklearn.tree.DecisionTreeRegressor.html?"
                    "highlight=decision%20tree%20regressor#sklearn.tree.DecisionTreeRegressor"
        },

        "ExtraTree": {
            "class": ExtraTreeRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html"
                    "#sklearn.tree.ExtraTreeRegressor"
        },

        "RandomForest": {
            "class": RandomForestRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?"
                    "highlight=random%20forest#sklearn.ensemble.RandomForestRegressor"},

        "ExtraTrees": {
            "class": ExtraTreesRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html?"
                    "highlight=extra%20trees#sklearn.ensemble.ExtraTreesRegressor"},

        "SVM": {
            "class": SVR,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html?"
                    "highlight=svr#sklearn.svm.SVR"},

        "LinearSVM": {
            "class": LinearSVR,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR"
        },

        "NuSVM": {
            "class": NuSVR,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html#sklearn.svm.NuSVR"
        },

        "NearestNeighbor": {
            "class": KNeighborsRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html?"
                    "highlight=neighbor#sklearn.neighbors.KNeighborsRegressor"},

        "NeuralNetwork": {
            "class": MLPRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html?"
                    "highlight=mlp#sklearn.neural_network.MLPRegressor"
        },

        "ElasticNet": {
            "class": ElasticNet,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html?"
                    "highlight=elasticnet#sklearn.linear_model.ElasticNet",
            "cv_class": ElasticNetCV
        },

        "BernoulliRBM": {
            "class": BernoulliRBM,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html#"
                    "sklearn.neural_network.BernoulliRBM"
        },

        "BoltzmannMachine": {
            "class": BernoulliRBM,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html#"
                    "sklearn.neural_network.BernoulliRBM"
        },

        "Adaboost": {
            "class": AdaBoostRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#"
                    "sklearn.ensemble.AdaBoostRegressor"
        },

        "Bagging": {
            "class": BaggingRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html"
                    "#sklearn.ensemble.BaggingRegressor"
        },

        "GradientBoosting": {
            "class": GradientBoostingRegressor,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html"
                    "#sklearn.ensemble.GradientBoostingRegressor"
        }
    },

    "classification": {

        "LogisticRegression": {

            "class": LogisticRegression,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?"
                    "highlight=regression#sklearn.linear_model.LogisticRegression",
            "cv_class": LogisticRegressionCV
        },

        "SGDClassifier": {

            "class": SGDClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html?"
                    "highlight=sgd#sklearn.linear_model.SGDClassifier",
        },

        "Ridge": {
            "class": RidgeClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html?"
                    "highlight=ridgeclassifier#sklearn.linear_model.RidgeClassifier",
            "cv_class": RidgeClassifierCV
        },

        "DecisionTree": {
            "class": DecisionTreeClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?"
                    "highlight=decision%20tree#sklearn.tree.DecisionTreeClassifier"},

        "ExtraTree": {
            "class": ExtraTreeClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html"
                    "#sklearn.tree.ExtraTreeClassifier"
        },

        "RandomForest": {
            "class": RandomForestClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?"
                    "highlight=random%20forest#sklearn.ensemble.RandomForestClassifier"},

        "ExtraTrees": {
            "class": ExtraTreesClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html?"
                    "highlight=extra%20trees#sklearn.ensemble.ExtraTreesClassifier"},

        "SVM": {
            "class": SVC,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?"
                    "highlight=svc#sklearn.svm.SVC"},

        "LinearSVM": {
            "class": LinearSVC,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC"
        },

        "NuSVM": {
            "class": NuSVC,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC"
        },


        "NearestNeighbor": {
            "class": KNeighborsClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?"
                    "highlight=neighbor#sklearn.neighbors.KNeighborsClassifier"},

        "NeuralNetwork": {
            "class": MLPClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html?"
                    "highlight=mlp#sklearn.neural_network.MLPClassifier"
        },

        "PassiveAgressiveClassifier": {
            "class": PassiveAggressiveClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html?"
                    "highlight=passiveaggressiveclassifier#sklearn.linear_model.PassiveAggressiveClassifier"
        },

        "Perceptron": {
            "class": Perceptron,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#"
                    "sklearn.linear_model.Perceptron"
        },

        "BernoulliRBM": {
                    "class": BernoulliRBM,
                    "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html#"
                            "sklearn.neural_network.BernoulliRBM"
                },

        "BoltzmannMachine": {
            "class": BernoulliRBM,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html#"
                    "sklearn.neural_network.BernoulliRBM"
        },

        "CalibratedClassifier": {
            "class": CalibratedClassifierCV,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#"
                    "sklearn.calibration.CalibratedClassifierCV"
        },

        "Adaboost": {
            "class": AdaBoostClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#"
                    "sklearn.ensemble.AdaBoostClassifier"
        },

        "Bagging": {
            "class": BaggingClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html"
                    "#sklearn.ensemble.BaggingClassifier"
        },

        "GradientBoosting": {
            "class": GradientBoostingClassifier,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html"
                    "#sklearn.ensemble.GradientBoostingClassifier"
        },

        "BernoulliNaiveBayes": {
            "class": BernoulliNB,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html"
                    "#sklearn.naive_bayes.BernoulliNB"
        },

        "CategoricalNaiveBayes": {
            "class": CategoricalNB,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html"
                    "#sklearn.naive_bayes.CategoricalNB"
        },

        "ComplementNaiveBayes": {
            "class": ComplementNB,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html"
                    "#sklearn.naive_bayes.ComplementNB"
        },

        "GaussianNaiveBayes": {
            "class": GaussianNB,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html"
                    "#sklearn.naive_bayes.GaussianNB"
        },

        "MultinomialNaiveBayes": {
            "class": MultinomialNB,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html"
                    "#sklearn.naive_bayes.MultinomialNB"
        }


    },

    "clustering": {
        "KMeans": {
            "class": KMeans,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html"
                    "#sklearn.cluster.KMeans"
        },

        "AffinityPropagation": {
            "class": AffinityPropagation,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html"
                    "#sklearn.cluster.AffinityPropagation"
        },

        "Birch": {
            "class": Birch,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch"
        },

        "AgglomerativeClustering": {
            "class": AgglomerativeClustering,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#"
                    "sklearn.cluster.AgglomerativeClustering"
        },

        "FeatureAgglomeration": {
            "class": FeatureAgglomeration,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html#"
                    "sklearn.cluster.FeatureAgglomeration"
        },
        "DBSCAN": {
            "class": DBSCAN,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html"
                    "#sklearn.cluster.DBSCAN"
        },
        "MiniBatchKMeans": {
            "class": MiniBatchKMeans,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html"
                    "#sklearn.cluster.MiniBatchKMeans"
        },

        "SpectralBiclustering": {
            "class": SpectralBiclustering,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralBiclustering.html"
                    "#sklearn.cluster.SpectralBiclustering"
        },

        "SpectralCoclustering": {
            "class": SpectralCoclustering,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralCoclustering.html"
                    "#sklearn.cluster.SpectralCoclustering"
        },

        "SpectralClustering": {
            "class": SpectralClustering,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html"
                    "#sklearn.cluster.SpectralClustering"
        },

        "MeanShift": {
            "class": MeanShift,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#"
                    "sklearn.cluster.MeanShift"
        },
        "OPTICS": {
            "class": OPTICS,
            "link": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html"
                    "#sklearn.cluster.OPTICS"
        },

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


def evaluate_model(model, model_type, x_test, y_pred, y_true, get_score_only, **kwargs):
    if get_score_only:
        logger.info(f"calculating {model_type} score...")
        return {f"{model_type} score": model.score(x_test, y_true)}

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

