"""Main module."""

import pandas as pd
import pickle
import os
import json
import warnings
import logging

try:
    from igel.utils import read_yaml, create_yaml, extract_params, _reshape, read_json
    from igel.data import evaluate_model
    from igel.configs import configs
    from igel.data import models_dict, metrics_dict
    from igel.preprocessing import update_dataset_props
    from igel.preprocessing import handle_missing_values, encode, normalize
    from igel.hyperparams import hyperparameter_search
except ImportError:
    from utils import read_yaml, create_yaml, extract_params, _reshape, read_json
    from data import evaluate_model
    from configs import configs
    from data import models_dict, metrics_dict
    from preprocessing import update_dataset_props
    from preprocessing import handle_missing_values, encode, normalize
    from hyperparams import hyperparameter_search

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

warnings.filterwarnings("ignore")
logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Igel(object):
    """
    Igel is the base model to use the fit, evaluate and predict functions of the sklearn library
    """

    available_commands = ('fit', 'evaluate', 'predict', 'experiment')
    supported_types = ('regression', 'classification', 'clustering')
    results_path = configs.get('results_path')  # path to the results folder
    default_model_path = configs.get('default_model_path')  # path to the pre-fitted model
    description_file = configs.get('description_file')  # path to the description.json file
    evaluation_file = configs.get('evaluation_file')  # path to the evaluation.json file
    prediction_file = configs.get('prediction_file')  # path to the predictions.csv
    default_dataset_props = configs.get('dataset_props')  # dataset props that can be changed from the yaml file
    default_model_props = configs.get('model_props')  # model props that can be changed from the yaml file
    model = None

    def __init__(self, **cli_args):
        logger.info(f"Entered CLI args: {cli_args}")
        logger.info(f"Executing command: {cli_args.get('cmd')} ...")
        self.data_path: str = cli_args.get('data_path')  # path to the dataset
        logger.info(f"reading data from {self.data_path}")
        self.command = cli_args.get('cmd', None)
        if not self.command or self.command not in self.available_commands:
            raise Exception(f"You must enter a valid command.\n"
                            f"available commands: {self.available_commands}")

        if self.command == "fit":
            self.yml_path = cli_args.get('yaml_path')
            file_ext = self.yml_path.split('.')[-1]
            logger.info(f"You passed the configurations as a {file_ext} file.")

            self.yaml_configs = read_yaml(self.yml_path) if file_ext == 'yaml' else read_json(self.yml_path)
            logger.info(f"your chosen configuration: {self.yaml_configs}")

            # dataset options given by the user
            self.dataset_props: dict = self.yaml_configs.get('dataset', self.default_dataset_props)
            # model options given by the user
            self.model_props: dict = self.yaml_configs.get('model', self.default_model_props)
            # list of target(s) to predict
            self.target: list = self.yaml_configs.get('target')

            self.model_type: str = self.model_props.get('type')
            logger.info(f"dataset_props: {self.dataset_props} \n"
                        f"model_props: {self.model_props} \n "
                        f"target: {self.target} \n")

        # if entered command is evaluate or predict, then the pre-fitted model needs to be loaded and used
        else:
            self.model_path = cli_args.get('model_path', self.default_model_path)
            logger.info(f"path of the pre-fitted model => {self.model_path}")
            # load description file to read stored training parameters
            with open(self.description_file, 'r') as f:
                dic = json.load(f)
                self.target: list = dic.get("target")  # target to predict as a list
                self.model_type: str = dic.get("type")  # type of the model -> regression or classification
                self.dataset_props: dict = dic.get('dataset_props')  # dataset props entered while fitting
        getattr(self, self.command)()

    def _create_model(self, **kwargs):
        """
        fetch a model depending on the provided type and algorithm by the user and return it
        @return: class of the chosen model
        """
        model_type: str = self.model_props.get('type')
        model_algorithm: str = self.model_props.get('algorithm')
        use_cv = self.model_props.get('use_cv_estimator', None)

        model_args = None
        if not model_type or not model_algorithm:
            raise Exception(f"model_type and algorithm cannot be None")
        algorithms: dict = models_dict.get(model_type)  # extract all algorithms as a dictionary
        model = algorithms.get(model_algorithm)  # extract model class depending on the algorithm
        logger.info(f"Solving a {model_type} problem using ===> {model_algorithm}")
        if not model:
            raise Exception("Model not found in the algorithms list")
        else:
            model_props_args = self.model_props.get('arguments', None)
            if model_props_args and type(model_props_args) == dict:
                model_args = model_props_args
            elif not model_props_args or model_props_args.lower() == "default":
                model_args = None

            if use_cv:
                model_class = model.get('cv_class', None)
                if model_class:
                    logger.info(
                                f"cross validation estimator detected. "
                                f"Switch to the CV version of the {model_algorithm} algorithm")
                else:
                    logger.info(
                        f"No CV class found for the {model_algorithm} algorithm"
                    )
            else:
                model_class = model.get('class')
            logger.info(f"model arguments: \n"
                        f"{self.model_props.get('arguments')}")
            model = model_class(**kwargs) if not model_args else model_class(**model_args)
            return model, model_args

    def _save_model(self, model):
        """
        save the model to a binary file
        @param model: model to save
        @return: bool
        """
        try:
            if not os.path.exists(self.results_path):
                logger.info(f"creating model_results folder to save results...\n"
                            f"path of the results folder: {self.results_path}")
                os.mkdir(self.results_path)
            else:
                logger.info(f"Folder {self.results_path} already exists")
                logger.warning(f"data in the {self.results_path} folder will be overridden. If you don't "
                               f"want this, then move the current {self.results_path} to another path")

        except OSError:
            logger.exception(f"Creating the directory {self.results_path} failed ")
        else:
            logger.info(f"Successfully created the directory in {self.results_path} ")
            pickle.dump(model, open(self.default_model_path, 'wb'))
            return True

    def _load_model(self, f: str = ''):
        """
        load a saved model from file
        @param f: path to model
        @return: loaded model
        """
        try:
            if not f:
                logger.info(f"result path: {self.results_path} ")
                logger.info(f"loading model form {self.default_model_path} ")
                model = pickle.load(open(self.default_model_path, 'rb'))
            else:
                logger.info(f"loading from {f}")
                model = pickle.load(open(f, 'rb'))
            return model
        except FileNotFoundError:
            logger.error(f"File not found in {self.default_model_path} ")

    def _prepare_fit_data(self):
        return self._process_data(target='fit')

    def _prepare_eval_data(self):
        return self._process_data(target='evaluate')

    def _process_data(self, target='fit'):
        """
        read and return data as x and y
        @return: list of separate x and y
        """
        assert isinstance(self.target, list), "provide target(s) as a list in the yaml file"
        if self.model_type != "clustering":
            assert len(self.target) > 0, "please provide at least a target to predict"

        try:
            read_data_options = self.dataset_props.get('read_data_options', None)
            dataset = pd.read_csv(self.data_path) if not read_data_options else pd.read_csv(self.data_path,
                                                                                            **read_data_options)
            logger.info(f"dataset shape: {dataset.shape}")
            attributes = list(dataset.columns)
            logger.info(f"dataset attributes: {attributes}")

            # handle missing values in the dataset
            preprocess_props = self.dataset_props.get('preprocess', None)
            if preprocess_props:
                # handle encoding
                encoding = preprocess_props.get('encoding')
                if encoding:
                    encoding_type = encoding.get('type', None)
                    column = encoding.get('column', None)
                    if column in attributes:
                        dataset, classes_map = encode(df=dataset,
                                                      encoding_type=encoding_type.lower(),
                                                      column=column)
                        if classes_map:
                            self.dataset_props['label_encoding_classes'] = classes_map
                            logger.info(f"adding classes_map to dataset props: \n{classes_map}")
                        logger.info(f"shape of the dataset after encoding => {dataset.shape}")

                # preprocessing strategy: mean, median, mode etc..
                strategy = preprocess_props.get('missing_values')
                if strategy:
                    dataset = handle_missing_values(dataset,
                                                    strategy=strategy)
                    logger.info(f"shape of the dataset after handling missing values => {dataset.shape}")

            if target == 'predict' or target == 'fit_cluster':
                x = _reshape(dataset.to_numpy())
                if not preprocess_props:
                    return x
                scaling_props = preprocess_props.get('scale', None)
                if not scaling_props:
                    return x
                else:
                    scaling_method = scaling_props.get('method', None)
                    return normalize(x, method=scaling_method)

            if any(col not in attributes for col in self.target):
                raise Exception("chosen target(s) to predict must exist in the dataset")

            y = pd.concat([dataset.pop(x) for x in self.target], axis=1)
            x = _reshape(dataset.to_numpy())
            y = _reshape(y.to_numpy())
            logger.info(f"y shape: {y.shape} and x shape: {x.shape}")

            # handle data scaling
            if preprocess_props:
                scaling_props = preprocess_props.get('scale', None)
                if scaling_props:
                    scaling_method = scaling_props.get('method', None)
                    scaling_target = scaling_props.get('target', None)
                    if scaling_target == 'all':
                        x = normalize(x, method=scaling_method)
                        y = normalize(y, method=scaling_method)
                    elif scaling_target == 'inputs':
                        x = normalize(x, method=scaling_method)
                    elif scaling_target == 'outputs':
                        y = normalize(y, method=scaling_method)

            if target == 'evaluate':
                return x, y

            split_options = self.dataset_props.get('split', None)
            if not split_options:
                return x, y, None, None
            test_size = split_options.get('test_size')
            shuffle = split_options.get('shuffle')
            stratify = split_options.get('stratify')
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=test_size,
                shuffle=shuffle,
                stratify=None if not stratify or stratify.lower() == "default" else stratify)

            return x_train, y_train, x_test, y_test

        except Exception as e:
            logger.exception(f"error occured while preparing the data: {e.args}")

    def _prepare_clustering_data(self):
        """
        preprocess data for the clustering algorithm
        """
        return self._process_data(target='fit_cluster')

    def _prepare_predict_data(self):
        """
        preprocess predict data to get similar data to the one used when training the model
        """
        return self._process_data(target='predict')

    def get_evaluation(self, model, x_test, y_true, y_pred, **kwargs):
        try:
            res = evaluate_model(model_type=self.model_type,
                                 model=model,
                                 x_test=x_test,
                                 y_pred=y_pred,
                                 y_true=y_true,
                                 get_score_only=False,
                                 **kwargs)
        except Exception as e:
            logger.debug(e)
            res = evaluate_model(model_type=self.model_type,
                                 model=model,
                                 x_test=x_test,
                                 y_pred=y_pred,
                                 y_true=y_true,
                                 get_score_only=True,
                                 **kwargs)
        return res

    def fit(self, **kwargs):
        """
        fit a machine learning model and save it to a file along with a description.json file
        @return: None
        """
        x_train = None
        x_test = None
        y_train = None
        y_test = None
        cv_results = None
        eval_results = None
        cv_params = None
        hp_search_results = {}

        if self.model_type == 'clustering':
            x_train = self._prepare_clustering_data()
        else:
            x_train, y_train, x_test, y_test = self._prepare_fit_data()
        self.model, model_args = self._create_model(**kwargs)
        logger.info(f"executing a {self.model.__class__.__name__} algorithm...")

        # convert to multioutput if there is more than one target to predict:
        if self.model_type != 'clustering' and len(self.target) > 1:
            logger.info(f"predicting multiple targets detected. Hence, the model will be automatically "
                        f"converted to a multioutput model")
            self.model = MultiOutputClassifier(self.model) \
                if self.model_type == 'classification' else MultiOutputRegressor(self.model)

        if self.model_type != 'clustering':
            cv_params = self.model_props.get('cross_validate', None)
            if not cv_params:
                logger.info(f"cross validation is not provided")
            else:
                # perform cross validation
                logger.info("performing cross validation ...")
                cv_results = cross_validate(estimator=self.model,
                                            X=x_train,
                                            y=y_train,
                                            **cv_params)
            hyperparams_props = self.model_props.get('hyperparameter_search', None)
            if hyperparams_props:
                method = hyperparams_props.get('method', None)
                grid_params = hyperparams_props.get('parameter_grid', None)
                hp_args = hyperparams_props.get('arguments', None)
                logger.info(f"Performing hyperparameter search using -> {method}")
                logger.info(f"Grid parameters entered by the user: {grid_params}")
                logger.info(f"Additional hyperparameter arguments: {hp_args}")
                best_estimator, best_params, best_score = hyperparameter_search(model=self.model,
                                                                                method=method,
                                                                                params=grid_params,
                                                                                x_train=x_train,
                                                                                y_train=y_train,
                                                                                **hp_args)
                hp_search_results['best_params'] = best_params
                hp_search_results['best_score'] = best_score
                self.model = best_estimator

            self.model.fit(x_train, y_train)

        else:   # if the model type is clustering
            self.model.fit(x_train)

        saved = self._save_model(self.model)
        if saved:
            logger.info(f"model saved successfully and can be found in the {self.results_path} folder")

        if self.model_type == 'clustering':
            eval_results = self.model.score(x_train)
        else:
            if x_test is None:
                logger.info(f"no split options was provided. training score will be calculated")
                eval_results = self.model.score(x_train, y_train)

            else:
                logger.info(f"split option detected. The performance will be automatically evaluated "
                            f"using the test data portion")
                y_pred = self.model.predict(x_test)
                eval_results = self.get_evaluation(model=self.model,
                                                   x_test=x_test,
                                                   y_true=y_test,
                                                   y_pred=y_pred,
                                                   **kwargs)

        fit_description = {
            "model": self.model.__class__.__name__,
            "arguments": model_args if model_args else "default",
            "type": self.model_props['type'],
            "algorithm": self.model_props['algorithm'],
            "dataset_props": self.dataset_props,
            "model_props": self.model_props,
            "data_path": self.data_path,
            "train_data_shape": x_train.shape,
            "test_data_shape": None if x_test is None else x_test.shape,
            "train_data_size": x_train.shape[0],
            "test_data_size": None if x_test is None else x_test.shape[0],
            "results_path": str(self.results_path),
            "model_path": str(self.default_model_path),
            "target": None if self.model_type == 'clustering' else self.target,
            "results_on_test_data": eval_results,
            "hyperparameter_search_results": hp_search_results

        }
        if self.model_type == 'clustering':
            clustering_res = {
                "cluster_centers": self.model.cluster_centers_,
                "cluster_labels": self.model.labels_
            }
            fit_description['clustering_results'] = clustering_res

        if cv_params:
            cv_res = {
                "fit_time": cv_results['fit_time'].tolist(),
                "score_time": cv_results['score_time'].tolist(),
                "test_score": cv_results['test_score'].tolist()
            }
            fit_description['cross_validation_params'] = cv_params
            fit_description['cross_validation_results'] = cv_res

        try:
            logger.info(f"saving fit description to {self.description_file}")
            with open(self.description_file, 'w', encoding='utf-8') as f:
                json.dump(fit_description, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.exception(f"Error while storing the fit description file: {e}")

    def evaluate(self, **kwargs):
        """
        evaluate a pre-fitted model and save results to a evaluation.json
        @return: None
        """
        x_val = None
        y_true = None
        eval_results = None
        try:
            model = self._load_model()
            if self.model_type != 'clustering':
                x_val, y_true = self._prepare_eval_data()
                y_pred = model.predict(x_val)
                eval_results = self.get_evaluation(model=model,
                                                   x_test=x_val,
                                                   y_true=y_true,
                                                   y_pred=y_pred,
                                                   **kwargs)
            else:
                x_val = self._prepare_clustering_data()
                y_pred = model.predict(x_val)
                eval_results = model.score(x_val, y_pred)

            logger.info(f"saving fit description to {self.evaluation_file}")
            with open(self.evaluation_file, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, ensure_ascii=False, indent=4)

        except Exception as e:
            logger.exception(f"error occured during evaluation: {e}")

    def predict(self):
        """
        use a pre-fitted model to make predictions and save them as csv
        @return: None
        """
        try:
            model = self._load_model(f=self.model_path)
            x_val = self._prepare_predict_data()  # the same is used for clustering
            y_pred = model.predict(x_val)
            y_pred = _reshape(y_pred)
            logger.info(f"predictions shape: {y_pred.shape} | shape len: {len(y_pred.shape)}")
            logger.info(f"predict on targets: {self.target}")
            df_pred = pd.DataFrame.from_dict(
                {self.target[i]: y_pred[:, i] if len(y_pred.shape) > 1 else y_pred for i in range(len(self.target))})

            logger.info(f"saving the predictions to {self.prediction_file}")
            df_pred.to_csv(self.prediction_file)

        except Exception as e:
            logger.exception(f"Error while preparing predictions: {e}")

    @staticmethod
    def create_init_mock_file(model_type=None, model_name=None, target=None, *args, **kwargs):
        path = configs.get('init_file_path', None)
        if not path:
            raise Exception("You need to provide a path for the init file")

        dataset_props = Igel.default_dataset_props
        model_props = Igel.default_model_props
        if model_type:
            logger.info(f"user selected model type = {model_type}")
            model_props['type'] = model_type
        if model_name:
            logger.info(f"user selected algorithm = {model_name}")
            model_props['algorithm'] = model_name

        logger.info(f"initalizing a default igel.yaml in {path}")
        default_data = {
            "dataset": dataset_props,
            "model": model_props,
            "target": ['provide your target(s) here'] if not target else [tg for tg in target.split()]
        }
        created = create_yaml(default_data, path)
        if created:
            logger.info(f"a default igel.yaml is created for you in {path}. "
                        f"you just need to overwrite the values to meet your expectations")
        else:
            logger.warning(f"something went wrong while initializing a default file")

