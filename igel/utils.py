import json
import logging

import joblib
import yaml
from igel.configs import configs

logger = logging.getLogger(__name__)


def create_yaml(data, f):
    try:
        with open(f, "w") as yf:
            yaml.dump(data, yf, default_flow_style=False)
    except yaml.YAMLError as exc:
        logger.exception(exc)
        return False
    else:
        return True


def read_yaml(f):
    with open(f) as stream:
        try:
            res = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.exception(exc)
        else:
            return res


def read_json(f):
    try:
        with open(f) as file:
            data = json.load(file)
    except Exception as e:
        logger.exception(e.args)
    else:
        return data


def extract_params(config):
    assert (
        "model" in config.keys()
    ), "model parameters need to be provided in the yaml file"
    assert (
        "target" in config.keys()
    ), "target variable needs to be provided in the yaml file"
    model_params = config.get("model")
    model_type = model_params.get("type")
    algorithm = model_params.get("algorithm")
    target = config.get("target")

    if any(not item for item in [model_type, target, algorithm]):
        raise Exception("parameters in the model yaml file cannot be None")
    else:
        return model_type, target, algorithm


def _reshape(arr):
    if len(arr.shape) <= 1:
        arr = arr.reshape(-1, 1)
    return arr


def load_trained_model(f: str = ""):
    """
    load a saved model from file
    @param f: path to model
    @return: loaded model
    """
    try:
        if not f:
            logger.info(f"result path: {configs.get('results_path')} ")
            logger.info(
                f"loading model form {configs.get('default_model_path')} "
            )
            with open(configs.get("default_model_path"), "rb") as _model:
                model = joblib.load(_model)
        else:
            logger.info(f"loading from {f}")
            with open(f, "rb") as _model:
                model = joblib.load(_model)
        return model
    except FileNotFoundError:
        logger.error(f"File not found in {configs.get('default_model_path')}")


def load_train_configs(f=""):
    """
    load train configurations from model_results/descriptions.json
    """
    try:
        if not f:
            logger.info(
                f"loading descriptions.json form {configs.get('description_file')} "
            )
            with open(configs.get("description_file"), "rb") as desc_file:
                training_config = json.load(desc_file)
        else:
            with open(f, "rb") as desc_file:
                training_config = json.load(desc_file)
        return training_config

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(e)


def get_expected_scaling_method(training_config):
    """
    get expected scaling method from the parsed training configuration (description.json)
    """
    dataset_props = training_config.get("dataset_props")
    if not dataset_props:
        return
    preprocess_options = dataset_props.get("preprocess")
    if not preprocess_options:
        return
    scaling_options = preprocess_options.get("scale")
    if not scaling_options:
        return
    return scaling_options.get("method")
