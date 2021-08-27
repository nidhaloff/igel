import json
import logging

import joblib
import pandas as pd
import yaml
from igel.configs import configs
from igel.data import metrics_dict, models_dict

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


def show_model_info(model_name: str, model_type: str):
    if not model_name:
        print(f"Please enter a supported model")
        print_models_overview()
    else:
        if not model_type:
            print(
                f"Please enter a type argument to get help on the chosen model\n"
                f"type can be whether regression, classification or clustering \n"
            )
            print_models_overview()
            return
        if model_type not in ("regression", "classification", "clustering"):
            raise Exception(
                f"{model_type} is not supported! \n"
                f"model_type need to be regression, classification or clustering"
            )

        models = models_dict.get(model_type)
        model_data = models.get(model_name)
        model, link, *cv_class = model_data.values()
        print(
            f"model type: {model_type} \n"
            f"model name: {model_name} \n"
            f"sklearn model class: {model.__name__} \n"
            f"{'-' * 60}\n"
            f"You can click the link below to know more about the optional arguments\n"
            f"that you can use with your chosen model ({model_name}).\n"
            f"You can provide these optional arguments in the yaml file if you want to use them.\n"
            f"link:\n{link} \n"
        )


def tableize(df):
    """
    pretty-print a dataframe as table
    """
    if not isinstance(df, pd.DataFrame):
        return
    df_columns = df.columns.tolist()
    max_len_in_lst = lambda lst: len(sorted(lst, reverse=True, key=len)[0])
    align_center = (
        lambda st, sz: "{0}{1}{0}".format(" " * (1 + (sz - len(st)) // 2), st)[
            :sz
        ]
        if len(st) < sz
        else st
    )
    align_right = (
        lambda st, sz: "{}{} ".format(" " * (sz - len(st) - 1), st)
        if len(st) < sz
        else st
    )
    max_col_len = max_len_in_lst(df_columns)
    max_val_len_for_col = {
        col: max_len_in_lst(df.iloc[:, idx].astype("str"))
        for idx, col in enumerate(df_columns)
    }
    col_sizes = {
        col: 2 + max(max_val_len_for_col.get(col, 0), max_col_len)
        for col in df_columns
    }
    build_hline = lambda row: "+".join(
        ["-" * col_sizes[col] for col in row]
    ).join(["+", "+"])
    build_data = lambda row, align: "|".join(
        [
            align(str(val), col_sizes[df_columns[idx]])
            for idx, val in enumerate(row)
        ]
    ).join(["|", "|"])
    hline = build_hline(df_columns)
    out = [hline, build_data(df_columns, align_center), hline]
    for _, row in df.iterrows():
        out.append(build_data(row.tolist(), align_right))
    out.append(hline)
    return "\n".join(out)


def print_models_overview():
    print(f"\nIgel's supported models overview: \n")
    reg_algs = list(models_dict.get("regression").keys())
    clf_algs = list(models_dict.get("classification").keys())
    cluster_algs = list(models_dict.get("clustering").keys())
    df_algs = (
        pd.DataFrame.from_dict(
            {
                "regression": reg_algs,
                "classification": clf_algs,
                "clustering": cluster_algs,
            },
            orient="index",
        )
        .transpose()
        .fillna("----")
    )

    df = tableize(df_algs)
    print(df)
