"""Console script for igel."""
import logging
import os
import subprocess
from pathlib import Path

import click
import pandas as pd
from igel import Igel, metrics_dict
from igel.constants import Constants
from igel.servers import fastapi_server
from igel.utils import print_models_overview, show_model_info, tableize

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """
    The igel command line interface
    """
    pass


@cli.command()
@click.option(
    "--model_type",
    "-type",
    default="regression",
    show_default=True,
    type=click.Choice(Constants.supported_model_types, case_sensitive=False),
    help="type of the problem you want to solve",
)
@click.option(
    "--model_name",
    "-name",
    default="NeuralNetwork",
    show_default=True,
    help="algorithm you want to use",
)
@click.option(
    "--target",
    "-tg",
    required=True,
    help="target you want to predict (this is usually the name of column you want to predict)",
)
def init(model_type: str, model_name: str, target: str) -> None:
    """
    Initialize a new igel project.
    """
    Igel.create_init_mock_file(
        model_type=model_type, model_name=model_name, target=target
    )


@cli.command()
@click.option(
    "--data_path", "-dp", required=True, help="Path to your training dataset"
)
@click.option(
    "--yaml_path",
    "-yml",
    required=True,
    help="Path to your igel configuration file (yaml or json file)",
)
def fit(data_path: str, yaml_path: str) -> None:
    """
    fit/train a machine learning model
    """
    Igel(cmd="fit", data_path=data_path, yaml_path=yaml_path)


@cli.command()
@click.option(
    "--data_path", "-dp", required=True, help="Path to your evaluation dataset"
)
def evaluate(data_path: str) -> None:
    """
    Evaluate the performance of an existing machine learning model
    """
    Igel(cmd="evaluate", data_path=data_path)


@cli.command()
@click.option("--data_path", "-dp", required=True, help="Path to your dataset")
def predict(data_path: str) -> None:
    """
    Use an existing machine learning model to generate predictions
    """
    Igel(cmd="predict", data_path=data_path)


@cli.command()
@click.option(
    "--data_paths",
    "-DP",
    required=True,
    help="Path to your datasets as string separated by space",
)
@click.option(
    "--yaml_path",
    "-yml",
    required=True,
    help="Path to your igel configuration file (yaml or json file)",
)
def experiment(data_paths: str, yaml_path: str) -> None:
    """
    train, evaluate and use pre-trained model for predictions in one command
    """
    train_data_path, eval_data_path, pred_data_path = data_paths.strip().split(
        " "
    )
    Igel(cmd="fit", data_path=train_data_path, yaml_path=yaml_path)
    Igel(cmd="evaluate", data_path=eval_data_path)
    Igel(cmd="predict", data_path=pred_data_path)


@cli.command()
@click.option(
    "--model_results_dir",
    "-res_dir",
    required=True,
    help="Path to your model results directory",
)
@click.option(
    "--host", "-h", default="localhost", show_default=True, help="server host"
)
@click.option(
    "--port", "-p", default="8080", show_default=True, help="server host"
)
def serve(model_results_dir: str, host: str, port: int):
    """
    expose a REST endpoint in order to use the trained machine learning model
    """
    try:
        os.environ[Constants.model_results_path] = model_results_dir
        uvicorn_params = {"host": host, "port": int(port)}
        fastapi_server.run(**uvicorn_params)

    except Exception as ex:
        logger.exception(ex)


@cli.command()
@click.option(
    "--model_type",
    "-type",
    type=click.Choice(Constants.supported_model_types, case_sensitive=False),
    help="type of the model you want to inspect",
)
@click.option("--model_name", "-name", help="algorithm you want to use")
def models(model_type: str, model_name: str) -> None:
    """
    show an overview of all models supported by igel
    """
    if not model_type or not model_name:
        print_models_overview()
    else:
        show_model_info(model_type=model_type, model_name=model_name)


@cli.command()
def metrics():
    """
    show an overview of all metrics supported by igel
    """
    print(f"\nIgel's supported metrics overview: \n")
    reg_metrics = [func.__name__ for func in metrics_dict.get("regression")]
    clf_metrics = [func.__name__ for func in metrics_dict.get("classification")]

    df_metrics = (
        pd.DataFrame.from_dict(
            {"regression": reg_metrics, "classification": clf_metrics},
            orient="index",
        )
        .transpose()
        .fillna("----")
    )

    df_metrics = tableize(df_metrics)
    print(df_metrics)


@cli.command()
def gui():
    """
    Launch the igel gui application.
    PS: you need to have nodejs on your machine
    """
    igel_ui_path = Path(os.getcwd()) / "igel-ui"
    if not Path.exists(igel_ui_path):
        subprocess.check_call(
            ["git"] + ["clone", "https://github.com/nidhaloff/igel-ui.git"]
        )
        logger.info(f"igel UI cloned successfully")

    os.chdir(igel_ui_path)
    logger.info(f"switching to -> {igel_ui_path}")
    logger.info(f"current dir: {os.getcwd()}")
    logger.info(f"make sure you have nodejs installed!!")

    subprocess.Popen(["node", "npm", "install", "open"], shell=True)
    subprocess.Popen(["node", "npm", "install electron", "open"], shell=True)
    logger.info("installing dependencies ...")
    logger.info(f"dependencies installed successfully")
    logger.info(f"node version:")
    subprocess.check_call("node -v", shell=True)
    logger.info(f"npm version:")
    subprocess.check_call("npm -v", shell=True)
    subprocess.check_call("npm i electron", shell=True)
    logger.info("running igel UI...")
    subprocess.check_call("npm start", shell=True)
