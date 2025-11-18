"""Console script for igel."""
import logging
import os
import subprocess
from pathlib import Path

import click
import igel
import pandas as pd
from igel import Igel, metrics_dict
from igel.constants import Constants
from igel.servers import fastapi_server
from igel.utils import print_models_overview, show_model_info, tableize

logger = logging.getLogger(__name__)
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group()
def cli():
    """
    The igel command line interface
    """
    pass


@cli.command(context_settings=CONTEXT_SETTINGS)
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


@cli.command(context_settings=CONTEXT_SETTINGS)
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


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--data_path", "-dp", required=True, help="Path to your evaluation dataset"
)
def evaluate(data_path: str) -> None:
    """
    Evaluate the performance of an existing machine learning model
    """
    Igel(cmd="evaluate", data_path=data_path)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option("--data_path", "-dp", required=True, help="Path to your dataset")
def predict(data_path: str) -> None:
    """
    Use an existing machine learning model to generate predictions
    """
    Igel(cmd="predict", data_path=data_path)


@cli.command(context_settings=CONTEXT_SETTINGS)
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

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option("--model_path", "-dp", required=True, help="Path to your sklearn model")
def export(model_path: str) -> None:
    """
    Export an existing machine learning model to ONNX
    """
    Igel(cmd="export", model_path=model_path)

@cli.command(context_settings=CONTEXT_SETTINGS)
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


@cli.command(context_settings=CONTEXT_SETTINGS)
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


@cli.command(context_settings=CONTEXT_SETTINGS)
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


@cli.command(context_settings=CONTEXT_SETTINGS)
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


@cli.command(context_settings=CONTEXT_SETTINGS)
def help():
    """get help about how to use igel"""
    with click.Context(cli) as ctx:
        click.echo(cli.get_help(ctx))


@cli.command(context_settings=CONTEXT_SETTINGS)
def version():
    """get the version of igel installed on your machine"""
    print(f"igel version: {igel.__version__}")


@cli.command(context_settings=CONTEXT_SETTINGS)
def info():
    """get info & metadata about igel"""
    print(
        f"""
        package name:           igel
        version:                {igel.__version__}
        author:                 Nidhal Baccouri
        maintainer:             Nidhal Baccouri
        contact:                nidhalbacc@gmail.com
        license:                MIT
        description:            use machine learning without writing code
        dependencies:           pandas, sklearn, pyyaml
        requires python:        >= 3.6
        First release:          27.08.2020
        official repo:          https://github.com/nidhaloff/igel
        written in:             100% python
        status:                 stable
        operating system:       independent
    """
    )


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--model1_dir",
    "-m1",
    required=True,
    help="Path to first model results directory",
)
@click.option(
    "--model2_dir",
    "-m2",
    required=True,
    help="Path to second model results directory",
)
def compare_models(model1_dir: str, model2_dir: str) -> None:
    """
    Compare two trained models side by side
    """
    import json
    from datetime import datetime
    
    def load_model_info(model_dir: str) -> dict:
        """Load model information from a results directory."""
        model_path = Path(model_dir)
        description_file = model_path / "description.json"
        evaluation_file = model_path / "evaluation.json"
        
        info = {
            "dir": model_dir,
            "description": None,
            "evaluation": None,
            "model_file": None,
            "exists": description_file.exists()
        }
        
        if description_file.exists():
            try:
                with open(description_file, 'r') as f:
                    info["description"] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load description from {model_dir}: {e}")
        
        if evaluation_file.exists():
            try:
                with open(evaluation_file, 'r') as f:
                    info["evaluation"] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load evaluation from {model_dir}: {e}")
        
        model_file = model_path / "model.pkl"
        if model_file.exists():
            info["model_file"] = {
                "size_mb": model_file.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(model_file.stat().st_mtime)
            }
        
        return info
    
    model1_info = load_model_info(model1_dir)
    model2_info = load_model_info(model2_dir)
    
    if not model1_info["exists"]:
        click.echo(f"❌ Model 1 directory not found: {model1_dir}")
        return
    
    if not model2_info["exists"]:
        click.echo(f"❌ Model 2 directory not found: {model2_dir}")
        return
    
    click.echo("\n📊 Model Comparison\n")
    click.echo("=" * 70)
    
    # Model Type Comparison
    click.echo("\n🔹 Model Configuration:")
    desc1 = model1_info.get("description", {})
    desc2 = model2_info.get("description", {})
    
    model_type1 = desc1.get("type", "N/A")
    model_type2 = desc2.get("type", "N/A")
    click.echo(f"  Model 1 Type:     {model_type1}")
    click.echo(f"  Model 2 Type:     {model_type2}")
    
    # Algorithm Comparison
    algo1 = desc1.get("model_props", {}).get("algorithm", "N/A")
    algo2 = desc2.get("model_props", {}).get("algorithm", "N/A")
    click.echo(f"  Model 1 Algorithm: {algo1}")
    click.echo(f"  Model 2 Algorithm: {algo2}")
    
    # Performance Metrics Comparison
    eval1 = model1_info.get("evaluation", {})
    eval2 = model2_info.get("evaluation", {})
    
    if eval1 and eval2:
        click.echo("\n🔹 Performance Metrics:")
        click.echo(f"{'Metric':<25} {'Model 1':<20} {'Model 2':<20} {'Winner':<10}")
        click.echo("-" * 75)
        
        # Get all unique metrics
        all_metrics = set(eval1.keys()) | set(eval2.keys())
        
        for metric in sorted(all_metrics):
            val1 = eval1.get(metric, None)
            val2 = eval2.get(metric, None)
            
            if val1 is None or val2 is None:
                continue
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Determine winner (higher is better for most metrics)
                # For regression, lower is better for MSE, MAE, etc.
                if "mse" in metric.lower() or "mae" in metric.lower() or "error" in metric.lower():
                    winner = "Model 1" if val1 < val2 else "Model 2" if val2 < val1 else "Tie"
                else:
                    winner = "Model 1" if val1 > val2 else "Model 2" if val2 > val1 else "Tie"
                
                click.echo(f"{metric:<25} {val1:<20.4f} {val2:<20.4f} {winner:<10}")
    
    # File Size Comparison
    if model1_info.get("model_file") and model2_info.get("model_file"):
        click.echo("\n🔹 Model File Information:")
        size1 = model1_info["model_file"]["size_mb"]
        size2 = model2_info["model_file"]["size_mb"]
        click.echo(f"  Model 1 Size:     {size1:.2f} MB")
        click.echo(f"  Model 2 Size:     {size2:.2f} MB")
        click.echo(f"  Size Difference: {abs(size1 - size2):.2f} MB")
    
    click.echo("\n" + "=" * 70)
    click.echo()