"""Console script for igel."""
import logging
import os
import subprocess
from pathlib import Path
import json

import click
import igel
import pandas as pd
from igel import Igel, metrics_dict
from igel.constants import Constants
from igel.servers import fastapi_server
from igel.utils import print_models_overview, show_model_info, tableize
from igel.model_registry import models_dict

logger = logging.getLogger(__name__)
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group()
@click.option('--verbose', is_flag=True, help='Enable verbose output for debugging.')
def cli(verbose):
    """
    The igel command line interface
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        click.echo('Verbose mode is on.')
    else:
        logging.basicConfig(level=logging.WARNING)
        logger.setLevel(logging.WARNING)


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--model_type",
    "-type",
    type=click.Choice(Constants.supported_model_types, case_sensitive=False),
    help="type of the problem you want to solve",
)
@click.option(
    "--model_name",
    "-name",
    help="algorithm you want to use",
)
@click.option(
    "--target",
    "-tg",
    help="target you want to predict (this is usually the name of column you want to predict)",
)
def init(model_type: str, model_name: str, target: str) -> None:
    """
    Initialize a new igel project.
    This command can be run interactively or by providing command line arguments.

    Example:
        igel init
        igel init --model_type=classification --model_name=RandomForest --target=label
    """
    if not model_type:
        model_type = click.prompt(
            "Please choose a model type",
            type=click.Choice(Constants.supported_model_types, case_sensitive=False)
        )

    algorithms = models_dict.get(model_type, {})
    available_algorithms = list(algorithms.keys())

    if not model_name:
        model_name = click.prompt(
            "Please choose an algorithm",
            type=click.Choice(available_algorithms, case_sensitive=False)
        )

    if not target:
        target = click.prompt("Please enter the target column(s) you want to predict (comma-separated)")

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
    Fit/train a machine learning model.

    Example:
        igel fit --data_path=data/train.csv --yaml_path=config.yaml
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
    """
    Show the current igel version.
    """
    click.echo(f"igel version: {igel.__version__}")


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


@cli.group()
def registry():
    """
    Manage model registry operations
    """
    pass

@registry.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--model_name",
    "-name",
    help="Filter models by name",
)
def list_models(model_name: str) -> None:
    """
    List all registered models or filter by name
    """
    models = Igel.list_models(model_name)
    if not models:
        click.echo("No models found in registry.")
        return
    
    # Create a table of models
    table_data = []
    for model in models:
        table_data.append({
            "ID": model["model_id"],
            "Name": model["model_name"],
            "Version": model["version"],
            "Type": model["model_type"],
            "Algorithm": model["model_algorithm"],
            "Created": model["timestamp"]
        })
    
    df = pd.DataFrame(table_data)
    click.echo(tableize(df))

@registry.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--model_id",
    "-id",
    required=True,
    help="Model ID to get information about",
)
def info(model_id: str) -> None:
    """
    Get detailed information about a specific model
    """
    model_info = Igel.get_model_info(model_id)
    if not model_info:
        click.echo(f"No model found with ID: {model_id}")
        return
    
    click.echo("\nModel Information:")
    for key, value in model_info.items():
        if isinstance(value, dict):
            click.echo(f"\n{key}:")
            for k, v in value.items():
                click.echo(f"  {k}: {v}")
        else:
            click.echo(f"{key}: {value}")

@registry.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--model_id",
    "-id",
    required=True,
    help="Model ID to delete",
)
def delete(model_id: str) -> None:
    """
    Delete a model from the registry
    """
    if Igel.delete_model(model_id):
        click.echo(f"Successfully deleted model: {model_id}")
    else:
        click.echo(f"Failed to delete model: {model_id}")

@registry.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--model_id",
    "-id",
    required=True,
    help="Model ID to update",
)
@click.option(
    "--metadata",
    "-m",
    required=True,
    help="JSON string containing metadata to update",
)
def update(model_id: str, metadata: str) -> None:
    """
    Update metadata for a registered model
    """
    try:
        metadata_dict = json.loads(metadata)
        if Igel.update_model_metadata(model_id, metadata_dict):
            click.echo(f"Successfully updated metadata for model: {model_id}")
        else:
            click.echo(f"Failed to update metadata for model: {model_id}")
    except json.JSONDecodeError:
        click.echo("Error: Invalid JSON format for metadata")

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--model_a_path",
    "-ma",
    required=True,
    help="Path to first model for comparison",
)
@click.option(
    "--model_b_path",
    "-mb",
    required=True,
    help="Path to second model for comparison",
)
@click.option(
    "--test_data",
    "-td",
    required=True,
    help="Path to test dataset for comparison",
)
@click.option(
    "--problem_type",
    "-pt",
    type=click.Choice(["classification", "regression"]),
    default="classification",
    help="Type of problem (classification or regression)",
)
def compare_models(model_a_path: str, model_b_path: str, test_data: str, problem_type: str) -> None:
    """
    Compare two trained models using A/B testing framework.
    
    This command loads two trained models and compares their performance using
    statistical tests to determine if there are significant differences.
    """
    try:
        from igel.ab_testing import ModelComparison
        import joblib
        import pandas as pd
        
        # Load models
        model_a = joblib.load(model_a_path)
        model_b = joblib.load(model_b_path)
        
        # Load test data
        test_df = pd.read_csv(test_data)
        X_test = test_df.drop(columns=['target']).values  # Assuming 'target' is the label column
        y_test = test_df['target'].values
        
        # Initialize comparison
        comparison = ModelComparison(model_a, model_b, test_type=problem_type)
        
        # Run comparison
        results = comparison.compare_predictions(X_test, y_test)
        
        # Generate and print report
        report = comparison.generate_report(results)
        print("\n" + report + "\n")
        
    except Exception as e:
        logger.exception(f"Error during model comparison: {e}")
        raise click.ClickException(str(e))
