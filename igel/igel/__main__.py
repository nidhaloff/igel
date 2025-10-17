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
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without executing"
)
def fit(data_path: str, yaml_path: str, dry_run: bool) -> None:
    """
    Fit/train a machine learning model.

    Example:
        igel fit --data_path=data/train.csv --yaml_path=config.yaml
        igel fit --data_path=data/train.csv --yaml_path=config.yaml --dry-run
    """
    if dry_run:
        print("DRY RUN MODE - No actual training will be performed")
        print(f"Data path: {data_path}")
        print(f"Config path: {yaml_path}")
        
        # Load and display config
        import yaml
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration: {config}")
        
        # Check if data exists
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"Dataset shape: {df.shape}")
            print(f"Dataset columns: {list(df.columns)}")
        else:
            print(f"Warning: Data file {data_path} not found")
        
        print("Dry run completed - no model was trained")
        return
    
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
@click.option(
    "--confidence_level",
    "-cl",
    type=float,
    default=0.95,
    help="Confidence level for statistical tests (default: 0.95)",
)
@click.option(
    "--target_column",
    "-tc",
    default="target",
    help="Name of the target column in the dataset (default: 'target')",
)
@click.option(
    "--export_results",
    "-er",
    help="Path to export results as JSON file",
)
@click.option(
    "--visualize",
    "-v",
    is_flag=True,
    help="Generate visualization plots",
)
@click.option(
    "--save_plot",
    "-sp",
    help="Path to save visualization plot",
)
@click.option(
    "--legacy_mode",
    is_flag=True,
    help="Use legacy A/B testing mode for backward compatibility",
)
def compare_models(model_a_path: str, model_b_path: str, test_data: str, problem_type: str,
                  confidence_level: float, target_column: str, export_results: str, 
                  visualize: bool, save_plot: str, legacy_mode: bool) -> None:
    """
    Compare two trained models using enhanced A/B testing framework.
    
    This command loads two trained models and compares their performance using
    comprehensive statistical tests to determine if there are significant differences.
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
        if target_column not in test_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {list(test_df.columns)}")
        
        X_test = test_df.drop(columns=[target_column]).values
        y_test = test_df[target_column].values
        
        # Initialize comparison
        comparison = ModelComparison(model_a, model_b, test_type=problem_type)
        
        if legacy_mode:
            # Use legacy mode for backward compatibility
            results = comparison.compare_predictions(X_test, y_test)
            # Legacy report format
            report = comparison.generate_report(results)
            print("\n" + report + "\n")
        else:
            # Enhanced mode with new features
            results = comparison.compare_predictions(X_test, y_test, confidence_level=confidence_level)
            
            # Generate and print report
            report = comparison.generate_report(results)
            print("\n" + report + "\n")
            
            # Export results if requested
            if export_results:
                comparison.export_results(results, export_results)
                print(f"Results exported to: {export_results}")
            
            # Generate visualization if requested
            if visualize or save_plot:
                comparison.visualize_comparison(results, save_path=save_plot)
                if not save_plot:
                    print("Visualization displayed.")
        
    except Exception as e:
        logger.exception(f"Error during model comparison: {e}")
        raise click.ClickException(str(e))

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--ref_data', required=True, help='Path to reference (e.g., training) data CSV')
@click.option('--new_data', required=True, help='Path to new (e.g., production) data CSV')
@click.option('--categorical', default='', help='Comma-separated list of categorical feature names')
def detect_drift(ref_data, new_data, categorical):
    """
    Detect data drift between two datasets (reference and new).
    Uses KS test for numerical and Chi-Squared for categorical features.
    """
    import pandas as pd
    from igel.drift_detection import detect_drift
    ref_df = pd.read_csv(ref_data)
    new_df = pd.read_csv(new_data)
    categorical_features = [c.strip() for c in categorical.split(',') if c.strip()] if categorical else None
    report = detect_drift(ref_df, new_df, categorical_features)
    print(report.to_string(index=False))

@cli.command(context_settings=CONTEXT_SETTINGS)
def gpu_info():
    """
    Show GPU availability and utilization (PyTorch, TensorFlow, GPUtil).
    """
    from igel.gpu_utils import detect_gpu, report_gpu_utilization
    print(detect_gpu())
    report_gpu_utilization()

@cli.command()
@click.option('--results_dir', default='results', help='Directory with model result files')
def leaderboard(results_dir):
    """Generate a leaderboard comparing all trained models."""
    from igel.leaderboard import generate_leaderboard
    generate_leaderboard(results_dir)

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--action', required=True, 
              type=click.Choice(['optimize', 'allocate', 'simulate']),
              help='Action to perform: optimize trajectory, allocate resources, or simulate mission')
@click.option('--config_path', required=True, help='Path to mission configuration JSON file')
@click.option('--output_path', default='mission_results.json', help='Output file path')
def space_mission(action, config_path, output_path):
    """
    Perform space mission planning operations.
    """
    import json
    from igel.space_mission import optimize_trajectory, allocate_resources, simulate_mission
    
    # Load mission configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Perform requested action
    if action == 'optimize':
        start_point = config.get('start_point', [0, 0, 0])
        end_point = config.get('end_point', [1000, 0, 0])
        constraints = config.get('constraints', {})
        results = optimize_trajectory(start_point, end_point, constraints)
    
    elif action == 'allocate':
        mission_goals = config.get('mission_goals', [])
        available_resources = config.get('available_resources', {})
        results = allocate_resources(mission_goals, available_resources)
    
    else:  # simulate
        mission_plan = config.get('mission_plan', {})
        environment_params = config.get('environment_params', {})
        results = simulate_mission(mission_plan, environment_params)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Space mission {action} completed. Results saved to {output_path}")
    print(f"Results: {results}")

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--data_path', '-dp', required=True, help='Path to your dataset')
@click.option('--yaml_path', '-yml', required=True, help='Path to your igel configuration file')
@click.option('--n_way', default=2, help='Number of classes per task (default: 2)')
@click.option('--k_shot', default=5, help='Number of examples per class for support set (default: 5)')
@click.option('--n_query', default=5, help='Number of examples per class for query set (default: 5)')
@click.option('--num_tasks', default=10, help='Number of tasks to create (default: 10)')
def few_shot_learn(data_path, yaml_path, n_way, k_shot, n_query, num_tasks):
    """
    Train a few-shot learning model using MAML or Prototypical Networks.
    
    Example:
        igel few-shot-learn --data_path=data/train.csv --yaml_path=config.yaml --n_way=3 --k_shot=5
    """
    try:
        from igel.few_shot_learning import create_few_shot_dataset, evaluate_few_shot_model
        import pandas as pd
        
        # Load data
        df = pd.read_csv(data_path)
        target_col = 'target'  # This should be configurable from yaml
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        
        # Create few-shot tasks
        tasks = create_few_shot_dataset(X, y, n_way=n_way, k_shot=k_shot, n_query=n_query)
        
        # Load configuration
        file_ext = yaml_path.split(".")[-1]
        if file_ext == "yaml":
            from igel.utils import read_yaml
            config = read_yaml(yaml_path)
        else:
            from igel.utils import read_json
            config = read_json(yaml_path)
        
        # Get model configuration
        model_props = config.get("model", {})
        model_type = model_props.get("type")
        model_algorithm = model_props.get("algorithm")
        
        if model_type != "few_shot_learning":
            raise ValueError("Model type must be 'few_shot_learning' for few-shot learning")
        
        # Create model
        if model_algorithm == "MAML":
            from igel.few_shot_learning import MAMLClassifier
            model = MAMLClassifier(
                inner_lr=model_props.get("arguments", {}).get("inner_lr", 0.01),
                outer_lr=model_props.get("arguments", {}).get("outer_lr", 0.001),
                num_tasks=model_props.get("arguments", {}).get("num_tasks", 10),
                shots_per_task=model_props.get("arguments", {}).get("shots_per_task", 5),
                inner_steps=model_props.get("arguments", {}).get("inner_steps", 5),
                meta_epochs=model_props.get("arguments", {}).get("meta_epochs", 100)
            )
        elif model_algorithm == "PrototypicalNetwork":
            from igel.few_shot_learning import PrototypicalNetwork
            model = PrototypicalNetwork(
                embedding_dim=model_props.get("arguments", {}).get("embedding_dim", 64),
                num_tasks=model_props.get("arguments", {}).get("num_tasks", 10),
                shots_per_task=model_props.get("arguments", {}).get("shots_per_task", 5),
                meta_epochs=model_props.get("arguments", {}).get("meta_epochs", 100)
            )
        else:
            raise ValueError(f"Unsupported few-shot learning algorithm: {model_algorithm}")
        
        # Train model
        print(f"Training {model_algorithm} model...")
        model.fit(X, y)
        
        # Evaluate on few-shot tasks
        print("Evaluating model on few-shot tasks...")
        results = evaluate_few_shot_model(model, tasks)
        
        print(f"\nFew-shot learning results:")
        print(f"Mean accuracy: {results['mean_accuracy']:.4f}")
        print(f"Std accuracy: {results['std_accuracy']:.4f}")
        
        # Save model
        import joblib
        joblib.dump(model, "few_shot_model.joblib")
        print("Model saved as 'few_shot_model.joblib'")
        
    except Exception as e:
        logger.exception(f"Error during few-shot learning: {e}")
        raise click.ClickException(str(e))

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--source_data', required=True, help='Path to source domain data')
@click.option('--target_data', required=True, help='Path to target domain data')
@click.option('--method', default='fine_tune', 
              type=click.Choice(['fine_tune', 'domain_adversarial', 'maml']),
              help='Domain adaptation method')
@click.option('--output_model', default='adapted_model.joblib', help='Output model path')
def domain_adapt(source_data, target_data, method, output_model):
    """
    Perform domain adaptation from source to target domain.
    
    Example:
        igel domain-adapt --source_data=source.csv --target_data=target.csv --method=fine_tune
    """
    try:
        from igel.few_shot_learning import DomainAdaptation
        import pandas as pd
        import joblib
        
        # Load data
        source_df = pd.read_csv(source_data)
        target_df = pd.read_csv(target_data)
        
        # Assume target column is 'target'
        source_X = source_df.drop(columns=['target']).values
        source_y = source_df['target'].values
        target_X = target_df.drop(columns=['target']).values
        target_y = target_df['target'].values if 'target' in target_df.columns else None
        
        # Create a base model (you could load a pre-trained model here)
        from sklearn.ensemble import RandomForestClassifier
        base_model = RandomForestClassifier(random_state=42)
        
        # Perform domain adaptation
        adapter = DomainAdaptation(base_model)
        adapted_model = adapter.adapt_model(source_X, source_y, target_X, target_y, method)
        
        # Save adapted model
        joblib.dump(adapted_model, output_model)
        print(f"Domain adapted model saved to {output_model}")
        
    except Exception as e:
        logger.exception(f"Error during domain adaptation: {e}")
        raise click.ClickException(str(e))

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--data_path', '-dp', required=True, help='Path to your dataset')
@click.option('--yaml_path', '-yml', required=True, help='Path to your igel configuration file')
@click.option('--ensemble_type', default='voting',
              type=click.Choice(['voting', 'stacking', 'blending', 'bagging', 'boosting']),
              help='Type of ensemble to create')
@click.option('--output_path', default='ensemble_model', help='Output path for ensemble model')
@click.option('--auto_ensemble', is_flag=True, help='Automatically select best ensemble type')
def create_ensemble(data_path, yaml_path, ensemble_type, output_path, auto_ensemble):
    """
    Create an advanced ensemble model using multiple base models.
    
    Example:
        igel create-ensemble --data_path=data/train.csv --yaml_path=config.yaml --ensemble_type=stacking
    """
    try:
        from igel.ensemble_framework import AdvancedEnsemble, EnsembleBuilder
        import pandas as pd
        import numpy as np
        
        # Load data
        df = pd.read_csv(data_path)
        target_col = 'target'  # This should be configurable from yaml
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        
        # Determine problem type from yaml
        file_ext = yaml_path.split(".")[-1]
        if file_ext == "yaml":
            from igel.utils import read_yaml
            config = read_yaml(yaml_path)
        else:
            from igel.utils import read_json
            config = read_json(yaml_path)
        
        model_props = config.get("model", {})
        problem_type = model_props.get("type", "classification")
        
        if auto_ensemble:
            # Automatically select best ensemble
            ensemble = EnsembleBuilder.create_auto_ensemble(X, y, problem_type)
            print(f"Auto-selected ensemble type: {ensemble.ensemble_type}")
        else:
            # Create ensemble with specified type
            if problem_type == "classification":
                ensemble = EnsembleBuilder.create_classification_ensemble(ensemble_type)
            else:
                ensemble = EnsembleBuilder.create_regression_ensemble(ensemble_type)
        
        # Create the ensemble
        ensemble.create_ensemble()
        
        # Train the ensemble
        print(f"Training {ensemble_type} ensemble...")
        ensemble.fit(X, y)
        
        # Generate and print report
        report = ensemble.generate_report()
        print("\n" + report + "\n")
        
        # Save ensemble
        ensemble.save_ensemble(output_path)
        print(f"Ensemble saved to {output_path}")
        
    except Exception as e:
        logger.exception(f"Error creating ensemble: {e}")
        raise click.ClickException(str(e))

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--ensemble_path', required=True, help='Path to saved ensemble model')
@click.option('--test_data', required=True, help='Path to test dataset')
@click.option('--output_predictions', help='Path to save predictions')
def predict_ensemble(ensemble_path, test_data, output_predictions):
    """
    Make predictions using a saved ensemble model.
    
    Example:
        igel predict-ensemble --ensemble_path=ensemble_model --test_data=test.csv
    """
    try:
        from igel.ensemble_framework import AdvancedEnsemble
        import pandas as pd
        import numpy as np
        
        # Load ensemble
        ensemble = AdvancedEnsemble.load_ensemble(ensemble_path)
        
        # Load test data
        test_df = pd.read_csv(test_data)
        if 'target' in test_df.columns:
            X_test = test_df.drop(columns=['target']).values
            y_true = test_df['target'].values
        else:
            X_test = test_df.values
            y_true = None
        
        # Make predictions
        predictions = ensemble.predict(X_test)
        
        print(f"Ensemble predictions completed. Shape: {predictions.shape}")
        
        # Calculate metrics if true labels available
        if y_true is not None:
            if ensemble.problem_type == "classification":
                accuracy = accuracy_score(y_true, predictions)
                print(f"Accuracy: {accuracy:.4f}")
            else:
                mse = mean_squared_error(y_true, predictions)
                r2 = r2_score(y_true, predictions)
                print(f"MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # Save predictions if requested
        if output_predictions:
            pred_df = pd.DataFrame({'predictions': predictions})
            pred_df.to_csv(output_predictions, index=False)
            print(f"Predictions saved to {output_predictions}")
        
    except Exception as e:
        logger.exception(f"Error making ensemble predictions: {e}")
        raise click.ClickException(str(e))

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--model_path', required=True, help='Path to the model to compress')
@click.option('--data_path', required=True, help='Path to training data')
@click.option('--compression_method', default='pruning',
              type=click.Choice(['pruning', 'quantization', 'knowledge_distillation', 'feature_selection']),
              help='Compression method to use')
@click.option('--compression_ratio', default=0.5, type=float, help='Target compression ratio (0.0 to 1.0)')
@click.option('--output_path', default='compressed_model', help='Output path for compressed model')
@click.option('--validation_data', help='Path to validation data for performance comparison')
def compress_model(model_path, data_path, compression_method, compression_ratio, output_path, validation_data):
    """
    Compress a machine learning model to reduce size while maintaining performance.
    
    Example:
        igel compress-model --model_path=model.joblib --data_path=train.csv --compression_method=pruning
    """
    try:
        from igel.model_compression import ModelCompressor
        import pandas as pd
        import joblib
        
        # Load model
        model = joblib.load(model_path)
        
        # Load data
        df = pd.read_csv(data_path)
        target_col = 'target'
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        
        # Load validation data if provided
        val_data = None
        if validation_data:
            val_df = pd.read_csv(validation_data)
            X_val = val_df.drop(columns=[target_col]).values
            y_val = val_df[target_col].values
            val_data = (X_val, y_val)
        
        # Create compressor
        compressor = ModelCompressor(
            compression_method=compression_method,
            target_compression_ratio=compression_ratio
        )
        
        # Compress model
        print(f"Compressing model using {compression_method}...")
        compressed_model = compressor.compress_model(model, X, y, validation_data=val_data)
        
        # Generate and print report
        report = compressor.get_compression_report()
        print("\n" + report + "\n")
        
        # Save compressed model
        compressor.save_compressed_model(output_path)
        print(f"Compressed model saved to {output_path}")
        
    except Exception as e:
        logger.exception(f"Error compressing model: {e}")
        raise click.ClickException(str(e))

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--model_path', required=True, help='Path to the model to optimize')
@click.option('--data_path', required=True, help='Path to training data')
@click.option('--optimization_goal', default='accuracy',
              type=click.Choice(['accuracy', 'speed', 'memory', 'balanced']),
              help='Optimization goal')
@click.option('--output_path', default='optimized_model.joblib', help='Output path for optimized model')
@click.option('--validation_data', help='Path to validation data for optimization')
def optimize_model(model_path, data_path, optimization_goal, output_path, validation_data):
    """
    Optimize a machine learning model for specific performance goals.
    
    Example:
        igel optimize-model --model_path=model.joblib --data_path=train.csv --optimization_goal=speed
    """
    try:
        from igel.model_compression import ModelOptimizer
        import pandas as pd
        import joblib
        
        # Load model
        model = joblib.load(model_path)
        
        # Load data
        df = pd.read_csv(data_path)
        target_col = 'target'
        X = df.drop(columns=[target_col]).values
        y = df[target_col].values
        
        # Load validation data if provided
        val_data = None
        if validation_data:
            val_df = pd.read_csv(validation_data)
            X_val = val_df.drop(columns=[target_col]).values
            y_val = val_df[target_col].values
            val_data = (X_val, y_val)
        
        # Create optimizer
        optimizer = ModelOptimizer(optimization_goal=optimization_goal)
        
        # Optimize model
        print(f"Optimizing model for {optimization_goal}...")
        optimized_model = optimizer.optimize_model(model, X, y, validation_data=val_data)
        
        # Generate and print report
        report = optimizer.get_optimization_report()
        print("\n" + report + "\n")
        
        # Save optimized model
        joblib.dump(optimized_model, output_path)
        print(f"Optimized model saved to {output_path}")
        
    except Exception as e:
        logger.exception(f"Error optimizing model: {e}")
        raise click.ClickException(str(e))

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--model_path', required=True, help='Path to the trained model')
@click.option('--data_path', required=True, help='Path to dataset for explanation')
@click.option('--explanation_types', default='feature_importance,partial_dependence,shap_values',
              help='Comma-separated list of explanation types')
@click.option('--output_path', default='explanation_report', help='Output path for explanation report')
@click.option('--create_dashboard', is_flag=True, help='Create visual explanation dashboard')
@click.option('--dashboard_path', help='Path to save dashboard image')
@click.option('--interactive_dashboard', is_flag=True, help='Launch interactive dashboard')
@click.option('--port', default=8050, help='Port for interactive dashboard')
def explain_model(model_path, data_path, explanation_types, output_path, create_dashboard, 
                  dashboard_path, interactive_dashboard, port):
    """
    Generate comprehensive model explanations and create explainability dashboard.
    
    Example:
        igel explain-model --model_path=model.joblib --data_path=data.csv --create_dashboard
    """
    try:
        from igel.explainability import ModelExplainer, ExplainabilityDashboard
        import pandas as pd
        import joblib
        
        # Load model
        model = joblib.load(model_path)
        
        # Load data
        df = pd.read_csv(data_path)
        target_col = 'target'
        if target_col in df.columns:
            X = df.drop(columns=[target_col]).values
            y = df[target_col].values
        else:
            X = df.values
            y = None
        
        # Get feature names
        feature_names = list(df.columns) if target_col in df.columns else [f'feature_{i}' for i in range(df.shape[1])]
        
        # Create explainer
        explainer = ModelExplainer(model, feature_names)
        
        # Parse explanation types
        exp_types = [exp.strip() for exp in explanation_types.split(',')]
        
        # Generate explanations
        print("Generating model explanations...")
        explanations = explainer.explain_model(X, y, explanation_types=exp_types)
        
        # Generate and print report
        report = explainer.generate_explanation_report()
        print("\n" + report + "\n")
        
        # Save explanations
        explainer.save_explanations(output_path)
        print(f"Explanations saved to {output_path}_explanations.json")
        
        # Create static dashboard
        if create_dashboard:
            dashboard_save_path = dashboard_path or f"{output_path}_dashboard.png"
            explainer.create_explanation_dashboard(X, y, save_path=dashboard_save_path)
            print(f"Dashboard saved to {dashboard_save_path}")
        
        # Launch interactive dashboard
        if interactive_dashboard:
            print(f"Launching interactive dashboard on port {port}...")
            dashboard = ExplainabilityDashboard(explainer)
            dashboard.launch_dashboard(X, y, port=port)
        
    except Exception as e:
        logger.exception(f"Error explaining model: {e}")
        raise click.ClickException(str(e))

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--data_paths', required=True, help='Comma-separated paths to client datasets')
@click.option('--problem_type', default='classification',
              type=click.Choice(['classification', 'regression']),
              help='Type of problem to solve')
@click.option('--model_type', default='logistic_regression',
              type=click.Choice(['logistic_regression', 'linear_regression', 'random_forest']),
              help='Type of model to use')
@click.option('--aggregation_method', default='fedavg',
              type=click.Choice(['fedavg', 'fedprox']),
              help='Federated aggregation method')
@click.option('--num_rounds', default=10, help='Number of federated training rounds')
@click.option('--epochs_per_round', default=1, help='Number of local epochs per round')
@click.option('--client_fraction', default=1.0, type=float, help='Fraction of clients per round')
@click.option('--output_path', default='federated_model', help='Output path for federated model')
def federated_train(data_paths, problem_type, model_type, aggregation_method, 
                   num_rounds, epochs_per_round, client_fraction, output_path):
    """
    Train a model using federated learning across multiple clients.
    
    Example:
        igel federated-train --data_paths=client1.csv,client2.csv,client3.csv --num_rounds=20
    """
    try:
        from igel.federated_learning import FederatedLearningManager
        import pandas as pd
        import numpy as np
        
        # Parse data paths
        data_path_list = [path.strip() for path in data_paths.split(',')]
        
        # Create federated learning manager
        fl_manager = FederatedLearningManager(problem_type=problem_type)
        
        # Create global model
        if problem_type == "classification":
            if model_type == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                global_model = RandomForestClassifier(n_estimators=10, random_state=42)
            else:
                from sklearn.linear_model import LogisticRegression
                global_model = LogisticRegression(random_state=42)
        else:
            if model_type == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                global_model = RandomForestRegressor(n_estimators=10, random_state=42)
            else:
                from sklearn.linear_model import LinearRegression
                global_model = LinearRegression()
        
        # Create federation
        fl_manager.create_federation(global_model, aggregation_method)
        
        # Add client data
        print(f"Loading data from {len(data_path_list)} clients...")
        for i, data_path in enumerate(data_path_list):
            df = pd.read_csv(data_path)
            target_col = 'target'
            
            if target_col in df.columns:
                X = df.drop(columns=[target_col]).values
                y = df[target_col].values
            else:
                # Assume last column is target
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            
            client_id = f"client_{i+1}"
            fl_manager.add_client_data(client_id, X, y, model_type)
            print(f"Added {client_id} with {len(X)} samples")
        
        # Run federated training
        print(f"Starting federated training with {num_rounds} rounds...")
        training_results = fl_manager.run_federated_training(
            num_rounds=num_rounds,
            epochs_per_round=epochs_per_round,
            client_fraction=client_fraction
        )
        
        # Generate and print report
        if fl_manager.server:
            report = fl_manager.server.get_training_report()
            print("\n" + report + "\n")
            
            # Save federated model
            fl_manager.server.save_federated_model(output_path)
            print(f"Federated model saved to {output_path}")
        
        # Print final metrics
        if 'final_metrics' in training_results:
            print("Final Model Performance:")
            for metric, value in training_results['final_metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        logger.exception(f"Error in federated training: {e}")
        raise click.ClickException(str(e))

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--federated_model_path', required=True, help='Path to saved federated model')
@click.option('--test_data', required=True, help='Path to test dataset')
@click.option('--output_predictions', help='Path to save predictions')
def federated_predict(federated_model_path, test_data, output_predictions):
    """
    Make predictions using a federated model.
    
    Example:
        igel federated-predict --federated_model_path=federated_model --test_data=test.csv
    """
    try:
        import joblib
        import pandas as pd
        import numpy as np
        
        # Load federated model
        global_model = joblib.load(f"{federated_model_path}_global_model.joblib")
        
        # Load test data
        test_df = pd.read_csv(test_data)
        target_col = 'target'
        
        if target_col in test_df.columns:
            X_test = test_df.drop(columns=[target_col]).values
            y_true = test_df[target_col].values
        else:
            X_test = test_df.values
            y_true = None
        
        # Make predictions
        predictions = global_model.predict(X_test)
        
        print(f"Federated model predictions completed. Shape: {predictions.shape}")
        
        # Calculate metrics if true labels available
        if y_true is not None:
            from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
            
            if hasattr(global_model, 'predict_proba'):
                accuracy = accuracy_score(y_true, predictions)
                print(f"Accuracy: {accuracy:.4f}")
            else:
                mse = mean_squared_error(y_true, predictions)
                r2 = r2_score(y_true, predictions)
                print(f"MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # Save predictions if requested
        if output_predictions:
            pred_df = pd.DataFrame({'predictions': predictions})
            pred_df.to_csv(output_predictions, index=False)
            print(f"Predictions saved to {output_predictions}")
        
    except Exception as e:
        logger.exception(f"Error making federated predictions: {e}")
        raise click.ClickException(str(e))

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--model_paths', required=True, help='Comma-separated paths to model files')
@click.option('--test_data', required=True, help='Path to test dataset')
@click.option('--problem_type', default='classification',
              type=click.Choice(['classification', 'regression']),
              help='Type of problem')
@click.option('--leaderboard_name', default='default', help='Name of the leaderboard')
@click.option('--output_path', default='leaderboard', help='Output path for leaderboard')
@click.option('--top_n', default=10, help='Number of top models to show')
def create_leaderboard(model_paths, test_data, problem_type, leaderboard_name, output_path, top_n):
    """
    Create a model comparison leaderboard.
    
    Example:
        igel create-leaderboard --model_paths=model1.joblib,model2.joblib --test_data=test.csv
    """
    try:
        from igel.model_leaderboard import ModelLeaderboard
        import pandas as pd
        import numpy as np
        
        # Parse model paths
        model_path_list = [path.strip() for path in model_paths.split(',')]
        
        # Load test data
        test_df = pd.read_csv(test_data)
        target_col = 'target'
        
        if target_col in test_df.columns:
            X_test = test_df.drop(columns=[target_col]).values
            y_test = test_df[target_col].values
        else:
            X_test = test_df.iloc[:, :-1].values
            y_test = test_df.iloc[:, -1].values
        
        # Create leaderboard
        leaderboard = ModelLeaderboard(leaderboard_name)
        
        # Add models to leaderboard
        print(f"Adding {len(model_path_list)} models to leaderboard...")
        for i, model_path in enumerate(model_path_list):
            model_id = f"model_{i+1}"
            model_name = Path(model_path).stem
            leaderboard.add_model(model_id, model_path, model_name)
        
        # Evaluate all models
        print("Evaluating models...")
        for model_id in leaderboard.models.keys():
            metrics = leaderboard.evaluate_model(model_id, X_test, y_test, problem_type)
            print(f"Model {model_id}: {metrics}")
        
        # Rank models
        rankings = leaderboard.rank_models()
        
        # Generate and print report
        report = leaderboard.generate_leaderboard_report()
        print("\n" + report + "\n")
        
        # Show top models table
        df = leaderboard.get_leaderboard_table(top_n=top_n)
        if not df.empty:
            print("Top Models:")
            print(df.to_string(index=False))
        
        # Save leaderboard
        leaderboard.save_leaderboard(output_path)
        print(f"Leaderboard saved to {output_path}_leaderboard.json")
        
    except Exception as e:
        logger.exception(f"Error creating leaderboard: {e}")
        raise click.ClickException(str(e))

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--leaderboard_path', required=True, help='Path to saved leaderboard')
@click.option('--top_n', default=10, help='Number of top models to show')
def show_leaderboard(leaderboard_path, top_n):
    """
    Display a saved leaderboard.
    
    Example:
        igel show-leaderboard --leaderboard_path=leaderboard --top_n=5
    """
    try:
        from igel.model_leaderboard import ModelLeaderboard
        
        # Load leaderboard
        leaderboard = ModelLeaderboard.load_leaderboard(leaderboard_path)
        
        # Generate and print report
        report = leaderboard.generate_leaderboard_report()
        print("\n" + report + "\n")
        
        # Show top models table
        df = leaderboard.get_leaderboard_table(top_n=top_n)
        if not df.empty:
            print("Top Models:")
            print(df.to_string(index=False))
        else:
            print("No models in leaderboard")
        
    except Exception as e:
        logger.exception(f"Error displaying leaderboard: {e}")
        raise click.ClickException(str(e))

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--data_path', required=True, help='Path to time series data')
@click.option('--method', default='isolation_forest',
              type=click.Choice(['isolation_forest', 'dbscan', 'statistical', 'lstm_autoencoder']),
              help='Anomaly detection method')
@click.option('--contamination', default=0.1, type=float, help='Expected proportion of anomalies')
@click.option('--output_path', default='anomaly_results', help='Output path for results')
@click.option('--true_labels', help='Path to true anomaly labels for evaluation')
def detect_anomalies(data_path, method, contamination, output_path, true_labels):
    """
    Detect anomalies in time series data.
    
    Example:
        igel detect-anomalies --data_path=timeseries.csv --method=isolation_forest
    """
    try:
        from igel.time_series_anomaly import TimeSeriesAnomalyDetector
        import pandas as pd
        import numpy as np
        
        # Load time series data
        df = pd.read_csv(data_path)
        
        # Assume first column is time, rest are features
        if 'time' in df.columns:
            time_col = 'time'
            feature_cols = [col for col in df.columns if col != 'time']
        else:
            time_col = df.columns[0]
            feature_cols = df.columns[1:]
        
        # Extract features
        X = df[feature_cols].values
        
        # Create anomaly detector
        detector = TimeSeriesAnomalyDetector(method=method)
        
        # Fit detector
        print(f"Fitting {method} anomaly detector...")
        detector.fit(X, contamination=contamination)
        
        # Detect anomalies
        print("Detecting anomalies...")
        results = detector.detect_anomalies(X, return_scores=True)
        
        # Generate and print report
        report = detector.generate_report()
        print("\n" + report + "\n")
        
        # Evaluate if true labels provided
        if true_labels:
            true_df = pd.read_csv(true_labels)
            if 'anomaly' in true_df.columns:
                y_true = true_df['anomaly'].values
                evaluation = detector.evaluate_detection(y_true)
                print("Evaluation Results:")
                for metric, value in evaluation.items():
                    print(f"  {metric}: {value:.4f}")
        
        # Save results
        detector.save_results(output_path)
        print(f"Anomaly detection results saved to {output_path}")
        
    except Exception as e:
        logger.exception(f"Error detecting anomalies: {e}")
        raise click.ClickException(str(e))

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--data_path', required=True, help='Path to dataset')
@click.option('--n_qubits', default=4, help='Number of qubits for quantum circuit')
@click.option('--backend', default='simulator', help='Quantum backend')
def quantum_ml(data_path, n_qubits, backend):
    """
    Basic quantum machine learning demonstration.
    
    Example:
        igel quantum-ml --data_path=data.csv --n_qubits=4
    """
    try:
        from igel.quantum_ml import QuantumML
        import pandas as pd
        
        # Load data
        df = pd.read_csv(data_path)
        X = df.values
        
        # Initialize quantum ML
        qml = QuantumML(backend=backend)
        qml.initialize_quantum_circuit(n_qubits)
        
        # Apply quantum feature map
        quantum_features = qml.quantum_feature_map(X)
        
        # Get circuit info
        circuit_info = qml.get_quantum_circuit_info()
        
        print("Quantum ML Results:")
        print(f"Quantum Circuit Info: {circuit_info}")
        print(f"Quantum Features Shape: {quantum_features.shape}")
        
    except Exception as e:
        logger.exception(f"Error in quantum ML: {e}")
        raise click.ClickException(str(e))

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--model_path', required=True, help='Path to trained model')
@click.option('--output_path', required=True, help='Path to save PMML file')
@click.option('--model_name', default='Model', help='Name for the model in PMML')
def export_pmml(model_path, output_path, model_name):
    """
    Export a trained model to PMML format.
    
    Example:
        igel export-pmml --model_path=model.joblib --output_path=model.pmml
    """
    try:
        from igel.pmml_export import PMMLExporter
        
        # Load model
        model = joblib.load(model_path)
        
        # Create exporter
        exporter = PMMLExporter()
        
        # Check if model is supported
        if not exporter.is_model_supported(model):
            print(f"Model type {type(model).__name__} not supported for PMML export")
            print(f"Supported models: {exporter.get_supported_models()}")
            return
        
        # Export to PMML
        success = exporter.export_to_pmml(model, output_path, model_name)
        
        if success:
            print(f"Model successfully exported to PMML: {output_path}")
        else:
            print("PMML export failed. Check logs for details.")
        
    except Exception as e:
        logger.exception(f"Error exporting to PMML: {e}")
        raise click.ClickException(str(e))

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--data_path', required=True, help='Path to dataset')
@click.option('--output_path', help='Path to save optimized data')
@click.option('--chunk_size', default=10000, help='Chunk size for large datasets')
def optimize_memory(data_path, output_path, chunk_size):
    """
    Optimize memory usage for large datasets.
    
    Example:
        igel optimize-memory --data_path=large_dataset.csv --output_path=optimized_data.csv
    """
    try:
        from igel.memory_optimizer import MemoryOptimizer
        import pandas as pd
        
        # Load data
        print("Loading data...")
        df = pd.read_csv(data_path)
        
        # Create optimizer
        optimizer = MemoryOptimizer()
        
        # Get initial memory usage
        initial_memory = optimizer.get_memory_usage()
        print(f"Initial memory usage: {initial_memory['rss_mb']:.2f} MB")
        
        # Optimize DataFrame
        print("Optimizing DataFrame...")
        optimized_df = optimizer.optimize_dataframe(df)
        
        # Generate report
        report = optimizer.get_optimization_report()
        print("\n" + report)
        
        # Save optimized data if output path provided
        if output_path:
            optimized_df.to_csv(output_path, index=False)
            print(f"Optimized data saved to: {output_path}")
        
        # Clear memory
        optimizer.clear_memory()
        
    except Exception as e:
        logger.exception(f"Error optimizing memory: {e}")
        raise click.ClickException(str(e))

@cli.command(context_settings=CONTEXT_SETTINGS)
@click.option('--source_model', required=True, help='Path to pre-trained source model')
@click.option('--target_data', required=True, help='Path to target data')
@click.option('--method', default='feature_extraction',
              type=click.Choice(['feature_extraction', 'fine_tuning']),
              help='Transfer learning method')
@click.option('--output_model', default='transfer_model.joblib', help='Output model path')
def transfer_learn(source_model, target_data, method, output_model):
    """
    Perform transfer learning using a pre-trained model.
    
    Example:
        igel transfer-learn --source_model=pretrained.joblib --target_data=new_data.csv
    """
    try:
        from igel.few_shot_learning import TransferLearning
        import pandas as pd
        import joblib
        
        # Load pre-trained model
        source_model = joblib.load(source_model)
        
        # Load target data
        target_df = pd.read_csv(target_data)
        target_X = target_df.drop(columns=['target']).values
        target_y = target_df['target'].values
        
        # Perform transfer learning
        transfer = TransferLearning(source_model)
        transfer_model = transfer.create_transfer_model(source_model, target_X, target_y, method)
        
        # Save transfer model
        joblib.dump(transfer_model, output_model)
        print(f"Transfer learning model saved to {output_model}")
        
    except Exception as e:
        logger.exception(f"Error during transfer learning: {e}")
        raise click.ClickException(str(e))
