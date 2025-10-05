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
