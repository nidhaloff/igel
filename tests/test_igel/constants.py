import os
from pathlib import Path


class Constants:
    train_data_file = "train_data.csv"
    eval_data_file = "eval_data.csv"
    test_data_file = "new_data.csv"
    igel_yaml_file = "igel.yaml"
    model_results_file = "model_results"
    model_results_dir = Path(os.getcwd()) / model_results_file
    description_file = model_results_dir / "description.json"
    evaluation_file = model_results_dir / "evaluation.json"
    prediction_file = model_results_dir / "prediction.csv"

    data_dir = Path(os.getcwd()) / "data"
    igel_files = Path(os.getcwd()) / "igel_files"
    train_data = data_dir / train_data_file
    eval_data = data_dir / eval_data_file
    test_data = data_dir / test_data_file
    yaml_file = igel_files / igel_yaml_file
