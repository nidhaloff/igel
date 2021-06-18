import os
from pathlib import Path

from igel.constants import Constants

res_path = Path(os.getcwd()) / Constants.stats_dir
init_file_path = Path(os.getcwd()) / Constants.init_file
temp_post_req_data_path = Path(os.getcwd()) / Constants.post_req_data_file

configs = {
    "stats_dir": Constants.stats_dir,
    "model_file": Constants.model_file,
    "results_path": res_path,
    "default_model_path": res_path / Constants.model_file,
    "description_file": res_path / Constants.description_file,
    "evaluation_file": res_path / Constants.evaluation_file,
    "prediction_file": res_path / Constants.prediction_file,
    "init_file_path": init_file_path,
    "dataset_props": {
        "type": "csv",
        "split": {"test_size": 0.1, "shuffle": True},
        "preprocess": {
            "missing_values": "mean",
            "scale": {"method": "standard", "target": "inputs"},
        },
    },
    "model_props": {"type": "classification", "algorithm": "NeuralNetwork"},
    "available_dataset_props": {
        "type": "csv",
        "separator": ",",
        "split": {"test_size": None, "shuffle": False, "stratify": None},
        "preprocess": {
            "missing_values": "mean",
            "encoding": None,
            "scale": None,
        },
    },
    "available_model_props": {
        "type": "regression",
        "algorithm": "linear regression",
        "arguments": "default",
    },
}
