import os
from pathlib import Path

stats_dir = "model_results"
model_file = "model.joblib"
init_file = "igel.yaml"
post_req_data_file = "post_req_data.csv"
res_path = Path(os.getcwd()) / stats_dir
init_file_path = Path(os.getcwd()) / init_file
temp_post_req_data_path = Path(os.getcwd()) / post_req_data_file
description_file = "description.json"
evaluation_file = "evaluation.json"
prediction_file = "predictions.csv"

configs = {
    "stats_dir": stats_dir,
    "model_file": model_file,
    "results_path": res_path,
    "default_model_path": res_path / model_file,
    "description_file": res_path / description_file,
    "evaluation_file": res_path / evaluation_file,
    "prediction_file": res_path / prediction_file,
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
