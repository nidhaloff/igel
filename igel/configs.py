from pathlib import Path
import os


stats_dir = 'model_results'
model_file = 'model.sav'
res_path = Path(os.getcwd()) / stats_dir
description_file = 'description.json'
evaluation_file = 'evaluation.json'
predictions_file = 'predictions.csv'

configs = {
    "stats_dir": stats_dir,
    "model_file": model_file,
    "results_path": res_path,
    "default_model_path": res_path / model_file,
    "description_file": res_path / description_file,
    "evaluation_file": res_path / evaluation_file,
    "prediction_file": res_path / predictions_file,

    "dataset_props": {
        "csv_separator": ",",
        "normalize": False
    },
    "model_props": {
        "type": "regression",
        "algorithm": "linear regression"
    }
}
