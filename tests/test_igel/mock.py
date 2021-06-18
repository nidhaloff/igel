import os
from pathlib import Path

from .constants import Constants


class MockCliArgs:
    """mock class to provide mock cli args"""

    fit = {
        "cmd": "fit",
        "data_path": Constants.train_data,
        "yaml_path": Constants.yaml_file,
    }
    evaluate = {"cmd": "evaluate", "data_path": Constants.eval_data}
    predict = {"cmd": "predict", "data_path": Constants.test_data}
    experiment = {
        "cmd": "experiment",
        "data_paths": f"{Constants.train_data} {Constants.eval_data} {Constants.test_data}",
    }
