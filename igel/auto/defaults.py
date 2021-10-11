from igel.configs import configs


class Defaults:
    dataset_props = {}
    model_props = {}
    training_args = {}
    available_commands = ("fit", "evaluate", "predict", "experiment")
    supported_types = ("regression", "classification", "clustering")
    results_path = configs.get("results_path")  # path to the results folder
    model_path = configs.get(
        "default_model_path"
    )  # path to the pre-fitted model
    description_file = configs.get(
        "description_file"
    )  # path to the description.json file
    evaluation_file = configs.get(
        "evaluation_file"
    )  # path to the evaluation.json file
    prediction_file = configs.get(
        "prediction_file"
    )  # path to the predictions.csv
