import json
import logging
import os

import autokeras as ak
from igel.auto.defaults import Defaults
from igel.auto.models import Models
from igel.constants import Constants
from igel.utils import read_json, read_yaml
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)


class IgelCNN:
    defaults = Defaults()
    model = None
    dataset_props = {}
    model_props = {}
    model_args = {}
    training_args = {}
    results_path = Constants.results_dir

    def __init__(self, **cli_args):
        self.cmd: str = cli_args.get("cmd")
        self.data_path: str = cli_args.get("data_path")
        self.config_path: str = cli_args.get("yaml_path")
        self.task = cli_args.get("task")
        logger.info(f"Executing command: {self.cmd}")
        logger.info(f"Reading data from: {self.data_path}")
        logger.info(f"Reading yaml configs from: {self.config_path}")

        if self.cmd == "train":
            if not self.config_path:
                self.model_type = self.task
            else:
                self.file_ext: str = self.config_path.split(".")[1]

                if self.file_ext not in ["yaml", "yml", "json"]:
                    raise Exception(
                        "Configuration file can be a yaml or a json file!"
                    )

                self.configs: dict = (
                    read_json(self.config_path)
                    if self.file_ext == "json"
                    else read_yaml(self.config_path)
                )

                self.dataset_props: dict = self.configs.get(
                    "dataset", self.defaults.dataset_props
                )
                self.model_props: dict = self.configs.get(
                    "model", self.defaults.model_props
                )
                self.training_args: dict = self.configs.get(
                    "training", self.defaults.training_args
                )
                self.model_args = self.model_props.get("arguments")
                self.model_type = self.task or self.model_props.get("type")

        else:
            self.model_path = cli_args.get(
                "model_path", self.defaults.model_path
            )
            logger.info(f"path of the pre-fitted model => {self.model_path}")
            self.prediction_file = cli_args.get(
                "prediction_file", self.defaults.prediction_file
            )
            # set description.json if provided:
            self.description_file = cli_args.get(
                "description_file", self.defaults.description_file
            )
            # load description file to read stored training parameters
            with open(self.description_file) as f:
                dic = json.load(f)
                self.model_type: str = dic.get("task")  # type of the model
                self.dataset_props: dict = dic.get(
                    "dataset_props"
                )  # dataset props entered while fitting
        getattr(self, self.cmd)()

    def _create_model(self, *args, **kwargs):
        model_cls = Models.get(self.model_type)
        model = (
            model_cls() if not self.model_args else model_cls(**self.model_args)
        )
        return model

    def save_desc_file(self):
        desc = {
            "task": self.model_type,
            "model": self.model.__class__.__name__,
            "dataset_props": self.dataset_props or None,
            "model_props": self.model_props or None,
        }
        try:
            logger.info(f"saving fit description to {self.description_file}")
            with open(self.description_file, "w", encoding="utf-8") as f:
                json.dump(desc, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.exception(
                f"Error while storing the fit description file: {e}"
            )

    def save_model(self):
        exp_model = self.model.export_model()
        logger.info(f"model type: {type(exp_model)}")
        try:
            exp_model.save("model", save_format="tf")
            return True
        except Exception:
            exp_model.save(f"model.h5")

    def train(self):
        train_data = ak.image_dataset_from_directory(self.data_path)
        self.model = self._create_model()
        logger.info(f"executing a {self.model.__class__.__name__} algorithm...")
        logger.info(f"Training started...")
        self.model.fit(train_data, **self.training_args)
        logger.info("finished training!")
        self.save_desc_file()
        saved = self.save_model()
        if saved:
            logger.info(f"model saved successfully")

    def load_model(self):
        logger.info("loading model...")
        loaded_model = load_model("model", custom_objects=ak.CUSTOM_OBJECTS)
        logger.info("model loaded successfully")
        return loaded_model

    def evaluate(self):
        trained_model = self.load_model()
        test_data = ak.image_dataset_from_directory(self.data_path)
        trained_model.evaluate(test_data)

    def predict(self):
        trained_model = self.load_model()
        pred_data = ak.image_dataset_from_directory(self.data_path)
        trained_model.predict(pred_data)
