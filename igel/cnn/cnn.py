import json
import logging
import os

import autokeras as ak
import numpy as np
import pandas as pd
from igel.cnn.defaults import Defaults
from igel.cnn.models import Models
from igel.constants import Constants
from igel.utils import read_json, read_yaml
from tensorflow.keras.preprocessing import image

logger = logging.getLogger(__name__)


class IgelCNN:
    defaults = Defaults()
    x = None
    y = None
    model = None
    results_path = Constants.results_dir

    def __init__(self, **cli_args):
        self.cmd: str = cli_args.get("cmd")
        self.data_path: str = cli_args.get("data_path")
        self.config_path: str = cli_args.get("yaml_path")
        logger.info(f"Executing command: {self.cmd}")
        logger.info(f"Reading data from: {self.data_path}")
        logger.info(f"Reading yaml configs from: {self.config_path}")

        if self.cmd == "train":
            self.file_ext: str = self.config_path.split(".")[1]

            if self.file_ext != "yaml" and self.file_ext != "json":
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
            self.target: list = self.configs.get("target")
            self.model_type = self.model_props.get("type")
            self.model_args = self.model_props.get("arguments")

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
                self.target: list = dic.get(
                    "target"
                )  # target to predict as a list
                self.model_type: str = dic.get("type")  # type of the model
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

    def _convert_img_to_np_array(self, paths):

        images = []
        logger.info(f"Reading images and converting them to arrays...")
        for path in paths:
            img = image.load_img(path, grayscale=True)
            img_arr = np.asarray(img)
            images.append(img_arr)
        return np.array(images)

    def _read_dataset(self):
        # read_data_options = self.dataset_props.get("read_data_options", {})
        # dataset = pd.read_csv(self.data_path, **read_data_options)
        # logger.info(f"dataset shape: {dataset.shape}")
        # attributes = list(dataset.columns)
        # logger.info(f"dataset attributes: {attributes}")
        # y = pd.concat([dataset.pop(x) for x in self.target], axis=1)
        # logger.info(f"x shape: {dataset.shape} | y shape: {y.shape}")
        # x = dataset.to_numpy()
        # num_images = x.shape[0]
        # x = x.reshape((num_images,))
        # self.x = self._convert_img_to_np_array(x)
        # self.y = y.to_numpy()
        # logger.info(
        #     f"After reading images: x shape {self.x.shape} | y shape: {self.y.shape}"
        # )
        train_data = ak.image_dataset_from_directory(
            self.data_path, subset="training", validation_split=0.2, seed=42
        )
        return train_data  # self.x, self.y

    def save_model(self, model):
        exp_model = model.export_model()
        logger.info(f"model type: {type(exp_model)}")
        try:
            exp_model.save("model", save_format="tf")
            return True
        except Exception:
            exp_model.save(f"model.h5")

    def train(self):
        train_data = self._read_dataset()
        self.model = self._create_model()
        logger.info(f"executing a {self.model.__class__.__name__} algorithm...")
        logger.info(f"Training started...")
        self.model.fit(train_data)
        saved = self.save_model(self.model)
        if saved:
            logger.info(f"model saved successfully")
