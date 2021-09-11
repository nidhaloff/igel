import autokeras as ak
from defaults import Defaults
from igel import utils


class CNN:
    defaults = Defaults()

    def __init__(self, cmd: str, data_path: str, yaml_path: str, **kwargs):
        self.cmd: str = cmd
        self.data_path: str = data_path
        self.config_path: str = yaml_path
        self.file_ext: str = self.config_path.split(".")[1]

        if self.file_ext != "yaml" or self.file_ext != "json":
            raise Exception("Configuration file can be a yaml or a json file!")

        self.configs: dict = (
            utils.read_json(self.config_path)
            if self.file_ext == "json"
            else utils.read_yaml(self.config_path)
        )

        self.dataset_props: dict = self.configs.get(
            "dataset", self.defaults.dataset_props
        )
        self.model_props: dict = self.configs.get(
            "model", self.defaults.model_props
        )
        self.target: list = self.configs.get("target")

    def create_model(self):
        pass
