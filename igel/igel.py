"""Main module."""

from igel import preprocessing
import pandas as pd
import pickle
import os
from pathlib import Path
from igel.utils import models_map, read_yaml


class Igel(object):
    def __init__(self, f):
        config = read_yaml(f)
        print("config: ", config)
        data_path, model_type, target = self.extract_params(config)
        self.dataset = pd.read_csv(data_path)
        self.model_type = model_type
        print("data columns: ", self.dataset.columns)
        print("data shape: ", self.dataset.shape)

        y = self.dataset.pop(target).to_numpy()
        self.y = y.reshape(-1, 1) if len(y.shape) == 1 else y
        self.X = self.dataset.to_numpy()
        self.stats_dir = Path(os.getcwd()) / 'train_results'
        self.save_to = Path(self.stats_dir) / 'model.sav'

    def extract_params(self, config):
        model_type = config['model']['type']
        target = config['target']
        data_path = config['input_features']['path']
        return data_path, model_type, target

    def save_model(self, model):
        try:
            os.mkdir(self.stats_dir)
        except OSError:
            print("Creation of the directory %s failed" % self.stats_dir)
        else:
            print("Successfully created the directory %s " % self.stats_dir)
            pickle.dump(model, open(self.save_to, 'wb'))
            return True

    def load_model(self):
        try:
            return pickle.load(open(self.save_to, 'rb'))
        except FileNotFoundError:
            print("File Not Found")

    def fit(self):
        model = self.create_model(self.model_type)
        if not model:
            raise Exception("model is not defined")

        model.fit(self.X, self.y)
        self.save_model(model)

    def predict(self):
        model = self.load_model()
        print(self.y.shape)
        y_pred = model.predict(self.y)
        return y_pred


if __name__ == '__main__':
    reg = Igel(f='/home/nidhal/my_projects/igel/examples/model.yaml')
    reg.fit()
    reg.predict()
