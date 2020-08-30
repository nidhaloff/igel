"""Main module."""

from igel import preprocessing
import pandas as pd
import pickle
import os
from pathlib import Path
from igel.utils import read_yaml


class IgelModel(object):
    def __init__(self, command, **dict_args):
        print("dict args: ", dict_args)
        if 'data_path' not in dict_args.keys():
            raise Exception("you need to provide the path for the data!")
        if 'model_path' not in dict_args.keys():
            raise Exception("you need to provide the path to the yaml file!")

        data_path = dict_args.get('data_path')
        model_path = dict_args.get('model_path')

        self.config = read_yaml(model_path)
        # print("config: ", self.config)

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
    reg = IgelModel(f='/home/nidhal/my_projects/igel/examples/model.yaml')
    reg.fit()
    reg.predict()
