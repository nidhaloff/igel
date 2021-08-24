"""Console script for igel."""
import click
import logging
import os
import subprocess
import sys
from pathlib import Path

import igel
import pandas as pd
from igel import Igel, metrics_dict, models_dict
from igel.constants import Constants
from igel.servers import fastapi_server

logger = logging.getLogger(__name__)

def validate_path (path) :
    if not os.path.isdir (path) :
        logger.warning (f"Unrecognized Path -> {path}")
        exit (1)

def validate_model_type (model_type) :
    if not model_type in ["regression", "classification", "cliustering"] :
        logger.warning (f"Unrecognized Model Type -> {model_type}")
        exit (1)

available_args = {
    # fit, evaluate and predict args:
    "data_path": validate_path,
    "yaml_path": validate_path,
    "data_paths": validate_path,
    "model_results_dir": None,
    # models arguments
    "model_name": None,
    "model_type": validate_model_type,
    "target": None,
    # host and port for serving the model
    "host": None,
    "port": None,
}

def validate_arguments (arguments) :
    for k, v in arguments.items () :
        if available_args[k] is not None :
            available_args[k] (v)

@click.group()
def main () :
    pass

@click.command ()
@click.option ('-dp', '--data_path', required=True, help='Path to your dataset (dp stand for data_path, you can use --data_path instead)')
@click.option ('-yml', '--yaml_path', required=True, help='Path to your yaml file (you can use --yaml_path instead)')
@click.option ('-res_dir', '--model_results_dir', default='model_results', show_default=True, help='Path to the model_results directory generated by igel after training the model')
@click.option ('-type', '--model_type', required=True, help='type of the model you want to get help on whether regression, classification or clustering. (you can use --model_type instead)')
@click.option ('-name', '--model_name', default='model', show_default=True, help='name of the model you want to get help on.    (you can use --model_name instead)')
def fit (data_path, yaml_path, model_results_dir, model_type, model_name) :
    arguments = {
        'data_path':data_path,
        'yaml_path':yaml_path,
        'model_results_dir':model_results_dir,
        'model_name':model_name,
        'model_type':model_type,
        'command':'fit'
    }

    validate_arguments (arguments)

    print(
        r"""
        _____          _       _
    |_   _| __ __ _(_)_ __ (_)_ __   __ _
        | || '__/ _` | | '_ \| | '_ \ / _` |
        | || | | (_| | | | | | | | | | (_| |
        |_||_|  \__,_|_|_| |_|_|_| |_|\__, |
                                    |___/
    """
    )
    Igel(**arguments)

@click.command ()
@click.option ('-dp', '--data_path', required=True, help='Path to your dataset (dp stand for data_path, you can use --data_path instead)')
@click.option ('-yml', '--yaml_path', required=True, help='Path to your yaml file (you can use --yaml_path instead)')
@click.option ('-res_dir', default='model_results', show_default=True, help='Path to the model_results directory generated by igel after training the model')
@click.option ('-type', '--model_type', required=True, help='type of the model you want to get help on whether regression, classification or clustering. (you can use --model_type instead)')
@click.option ('--name', '--model_name', default='model', show_default=True, help='name of the model you want to get help on.    (you can use --model_name instead)')
def predict (data_path, yaml_path, res_dir, model_type, model_name) :
    arguments = {
        'data_path':data_path,
        'yaml_path':yaml_path,
        'model_results_dir':res_dir,
        'model_name':model_name,
        'model_type':model_type,
        'command':'prediction'
    }

    validate_arguments (arguments)

    print(
        """
        ____               _ _      _   _
    |  _ \\ _ __ ___  __| (_) ___| |_(_) ___  _ __
    | |_) | '__/ _ \\/ _` | |/ __| __| |/ _ \\| '_ \
    |  __/| | |  __/ (_| | | (__| |_| | (_) | | | |
    |_|   |_|  \\___|\\__,_|_|\\___|\\__|_|\\___/|_| |_|


    """
    )
    Igel(**arguments)

@click.command ()
@click.option ('-dp', '--data_path', required=True, help='Path to your dataset (dp stand for data_path, you can use --data_path instead)')
@click.option ('-yml', '--yaml_path', required=True, help='Path to your yaml file (you can use --yaml_path instead)')
@click.option ('-res_dir', default='model_results', show_default=True, help='Path to the model_results directory generated by igel after training the model')
@click.option ('-type', '--model_type', required=True, help='type of the model you want to get help on whether regression, classification or clustering. (you can use --model_type instead)')
@click.option ('--name', '--model_name', default='model', show_default=True, help='name of the model you want to get help on.    (you can use --model_name instead)')
def evaluate (data_path, yaml_path, res_dir, model_type, model_name) :
    arguments = {
        'data_path':data_path,
        'yaml_path':yaml_path,
        'model_results_dir':res_dir,
        'model_name':model_name,
        'model_type':model_type,
        'command':'evaluate'
    }

    validate_arguments (arguments)

    print(
        """
        _____            _             _   _
    | ____|_   ____ _| |_   _  __ _| |_(_) ___  _ __
    |  _| \\ \\ / / _` | | | | |/ _` | __| |/ _ \\| '_ \
    | |___ \\ V / (_| | | |_| | (_| | |_| | (_) | | | |
    |_____| \\_/ \\__,_|_|\\__,_|\\__,_|\\__|_|\\___/|_| |_|

    """
    )
    Igel(**arguments)

main.add_command (fit)
main.add_command (predict)
main.add_command (evaluate)

if __name__ == '__main__' :
    main ()
