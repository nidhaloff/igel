from igel import Igel

"""
The goal of igel is to use ML without writing code. Therefore, the right and simplest way to use igel is from terminal.
You can run ` igel fit -dp path_to_dataset -yml path_to_yaml_file`.

Alternatively, you can write code if you want. This example below demonstrates how to use igel if you want to write code.
However, I suggest you try and use the igel CLI. Type igel -h in your terminal to know more.

===============================================================================================================

This example fits a machine learning model on the indian-diabetes dataset

- default model here is the neural network and the configuration are provided in neural-network.yaml file
- You can switch to random forest by providing the random-forest.yaml as the config file in the parameters

"""

mock_fit_params = {
                   'data_path': '../data/indian-diabetes/train-indians-diabetes.csv',
                   #'data_path': './data.json',
                   'yaml_path': './neural-network.yaml',
                   'cmd': 'fit'}

Igel(**mock_fit_params)

