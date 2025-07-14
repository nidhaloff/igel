from igel import Igel

"""
The goal of igel is to use ML without writing code. Therefore, the right and simplest way to use igel is from terminal.
You can run ` igel fit -dp path_to_dataset -yml path_to_yaml_file`.

Alternatively, you can write code if you want. This example below demonstrates how to use igel if you want to write code.
However, I suggest you try and use the igel CLI. Type igel -h in your terminal to know more.

"""
mock_fit_params = {'data_path': '../data/iris/train-Iris.csv',
                   'yaml_path': './iris.yaml',
                   'cmd': 'fit'}

Igel(**mock_fit_params)

