from igel import Igel


"""
The goal of igel is to use ML without writing code. Therefore, the right and simplest way to use igel is from terminal.
You can run ` igel evaluate -dp path_to_dataset`.

Alternatively, you can write code if you want. This example below demonstrates how to use igel if you want to write code.
However, I suggest you try and use the igel CLI. Type igel -h in your terminal to know more.

"""

mock_eval_params = {'data_path': '../data/iris/eval-Iris.csv',
                    'cmd': 'evaluate'}

Igel(**mock_eval_params)
