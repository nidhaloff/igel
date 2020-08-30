from igel import IgelModel


"""
In Terminal:

for fitting a model:
--------------------

igel fit --data_path 'examples/data/example.csv' --model_definition_file 'examples/model.yaml'

for using a pre-fitted model for prediction:
---------------------------------------------
igel predict --data_path 'examples/data/test_example.csv'

"""
fit_config_yaml = {'data_path': 'examples/data/example.csv',
                   'model_definition_file': 'examples/model.yaml'}

predict_config_yaml = {
    'data_path': 'examples/data/test_example.csv'
}

reg = IgelModel('fit', **fit_config_yaml)
reg.fit()
