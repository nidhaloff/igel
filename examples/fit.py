from igel import IgelModel

mock_fit_params = {'data_path': './data/indians-diabetes.csv',
                   'yaml_path': './model.yaml',
                   'cmd': 'fit'}

IgelModel(**mock_fit_params).fit()

