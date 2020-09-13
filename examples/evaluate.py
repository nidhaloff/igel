from igel import IgelModel

mock_eval_params = {'data_path': './data/indians-diabetes.csv',
                    'cmd': 'evaluate'}

IgelModel(**mock_eval_params).evaluate()
