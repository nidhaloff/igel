from igel import IgelModel

mock_pred_params = {'data_path': './data/test-indians-diabetes.csv',
                    'cmd': 'predict'}

IgelModel(**mock_pred_params).predict()
