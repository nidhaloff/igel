from igel import read_yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle

config = read_yaml('./model.yaml')
print(config)

model_type = config['model']['type']
print("model: ", model_type)
target = config['target']

data_path = config['input_features']['path']
data = pd.read_csv(data_path)
print("data columns: ", data.columns)
print("data shape: ", data.shape)

y = data.pop(target).to_numpy().reshape(-1, 1)
print("y shape : ", y.shape)
print("data shape: ", data.shape)

if model_type == "linear_regression":
    print("model type is: ", model_type)
    model = LinearRegression()
else:
    model = LinearRegression()

model.fit(data, y)
print("coefficients: ", model.coef_)

preds = model.predict(data)
print("MSE: ", mean_squared_error(y, preds))
print(preds[:10], y[:10])
