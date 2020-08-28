from igel import read_yaml
from sklearn.linear_model import LinearRegression
import pandas as pd

config = read_yaml('./model.yaml')
print(config)

model_type = config['model']['type']
print("model: ", model_type)

data_path = config['input_features']['path']
data = pd.read_csv(data_path)
print("data shape: ", data.shape)

if model_type == "linear_regression":
    model = LinearRegression()
