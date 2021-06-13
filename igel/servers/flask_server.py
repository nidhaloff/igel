import logging

import numpy as np
from igel.utils import load_trained_model

logger = logging.getLogger(__name__)

try:
    from flask import Flask, jsonify, request
except ImportError as err:
    logger.fatal(
        f"Make sure you install flask on your machine. You can do this by running pip install flask\n"
        f"Traceback:\n{err}"
    )

app = Flask(__name__)
model = load_trained_model()


@app.route("/")
def home():
    return {"success": True}


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    print(f"data: {data} | type: {type(data)}")
    x_pred = np.array(list(data.values()))
    print(f"x_pred: {x_pred.shape}")
    x_pred = x_pred.reshape(1, -1)
    print(f"x_pred after reshape: {x_pred.shape}")
    prediction = model.predict(x_pred)
    print("prediction: ", prediction)
    return {"prediction": prediction.tolist()}


def run(*args, **kwargs):
    app.run(debug=False)
