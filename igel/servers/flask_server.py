import pandas as pd
import logging
from igel.utils import load_trained_model

logger = logging.getLogger(__name__)

try:
    from flask import Flask, jsonify, request
except ImportError as err:
    logger.fatal(f"Make sure you install flask on your machine. You can do this by running pip install flask\n"
                 f"Traceback:\n{err}")

app = Flask(__name__)
model = load_trained_model()


@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_)
     query = pd.get_dummies(query_df)
     prediction = model.predict(query)
     return jsonify({'prediction': list(prediction)})
