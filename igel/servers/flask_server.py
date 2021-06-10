import pandas as pd
import logging

logger = logging.getLogger(__name__)

try:
    from flask import Flask, jsonify
except ImportError as err:
    logger.fatal(f"Make sure you install flask on your machine. You can do this by running pip install flask\n"
                 f"Error Trace: {err}")

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
     json_ = request.json
     query_df = pd.DataFrame(json_)
     query = pd.get_dummies(query_df)
     prediction = lr.predict(query)
     return jsonify({'prediction': list(prediction)})
