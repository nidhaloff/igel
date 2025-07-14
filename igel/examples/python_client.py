import json
import requests

from igel.preprocessing import handle_missing_values, read_data_to_df


class IgelClient:
    """ A mini client example that reads some data file and formats
        it for the post call to predict
    """
    def __init__(self, host, port, scheme="http", endpoint="predict", missing_values='mean'):
        self.url = f"{scheme}://{host}:{port}/{endpoint}"
        self.missing_values = missing_values

    def process_data(self, data_file):
        """ Use the inbuilt handle_missing_values since missing values can't be sent in 
            valid JSON without raising an error
        """
        data_df = handle_missing_values(read_data_to_df(data_file), strategy=self.missing_values)

        # make a dict of {header: column values}, JSONify, return        
        data_dict = data_df.to_dict(orient='list', into=dict)
        jsonified = json.dumps(data_dict)
        return jsonified

    def post(self, data_file):
        output_json = self.process_data(data_file)
        res = requests.post(self.url, data=output_json)
        print(f"{res}: {res.text}")
        return res
