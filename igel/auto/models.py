import autokeras as ak


class Models:
    models_map = {
        "ImageClassification": {
            "class": ak.ImageClassifier,
            "link": "https://autokeras.com/image_classifier/",
        },
        "ImageRegression": {
            "class": ak.ImageRegressor,
            "link": "https://autokeras.com/image_regressor/",
        },
        "TextClassification": {
            "class": ak.TextClassifier,
            "link": "https://autokeras.com/text_classifier/",
        },
        "TextRegression": {
            "class": ak.TextRegressor,
            "link": "https://autokeras.com/text_regressor/",
        },
        "StructuredDataClassification": {
            "class": ak.StructuredDataClassifier,
            "link": "https://autokeras.com/structured_data_classifier/",
        },
        "StructuredDataRegression": {
            "class": ak.StructuredDataRegressor,
            "link": "https://autokeras.com/structured_data_regressor/",
        },
    }

    @classmethod
    def get(cls, model_type: str, *args, **kwargs):
        if model_type not in cls.models_map.keys():
            raise Exception(
                f"{model_type} is not supported! "
                f"Choose one of the following supported tasks: {cls.models_map.keys()}"
            )
        return cls.models_map[model_type]["class"]
