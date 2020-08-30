import yaml


def read_yaml(f):
    with open(f, 'r') as stream:
        try:
            res = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        else:
            return res


def extract_params(config):
    assert 'model' in config.keys(), "model parameters need to be provided in the yaml file"
    assert 'target' in config.keys(), "target variable needs to be provided in the yaml file"
    model_params = config.get('model')
    model_type = model_params.get('type')
    algorithm = model_params.get('algorithm')
    target = config.get("target")

    if any(not item for item in [model_type, target, algorithm]):
        raise Exception("parameters in the model yaml file cannot be None")
    else:
        return model_type, target, algorithm


def _reshape(arr):
    if len(arr.shape) <= 1:
        arr = arr.reshape(-1, 1)
    return arr
