from igel import Igel
import yaml
import json


def read_yaml(f):
    with open(f, 'r') as stream:
        try:
            res = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        else:
            return res


# Igel.create_init_mock_file()
f = read_yaml('./igel.yaml')
print(json.dumps(f, sort_keys=False, indent=4))
