import pandas as pd
import numpy as np
import yaml


def read_yaml(f):
    with open(f, 'r') as stream:
        try:
            res = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        else:
            return res
