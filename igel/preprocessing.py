from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import logging

logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def update_dataset_props(dataset_props: dict, default_dataset_props: dict):
    for key1 in default_dataset_props.keys():
        if key1 in dataset_props.keys():
            for key2 in default_dataset_props[key1].keys():
                if key2 in dataset_props[key1].keys():
                    default_dataset_props[key1][key2] = dataset_props[key1][key2]

    return default_dataset_props


def handle_missing_values(df, fill_value=np.nan, strategy="mean"):
    logger.info(f"Check for missing values in the dataset ...  \n"
                f"{df.isna().sum()}  \n "
                f"{'-'*100}")

    if strategy.lower() == "drop":
        return df.dropna()

    cleaner = SimpleImputer(fill_value=fill_value, strategy=strategy)
    cleaned = cleaner.fit_transform(df)

    return pd.DataFrame(cleaned, columns=df.columns)


def encode(df, encoding_type="onehotencoding", column=None):
    if not encoding_type:
        raise Exception(f"encoding type should be -> oneHotEncoding or labelEncoding")

    if encoding_type == "onehotencoding":
        return pd.get_dummies(df)

    elif encoding_type == "labelencoding":
        encoder = LabelEncoder()
        encoder.fit(df[column])
        df[column] = encoder.transform(df[column])
        return df

    else:
        raise Exception(f"encoding type should be -> oneHotEncoding or labelEncoding")
