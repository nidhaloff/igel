import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def read_data_to_df(data_path: str, **read_data_options):
    """
    read data depending on its extension and convert it to a pandas dataframe
    """
    file_ext = data_path.split(".")[-1]
    if file_ext == "csv" or file_ext == "txt":
        return (
            pd.read_csv(data_path, **read_data_options)
            if read_data_options
            else pd.read_csv(data_path)
        )
    elif file_ext == "xlsx":
        return (
            pd.read_excel(data_path, **read_data_options)
            if read_data_options
            else pd.read_excel(data_path)
        )
    elif file_ext == "json":
        return (
            pd.read_json(data_path, **read_data_options)
            if read_data_options
            else pd.read_json(data_path)
        )
    elif file_ext == "html":
        return (
            pd.read_html(data_path, **read_data_options)
            if read_data_options
            else pd.read_html(data_path)
        )


def update_dataset_props(dataset_props: dict, default_dataset_props: dict):
    for key1 in default_dataset_props.keys():
        if key1 in dataset_props.keys():
            for key2 in default_dataset_props[key1].keys():
                if key2 in dataset_props[key1].keys():
                    default_dataset_props[key1][key2] = dataset_props[key1][
                        key2
                    ]

    return default_dataset_props


def handle_missing_values(df, fill_value=np.nan, strategy="mean"):
    logger.info(
        f"Check for missing values in the dataset ...  \n"
        f"{df.isna().sum()}  \n "
        f"{'-'*100}"
    )

    if strategy.lower() == "drop":
        return df.dropna()

    cleaner = SimpleImputer(fill_value=fill_value, strategy=strategy)
    cleaned = cleaner.fit_transform(df)

    return pd.DataFrame(cleaned, columns=df.columns)


def encode(df, encoding_type="onehotencoding", column=None):
    if not encoding_type:
        raise Exception(
            f"encoding type should be -> oneHotEncoding or labelEncoding"
        )

    if encoding_type == "onehotencoding":
        logger.info(f"performing a one hot encoding ...")
        return pd.get_dummies(df), None

    elif encoding_type == "labelencoding":
        if not column:
            raise Exception(
                "if you choose to label encode your data, "
                "then you need to provide the column you want to encode from your dataset"
            )
        logger.info(f"performing a label encoding ...")
        encoder = LabelEncoder()
        encoder.fit(df[column])
        classes_map = {
            cls: int(lbl)
            for (cls, lbl) in zip(
                encoder.classes_, encoder.transform(encoder.classes_)
            )
        }
        logger.info(f"label encoding classes => {encoder.classes_}")
        logger.info(f"classes map => {classes_map}")
        df[column] = encoder.transform(df[column])
        return df, classes_map

    else:
        raise Exception(
            f"encoding type should be -> oneHotEncoding or labelEncoding"
        )


def normalize(x, y=None, method="standard"):
    methods = ("minmax", "standard")

    if method not in methods:
        raise Exception(
            f"Please choose one of the available scaling methods => {methods}"
        )
    logger.info(f"performing a {method} scaling ...")
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    if not y:
        return scaler.fit_transform(X=x)
    else:
        return scaler.fit_transform(X=x, y=y)
