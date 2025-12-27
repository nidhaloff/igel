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
    file_ext = data_path.split(".")[-1].lower()  # Ensure case-insensitivity
    read_funcs = {
        "csv": pd.read_csv,
        "txt": pd.read_csv,
        "xlsx": pd.read_excel,
        "json": pd.read_json,
        "html": pd.read_html
    }

    if file_ext in read_funcs:
        read_func = read_funcs[file_ext]
        return read_func(data_path, **read_data_options) if read_data_options else read_func(data_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")


def update_dataset_props(dataset_props: dict, default_dataset_props: dict):
    for group_key, group_props in default_dataset_props.items():
        if group_key in dataset_props:
            for prop_key, prop_value in group_props.items():
                if prop_key in dataset_props[group_key]:
                    dataset_props[group_key][prop_key] = dataset_props[group_key].get(prop_key, prop_value)
    
    return dataset_props



def handle_missing_values(df, fill_value=np.nan, strategy="mean"):
    logger.info(f"Checking for missing values... \n{df.isna().sum()} \n{'-'*100}")

    if strategy.lower() == "drop":
        logger.info("Dropping rows with missing values...")
        return df.dropna()

    if strategy.lower() not in ["mean", "median", "most_frequent"]:
        raise ValueError(f"Unsupported strategy: {strategy}. Use 'mean', 'median', or 'most_frequent'.")

    logger.info(f"Filling missing values with strategy: {strategy}")
    cleaner = SimpleImputer(strategy=strategy, fill_value=fill_value)
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
    scalers = {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
    }

    if method not in scalers:
        raise ValueError(f"Please choose one of the available scaling methods: {list(scalers.keys())}")

    logger.info(f"Performing {method} scaling...")
    scaler = scalers[method]
    if y is None:
        return scaler.fit_transform(x)
    return scaler.fit_transform(x, y)
