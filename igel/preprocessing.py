from sklearn.impute import SimpleImputer
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

    imputer = SimpleImputer(fill_value=fill_value, strategy=strategy)
    cleaned = imputer.fit_transform(df)

    return pd.DataFrame(cleaned, columns=df.columns)
