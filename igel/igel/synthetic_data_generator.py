"""
Synthetic Data Generator for Igel.

This module provides functionality to generate synthetic datasets for testing and examples.
Addresses GitHub issue #285 - Add Support for Synthetic Data Generation.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
import logging

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic datasets for testing and examples."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_classification_data(self, n_samples: int = 1000, n_features: int = 20) -> pd.DataFrame:
        """Generate synthetic classification dataset."""
        X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=self.random_state)
        feature_names = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        return df
    
    def generate_regression_data(self, n_samples: int = 1000, n_features: int = 20) -> pd.DataFrame:
        """Generate synthetic regression dataset."""
        X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=self.random_state)
        feature_names = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        return df


def generate_sample_dataset(dataset_type: str = "classification", **kwargs) -> pd.DataFrame:
    """Quick function to generate sample datasets."""
    generator = SyntheticDataGenerator()
    if dataset_type == "classification":
        return generator.generate_classification_data(**kwargs)
    elif dataset_type == "regression":
        return generator.generate_regression_data(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
