"""
Synthetic Data Generation Utilities

- Tabular data generation
- Time series data generation
- Data with specific distributions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_tabular_data(n_samples=1000, n_features=10, target_column=True):
    """
    Generate synthetic tabular data for classification/regression.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        target_column: Whether to include a target column
    
    Returns:
        DataFrame with synthetic data
    """
    # Generate features
    features = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(features, columns=feature_names)
    
    # Add target column if requested
    if target_column:
        # Simple target based on first feature
        df['target'] = (df['feature_0'] > 0).astype(int)
    
    return df

def generate_time_series_data(n_samples=1000, trend=True, seasonality=True):
    """
    Generate synthetic time series data.
    
    Args:
        n_samples: Number of time points
        trend: Whether to add trend
        seasonality: Whether to add seasonality
    
    Returns:
        DataFrame with time series data
    """
    # Base time series
    time_index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate base signal
    signal = np.random.randn(n_samples) * 0.1
    
    # Add trend
    if trend:
        trend_component = np.linspace(0, 2, n_samples)
        signal += trend_component
    
    # Add seasonality
    if seasonality:
        seasonal_component = 0.5 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
        signal += seasonal_component
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': time_index,
        'value': signal
    })
    
    return df

def generate_categorical_data(n_samples=1000, n_categories=5):
    """
    Generate synthetic categorical data.
    
    Args:
        n_samples: Number of samples
        n_categories: Number of categories
    
    Returns:
        DataFrame with categorical data
    """
    categories = [f'category_{i}' for i in range(n_categories)]
    data = np.random.choice(categories, size=n_samples)
    
    df = pd.DataFrame({
        'category': data,
        'value': np.random.randn(n_samples)
    })
    
    return df