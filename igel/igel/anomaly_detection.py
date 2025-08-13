"""
Time Series Anomaly Detection Utilities

- Isolation Forest for anomaly detection
- Statistical methods for outlier detection
- Time series specific anomaly detection
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_anomalies_isolation_forest(data, contamination=0.1):
    """
    Detect anomalies using Isolation Forest.
    
    Args:
        data: Time series data (1D array)
        contamination: Expected proportion of anomalies
    
    Returns:
        Dictionary with anomaly scores and predictions
    """
    # Reshape data for sklearn
    X = data.reshape(-1, 1)
    
    # Fit isolation forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    predictions = iso_forest.fit_predict(X)
    scores = iso_forest.score_samples(X)
    
    return {
        "predictions": predictions,  # -1 for anomalies, 1 for normal
        "scores": scores,
        "anomaly_indices": np.where(predictions == -1)[0]
    }

def detect_statistical_anomalies(data, threshold=3):
    """
    Detect anomalies using statistical methods (Z-score).
    
    Args:
        data: Time series data
        threshold: Number of standard deviations for anomaly detection
    
    Returns:
        Dictionary with anomaly information
    """
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    
    anomalies = z_scores > threshold
    
    return {
        "anomaly_indices": np.where(anomalies)[0],
        "z_scores": z_scores,
        "threshold": threshold
    }