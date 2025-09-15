"""
Time Series Anomaly Detection for Igel.

This module provides time series anomaly detection capabilities.
Addresses GitHub issue #292 - Add Support for Time Series Anomaly Detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class TimeSeriesAnomalyDetector:
    """Time series anomaly detection using various algorithms."""
    
    def __init__(self, method: str = "isolation_forest", contamination: float = 0.1):
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        
        if method == "isolation_forest":
            self.model = IsolationForest(contamination=contamination, random_state=42)
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the anomaly detection model."""
        scaled_data = self.scaler.fit_transform(data)
        self.model.fit(scaled_data)
        logger.info(f"Fitted {self.method} anomaly detector")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict anomalies in the data."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        scaled_data = self.scaler.transform(data)
        predictions = self.model.predict(scaled_data)
        return predictions
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies and return results."""
        self.fit(data)
        predictions = self.predict(data)
        
        result = data.copy()
        result["anomaly"] = predictions == -1
        result["anomaly_score"] = self.model.decision_function(self.scaler.transform(data))
        
        return result


def detect_time_series_anomalies(data: pd.DataFrame, method: str = "isolation_forest") -> pd.DataFrame:
    """Quick function to detect anomalies in time series data."""
    detector = TimeSeriesAnomalyDetector(method=method)
    return detector.detect_anomalies(data)
