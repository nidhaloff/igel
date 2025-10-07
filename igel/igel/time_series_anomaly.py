"""
Time Series Anomaly Detection for igel.
Provides comprehensive anomaly detection capabilities for time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class TimeSeriesAnomalyDetector:
    """
    Comprehensive time series anomaly detection framework.
    """
    
    def __init__(self, method: str = "isolation_forest", **kwargs):
        """
        Initialize the anomaly detector.
        
        Args:
            method: Detection method ('isolation_forest', 'dbscan', 'statistical', 'lstm_autoencoder')
            **kwargs: Additional parameters for the detector
        """
        self.method = method
        self.detector = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.anomaly_threshold = 0.5
        self.detection_results = {}
        
    def fit(self, data: np.ndarray, **kwargs):
        """
        Fit the anomaly detector to the data.
        
        Args:
            data: Time series data (1D or 2D array)
            **kwargs: Additional fitting parameters
        """
        try:
            if self.method == "isolation_forest":
                self._fit_isolation_forest(data, **kwargs)
            elif self.method == "dbscan":
                self._fit_dbscan(data, **kwargs)
            elif self.method == "statistical":
                self._fit_statistical(data, **kwargs)
            elif self.method == "lstm_autoencoder":
                self._fit_lstm_autoencoder(data, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {self.method}")
            
            self.is_fitted = True
            logger.info(f"Anomaly detector fitted using {self.method}")
            
        except Exception as e:
            logger.error(f"Failed to fit anomaly detector: {e}")
            raise
    
    def _fit_isolation_forest(self, data: np.ndarray, **kwargs):
        """Fit Isolation Forest detector."""
        contamination = kwargs.get('contamination', 0.1)
        n_estimators = kwargs.get('n_estimators', 100)
        
        self.detector = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        
        # Reshape data if needed
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        self.detector.fit(data)
    
    def _fit_dbscan(self, data: np.ndarray, **kwargs):
        """Fit DBSCAN detector."""
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 5)
        
        self.detector = DBSCAN(eps=eps, min_samples=min_samples)
        
        # Reshape data if needed
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Scale data
        data_scaled = self.scaler.fit_transform(data)
        self.detector.fit(data_scaled)
    
    def _fit_statistical(self, data: np.ndarray, **kwargs):
        """Fit statistical detector."""
        window_size = kwargs.get('window_size', 10)
        threshold_multiplier = kwargs.get('threshold_multiplier', 2.0)
        
        # Calculate rolling statistics
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        self.detector = {
            'method': 'statistical',
            'window_size': window_size,
            'threshold_multiplier': threshold_multiplier,
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0)
        }
    
    def _fit_lstm_autoencoder(self, data: np.ndarray, **kwargs):
        """Fit LSTM autoencoder detector."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector
        except ImportError:
            raise ImportError("TensorFlow required for LSTM autoencoder. Install with: pip install tensorflow")
        
        sequence_length = kwargs.get('sequence_length', 10)
        encoding_dim = kwargs.get('encoding_dim', 32)
        epochs = kwargs.get('epochs', 50)
        
        # Prepare sequences
        sequences = self._create_sequences(data, sequence_length)
        
        # Build autoencoder
        input_layer = Input(shape=(sequence_length, data.shape[1]))
        encoder = LSTM(encoding_dim, activation='relu')(input_layer)
        decoder = RepeatVector(sequence_length)(encoder)
        decoder = LSTM(data.shape[1], return_sequences=True)(decoder)
        
        autoencoder = Model(input_layer, decoder)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train autoencoder
        autoencoder.fit(sequences, sequences, epochs=epochs, verbose=0)
        
        self.detector = {
            'method': 'lstm_autoencoder',
            'model': autoencoder,
            'sequence_length': sequence_length,
            'threshold': np.percentile(autoencoder.predict(sequences), 95)
        }
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Create sequences for LSTM."""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict anomalies in the data.
        
        Args:
            data: Time series data to analyze
            
        Returns:
            Array of anomaly predictions (-1 for anomaly, 1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        try:
            if self.method == "isolation_forest":
                return self._predict_isolation_forest(data)
            elif self.method == "dbscan":
                return self._predict_dbscan(data)
            elif self.method == "statistical":
                return self._predict_statistical(data)
            elif self.method == "lstm_autoencoder":
                return self._predict_lstm_autoencoder(data)
            else:
                raise ValueError(f"Unsupported method: {self.method}")
                
        except Exception as e:
            logger.error(f"Failed to predict anomalies: {e}")
            raise
    
    def _predict_isolation_forest(self, data: np.ndarray) -> np.ndarray:
        """Predict using Isolation Forest."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        return self.detector.predict(data)
    
    def _predict_dbscan(self, data: np.ndarray) -> np.ndarray:
        """Predict using DBSCAN."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        data_scaled = self.scaler.transform(data)
        predictions = self.detector.fit_predict(data_scaled)
        
        # Convert DBSCAN labels to anomaly format (-1 for anomaly, 1 for normal)
        predictions = np.where(predictions == -1, -1, 1)
        return predictions
    
    def _predict_statistical(self, data: np.ndarray) -> np.ndarray:
        """Predict using statistical method."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        window_size = self.detector['window_size']
        threshold_multiplier = self.detector['threshold_multiplier']
        
        predictions = []
        
        for i in range(len(data)):
            if i < window_size:
                # Use global statistics for initial points
                z_scores = np.abs((data[i] - self.detector['mean']) / self.detector['std'])
            else:
                # Use rolling window statistics
                window_data = data[i-window_size:i]
                window_mean = np.mean(window_data, axis=0)
                window_std = np.std(window_data, axis=0)
                z_scores = np.abs((data[i] - window_mean) / (window_std + 1e-8))
            
            # Anomaly if z-score exceeds threshold
            is_anomaly = np.any(z_scores > threshold_multiplier)
            predictions.append(-1 if is_anomaly else 1)
        
        return np.array(predictions)
    
    def _predict_lstm_autoencoder(self, data: np.ndarray) -> np.ndarray:
        """Predict using LSTM autoencoder."""
        sequence_length = self.detector['sequence_length']
        model = self.detector['model']
        threshold = self.detector['threshold']
        
        # Create sequences
        sequences = self._create_sequences(data, sequence_length)
        
        # Get reconstruction errors
        reconstructions = model.predict(sequences)
        mse = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
        
        # Pad predictions for initial points
        predictions = np.ones(len(data))
        predictions[sequence_length-1:] = np.where(mse > threshold, -1, 1)
        
        return predictions
    
    def detect_anomalies(self, data: np.ndarray, return_scores: bool = False) -> Dict[str, Any]:
        """
        Detect anomalies and return comprehensive results.
        
        Args:
            data: Time series data
            return_scores: Whether to return anomaly scores
            
        Returns:
            Dictionary with detection results
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        # Get predictions
        predictions = self.predict(data)
        
        # Calculate anomaly scores if possible
        scores = None
        if hasattr(self.detector, 'decision_function'):
            scores = self.detector.decision_function(data.reshape(-1, 1) if len(data.shape) == 1 else data)
        elif hasattr(self.detector, 'score_samples'):
            scores = self.detector.score_samples(data.reshape(-1, 1) if len(data.shape) == 1 else data)
        
        # Count anomalies
        n_anomalies = np.sum(predictions == -1)
        n_normal = np.sum(predictions == 1)
        anomaly_rate = n_anomalies / len(predictions)
        
        # Get anomaly indices
        anomaly_indices = np.where(predictions == -1)[0]
        
        results = {
            'predictions': predictions.tolist(),
            'n_anomalies': int(n_anomalies),
            'n_normal': int(n_normal),
            'anomaly_rate': float(anomaly_rate),
            'anomaly_indices': anomaly_indices.tolist(),
            'method': self.method,
            'timestamp': datetime.now().isoformat()
        }
        
        if scores is not None and return_scores:
            results['anomaly_scores'] = scores.tolist()
        
        self.detection_results = results
        return results
    
    def evaluate_detection(self, true_labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate anomaly detection performance.
        
        Args:
            true_labels: True anomaly labels (-1 for anomaly, 1 for normal)
            
        Returns:
            Evaluation metrics
        """
        if 'predictions' not in self.detection_results:
            raise ValueError("No detection results available. Run detect_anomalies() first.")
        
        predictions = np.array(self.detection_results['predictions'])
        
        # Calculate metrics
        accuracy = np.mean(predictions == true_labels)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions, labels=[-1, 1]).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        evaluation = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        return evaluation
    
    def generate_report(self) -> str:
        """Generate a comprehensive anomaly detection report."""
        if not self.detection_results:
            return "No detection results available. Run detect_anomalies() first."
        
        report = []
        report.append("Time Series Anomaly Detection Report")
        report.append("=" * 40)
        report.append(f"Detection Method: {self.detection_results['method']}")
        report.append(f"Total Data Points: {len(self.detection_results['predictions'])}")
        report.append(f"Anomalies Detected: {self.detection_results['n_anomalies']}")
        report.append(f"Normal Points: {self.detection_results['n_normal']}")
        report.append(f"Anomaly Rate: {self.detection_results['anomaly_rate']:.2%}")
        report.append(f"Detection Time: {self.detection_results['timestamp']}")
        
        if 'anomaly_scores' in self.detection_results:
            scores = self.detection_results['anomaly_scores']
            report.append(f"Score Range: [{min(scores):.4f}, {max(scores):.4f}]")
        
        return "\n".join(report)
    
    def save_results(self, filepath: str):
        """Save detection results to file."""
        with open(f"{filepath}_anomaly_results.json", 'w') as f:
            json.dump(self.detection_results, f, indent=2, default=str)
        
        logger.info(f"Anomaly detection results saved to {filepath}_anomaly_results.json")
