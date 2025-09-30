"""
Model Monitoring and Alerting System for Igel.

This module provides comprehensive model monitoring and alerting capabilities.
Addresses GitHub issue #341 - Create Model Monitoring and Alerting System.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitor model performance and data drift in production."""
    
    def __init__(self, model_name: str = "default_model"):
        self.model_name = model_name
        self.metrics_history = []
        self.alerts = []
        self.baseline_metrics = {}
        self.monitoring_enabled = True
    
    def set_baseline(self, metrics: Dict[str, float]):
        """Set baseline metrics for comparison."""
        self.baseline_metrics = metrics
        logger.info(f"Baseline metrics set for {self.model_name}")
    
    def log_prediction(self, features: Dict[str, Any], prediction: Any,
                       actual: Optional[Any] = None, timestamp: Optional[datetime] = None):
        """Log a single prediction."""
        if not self.monitoring_enabled:
            return
        
        log_entry = {
            'timestamp': timestamp or datetime.now().isoformat(),
            'features': features,
            'prediction': prediction,
            'actual': actual
        }
        self.metrics_history.append(log_entry)
    
    def check_performance_drift(self, current_metrics: Dict[str, float],
                               threshold: float = 0.1) -> bool:
        """Check if performance has drifted from baseline."""
        if not self.baseline_metrics:
            logger.warning("No baseline metrics set")
            return False
        
        drift_detected = False
        for metric, value in current_metrics.items():
            if metric in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric]
                drift = abs(value - baseline_value) / baseline_value
                
                if drift > threshold:
                    alert = {
                        'type': 'performance_drift',
                        'metric': metric,
                        'baseline': baseline_value,
                        'current': value,
                        'drift_percentage': drift * 100,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.alerts.append(alert)
                    drift_detected = True
                    logger.warning(f"Performance drift detected for {metric}: {drift*100:.2f}%")
        
        return drift_detected
    
    def check_data_drift(self, current_data: pd.DataFrame, reference_data: pd.DataFrame,
                        method: str = "statistical") -> Dict[str, Any]:
        """Check for data drift using statistical methods."""
        drift_report = {}
        
        for column in current_data.select_dtypes(include=[np.number]).columns:
            if column in reference_data.columns:
                # Simple statistical test
                current_mean = current_data[column].mean()
                reference_mean = reference_data[column].mean()
                current_std = current_data[column].std()
                reference_std = reference_data[column].std()
                
                mean_drift = abs(current_mean - reference_mean) / reference_mean if reference_mean != 0 else 0
                std_drift = abs(current_std - reference_std) / reference_std if reference_std != 0 else 0
                
                drift_report[column] = {
                    'mean_drift': mean_drift,
                    'std_drift': std_drift,
                    'drift_detected': mean_drift > 0.1 or std_drift > 0.1
                }
        
        return drift_report
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring activities."""
        return {
            'model_name': self.model_name,
            'total_predictions': len(self.metrics_history),
            'total_alerts': len(self.alerts),
            'monitoring_enabled': self.monitoring_enabled,
            'baseline_set': bool(self.baseline_metrics)
        }


class AlertManager:
    """Manage and dispatch alerts for model monitoring."""
    
    def __init__(self):
        self.alert_handlers = []
        self.alert_history = []
    
    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Add a custom alert handler."""
        self.alert_handlers.append(handler)
    
    def send_alert(self, alert: Dict[str, Any]):
        """Send alert to all registered handlers."""
        self.alert_history.append(alert)
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def clear_alerts(self):
        """Clear alert history."""
        self.alert_history = []


def monitor_model(model, current_data: pd.DataFrame, reference_data: pd.DataFrame,
                 model_name: str = "model") -> Dict[str, Any]:
    """Quick function to monitor model performance and data drift."""
    monitor = ModelMonitor(model_name)
    drift_report = monitor.check_data_drift(current_data, reference_data)
    return {
        'monitoring_summary': monitor.get_monitoring_summary(),
        'drift_report': drift_report
    }
