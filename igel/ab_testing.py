"""
Module for A/B testing of machine learning models.
Provides functionality to compare model versions and assess statistical significance.
"""

import numpy as np
from scipy import stats
import pandas as pd
from typing import List, Dict, Tuple, Union
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)

class ModelComparison:
    """Class to handle A/B testing between two model versions."""
    
    def __init__(self, model_a, model_b, test_type: str = "classification"):
        """
        Initialize model comparison.
        
        Args:
            model_a: First model to compare
            model_b: Second model to compare
            test_type: Type of models being compared ("classification" or "regression")
        """
        self.model_a = model_a
        self.model_b = model_b
        self.test_type = test_type
        
    def compare_predictions(self, X_test: np.ndarray, y_true: np.ndarray) -> Dict:
        """
        Compare predictions from both models.
        
        Args:
            X_test: Test features
            y_true: True labels/values
            
        Returns:
            Dict containing comparison metrics
        """
        # Get predictions from both models
        y_pred_a = self.model_a.predict(X_test)
        y_pred_b = self.model_b.predict(X_test)
        
        # Calculate metrics based on problem type
        if self.test_type == "classification":
            metrics_a = {
                "accuracy": accuracy_score(y_true, y_pred_a),
                "predictions": y_pred_a
            }
            metrics_b = {
                "accuracy": accuracy_score(y_true, y_pred_b),
                "predictions": y_pred_b
            }
            
            # Perform McNemar's test for statistical significance
            contingency_table = self._create_contingency_table(y_true, y_pred_a, y_pred_b)
            stat, p_value = stats.mcnemar(contingency_table, correction=True)
            
        else:  # regression
            metrics_a = {
                "mse": mean_squared_error(y_true, y_pred_a),
                "r2": r2_score(y_true, y_pred_a),
                "predictions": y_pred_a
            }
            metrics_b = {
                "mse": mean_squared_error(y_true, y_pred_b),
                "r2": r2_score(y_true, y_pred_b),
                "predictions": y_pred_b
            }
            
            # Perform Wilcoxon signed-rank test
            stat, p_value = stats.wilcoxon(
                (y_true - y_pred_a)**2,
                (y_true - y_pred_b)**2
            )
        
        return {
            "model_a_metrics": metrics_a,
            "model_b_metrics": metrics_b,
            "statistical_test": {
                "statistic": stat,
                "p_value": p_value,
                "significant_difference": p_value < 0.05
            }
        }
    
    def _create_contingency_table(
        self, y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray
    ) -> np.ndarray:
        """
        Create contingency table for McNemar's test.
        
        Returns:
            2x2 contingency table as numpy array
        """
        # Both correct
        n11 = sum((y_pred_a == y_true) & (y_pred_b == y_true))
        # A correct, B wrong
        n12 = sum((y_pred_a == y_true) & (y_pred_b != y_true))
        # A wrong, B correct
        n21 = sum((y_pred_a != y_true) & (y_pred_b == y_true))
        # Both wrong
        n22 = sum((y_pred_a != y_true) & (y_pred_b != y_true))
        
        return np.array([[n11, n12], [n21, n22]])
    
    def generate_report(self, comparison_results: Dict) -> str:
        """
        Generate a formatted report of the comparison results.
        
        Args:
            comparison_results: Results from compare_predictions
            
        Returns:
            Formatted string containing the comparison report
        """
        report = []
        report.append("Model A/B Testing Results")
        report.append("=" * 25)
        
        # Add metrics based on problem type
        if self.test_type == "classification":
            report.append(f"\nModel A Accuracy: {comparison_results['model_a_metrics']['accuracy']:.4f}")
            report.append(f"Model B Accuracy: {comparison_results['model_b_metrics']['accuracy']:.4f}")
        else:
            report.append(f"\nModel A MSE: {comparison_results['model_a_metrics']['mse']:.4f}")
            report.append(f"Model A R²: {comparison_results['model_a_metrics']['r2']:.4f}")
            report.append(f"Model B MSE: {comparison_results['model_b_metrics']['mse']:.4f}")
            report.append(f"Model B R²: {comparison_results['model_b_metrics']['r2']:.4f}")
        
        # Add statistical test results
        report.append("\nStatistical Test Results")
        report.append("-" * 22)
        report.append(f"Test Statistic: {comparison_results['statistical_test']['statistic']:.4f}")
        report.append(f"P-value: {comparison_results['statistical_test']['p_value']:.4f}")
        report.append(
            f"Significant Difference: {'Yes' if comparison_results['statistical_test']['significant_difference'] else 'No'}"
        )
        
        return "\n".join(report) 