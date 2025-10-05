"""
Module for A/B testing of machine learning models.
Provides functionality to compare model versions and assess statistical significance.
"""

import numpy as np
from scipy import stats
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score, 
    precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error
)
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

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
        
    def compare_predictions(self, X_test: np.ndarray, y_true: np.ndarray, 
                          confidence_level: float = 0.95) -> Dict:
        """
        Compare predictions from both models with comprehensive metrics.
        
        Args:
            X_test: Test features
            y_true: True labels/values
            confidence_level: Confidence level for statistical tests (default: 0.95)
            
        Returns:
            Dict containing comparison metrics and statistical tests
        """
        # Get predictions from both models
        y_pred_a = self.model_a.predict(X_test)
        y_pred_b = self.model_b.predict(X_test)
        
        # Calculate comprehensive metrics based on problem type
        if self.test_type == "classification":
            metrics_a = self._calculate_classification_metrics(y_true, y_pred_a)
            metrics_b = self._calculate_classification_metrics(y_true, y_pred_b)
            
            # Perform multiple statistical tests
            statistical_tests = self._perform_classification_tests(y_true, y_pred_a, y_pred_b)
            
        else:  # regression
            metrics_a = self._calculate_regression_metrics(y_true, y_pred_a)
            metrics_b = self._calculate_regression_metrics(y_true, y_pred_b)
            
            # Perform multiple statistical tests
            statistical_tests = self._perform_regression_tests(y_true, y_pred_a, y_pred_b)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            metrics_a, metrics_b, confidence_level
        )
        
        return {
            "model_a_metrics": metrics_a,
            "model_b_metrics": metrics_b,
            "statistical_tests": statistical_tests,
            "confidence_intervals": confidence_intervals,
            "test_summary": self._summarize_test_results(statistical_tests, confidence_level)
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
        report.append("=" * 30)
        report.append(f"Test Type: {self.test_type.title()}")
        report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add metrics based on problem type
        if self.test_type == "classification":
            metrics_a = comparison_results['model_a_metrics']
            metrics_b = comparison_results['model_b_metrics']
            
            report.append("\nModel Performance Metrics")
            report.append("-" * 25)
            report.append(f"Model A - Accuracy: {metrics_a['accuracy']:.4f}")
            report.append(f"Model A - Precision: {metrics_a.get('precision', 'N/A'):.4f}")
            report.append(f"Model A - Recall: {metrics_a.get('recall', 'N/A'):.4f}")
            report.append(f"Model A - F1-Score: {metrics_a.get('f1_score', 'N/A'):.4f}")
            
            report.append(f"\nModel B - Accuracy: {metrics_b['accuracy']:.4f}")
            report.append(f"Model B - Precision: {metrics_b.get('precision', 'N/A'):.4f}")
            report.append(f"Model B - Recall: {metrics_b.get('recall', 'N/A'):.4f}")
            report.append(f"Model B - F1-Score: {metrics_b.get('f1_score', 'N/A'):.4f}")
            
        else:  # regression
            metrics_a = comparison_results['model_a_metrics']
            metrics_b = comparison_results['model_b_metrics']
            
            report.append("\nModel Performance Metrics")
            report.append("-" * 25)
            report.append(f"Model A - MSE: {metrics_a['mse']:.4f}")
            report.append(f"Model A - RMSE: {metrics_a['rmse']:.4f}")
            report.append(f"Model A - MAE: {metrics_a['mae']:.4f}")
            report.append(f"Model A - R²: {metrics_a['r2']:.4f}")
            
            report.append(f"\nModel B - MSE: {metrics_b['mse']:.4f}")
            report.append(f"Model B - RMSE: {metrics_b['rmse']:.4f}")
            report.append(f"Model B - MAE: {metrics_b['mae']:.4f}")
            report.append(f"Model B - R²: {metrics_b['r2']:.4f}")
        
        # Add statistical test results
        report.append("\nStatistical Test Results")
        report.append("-" * 25)
        
        statistical_tests = comparison_results.get('statistical_tests', {})
        for test_name, test_result in statistical_tests.items():
            report.append(f"\n{test_result.get('test_name', test_name)}:")
            report.append(f"  Statistic: {test_result['statistic']:.4f}")
            report.append(f"  P-value: {test_result['p_value']:.4f}")
            report.append(f"  Significant: {'Yes' if test_result['significant'] else 'No'}")
        
        # Add test summary
        test_summary = comparison_results.get('test_summary', {})
        if test_summary:
            report.append("\nTest Summary")
            report.append("-" * 12)
            report.append(f"Overall Significance: {'Yes' if test_summary.get('overall_significance') else 'No'}")
            report.append(f"Significant Tests: {', '.join(test_summary.get('significant_tests', []))}")
            report.append(f"Non-significant Tests: {', '.join(test_summary.get('non_significant_tests', []))}")
            report.append(f"\nRecommendation: {test_summary.get('recommendation', 'N/A')}")
        
        return "\n".join(report)
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive classification metrics."""
        try:
            # Get prediction probabilities if available
            y_pred_proba = None
            if hasattr(self.model_a, 'predict_proba'):
                y_pred_proba = self.model_a.predict_proba(y_true.reshape(-1, 1) if len(y_true.shape) == 1 else y_true)
            
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
                "predictions": y_pred
            }
            
            # Add AUC if probabilities are available
            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                try:
                    metrics["auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                except:
                    pass
            
            return metrics
        except Exception as e:
            logger.warning(f"Error calculating classification metrics: {e}")
            return {"accuracy": accuracy_score(y_true, y_pred), "predictions": y_pred}
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive regression metrics."""
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "predictions": y_pred
        }
    
    def _perform_classification_tests(self, y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict:
        """Perform multiple statistical tests for classification."""
        tests = {}
        
        # McNemar's test
        try:
            contingency_table = self._create_contingency_table(y_true, y_pred_a, y_pred_b)
            stat, p_value = stats.mcnemar(contingency_table, correction=True)
            tests["mcnemar"] = {
                "statistic": stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "test_name": "McNemar's Test"
            }
        except Exception as e:
            logger.warning(f"McNemar test failed: {e}")
        
        # Chi-square test for independence
        try:
            chi2_stat, chi2_p = stats.chi2_contingency(contingency_table)[:2]
            tests["chi_square"] = {
                "statistic": chi2_stat,
                "p_value": chi2_p,
                "significant": chi2_p < 0.05,
                "test_name": "Chi-Square Test"
            }
        except Exception as e:
            logger.warning(f"Chi-square test failed: {e}")
        
        return tests
    
    def _perform_regression_tests(self, y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict:
        """Perform multiple statistical tests for regression."""
        tests = {}
        
        # Wilcoxon signed-rank test
        try:
            errors_a = (y_true - y_pred_a) ** 2
            errors_b = (y_true - y_pred_b) ** 2
            stat, p_value = stats.wilcoxon(errors_a, errors_b)
            tests["wilcoxon"] = {
                "statistic": stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "test_name": "Wilcoxon Signed-Rank Test"
            }
        except Exception as e:
            logger.warning(f"Wilcoxon test failed: {e}")
        
        # Paired t-test
        try:
            stat, p_value = stats.ttest_rel(errors_a, errors_b)
            tests["paired_ttest"] = {
                "statistic": stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "test_name": "Paired t-test"
            }
        except Exception as e:
            logger.warning(f"Paired t-test failed: {e}")
        
        return tests
    
    def _calculate_confidence_intervals(self, metrics_a: Dict, metrics_b: Dict, confidence_level: float) -> Dict:
        """Calculate confidence intervals for key metrics."""
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        intervals = {}
        
        # Calculate confidence intervals for accuracy (classification) or R² (regression)
        if self.test_type == "classification":
            if "accuracy" in metrics_a:
                n = len(metrics_a.get("predictions", []))
                if n > 0:
                    p_a = metrics_a["accuracy"]
                    p_b = metrics_b["accuracy"]
                    
                    se_a = np.sqrt(p_a * (1 - p_a) / n)
                    se_b = np.sqrt(p_b * (1 - p_b) / n)
                    
                    intervals["accuracy"] = {
                        "model_a": {
                            "lower": max(0, p_a - z_score * se_a),
                            "upper": min(1, p_a + z_score * se_a)
                        },
                        "model_b": {
                            "lower": max(0, p_b - z_score * se_b),
                            "upper": min(1, p_b + z_score * se_b)
                        }
                    }
        else:
            if "r2" in metrics_a:
                intervals["r2"] = {
                    "model_a": {"value": metrics_a["r2"]},
                    "model_b": {"value": metrics_b["r2"]}
                }
        
        return intervals
    
    def _summarize_test_results(self, statistical_tests: Dict, confidence_level: float) -> Dict:
        """Summarize the results of all statistical tests."""
        significant_tests = []
        non_significant_tests = []
        
        for test_name, test_result in statistical_tests.items():
            if test_result.get("significant", False):
                significant_tests.append(test_name)
            else:
                non_significant_tests.append(test_name)
        
        return {
            "confidence_level": confidence_level,
            "significant_tests": significant_tests,
            "non_significant_tests": non_significant_tests,
            "overall_significance": len(significant_tests) > 0,
            "recommendation": self._get_recommendation(significant_tests, statistical_tests)
        }
    
    def _get_recommendation(self, significant_tests: List[str], statistical_tests: Dict) -> str:
        """Generate a recommendation based on test results."""
        if not significant_tests:
            return "No significant difference found between models. Consider factors like computational cost, interpretability, or business requirements."
        
        if len(significant_tests) == 1:
            test_name = significant_tests[0]
            return f"Significant difference found using {statistical_tests[test_name]['test_name']}. Consider the practical significance of the difference."
        
        return f"Multiple tests ({len(significant_tests)}) show significant differences. Review all metrics and choose the model that best fits your requirements."
    
    def visualize_comparison(self, comparison_results: Dict, save_path: Optional[str] = None) -> None:
        """Create visualizations for model comparison."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Model A/B Testing Results', fontsize=16)
            
            if self.test_type == "classification":
                self._plot_classification_comparison(comparison_results, axes)
            else:
                self._plot_regression_comparison(comparison_results, axes)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
    
    def _plot_classification_comparison(self, results: Dict, axes) -> None:
        """Plot classification comparison visualizations."""
        metrics_a = results["model_a_metrics"]
        metrics_b = results["model_b_metrics"]
        
        # Accuracy comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values_a = [metrics_a.get(m, 0) for m in metrics]
        values_b = [metrics_b.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, values_a, width, label='Model A', alpha=0.8)
        axes[0, 0].bar(x + width/2, values_b, width, label='Model B', alpha=0.8)
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Classification Metrics Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Statistical test results
        test_names = list(results["statistical_tests"].keys())
        p_values = [results["statistical_tests"][t]["p_value"] for t in test_names]
        
        axes[0, 1].bar(test_names, p_values, color=['red' if p < 0.05 else 'blue' for p in p_values])
        axes[0, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        axes[0, 1].set_ylabel('P-value')
        axes[0, 1].set_title('Statistical Test Results')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Hide unused subplots
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
    
    def _plot_regression_comparison(self, results: Dict, axes) -> None:
        """Plot regression comparison visualizations."""
        metrics_a = results["model_a_metrics"]
        metrics_b = results["model_b_metrics"]
        
        # Metrics comparison
        metrics = ['mse', 'rmse', 'mae', 'r2']
        values_a = [metrics_a.get(m, 0) for m in metrics]
        values_b = [metrics_b.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, values_a, width, label='Model A', alpha=0.8)
        axes[0, 0].bar(x + width/2, values_b, width, label='Model B', alpha=0.8)
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title('Regression Metrics Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Statistical test results
        test_names = list(results["statistical_tests"].keys())
        p_values = [results["statistical_tests"][t]["p_value"] for t in test_names]
        
        axes[0, 1].bar(test_names, p_values, color=['red' if p < 0.05 else 'blue' for p in p_values])
        axes[0, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        axes[0, 1].set_ylabel('P-value')
        axes[0, 1].set_title('Statistical Test Results')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Hide unused subplots
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
    
    def export_results(self, comparison_results: Dict, file_path: str) -> None:
        """Export comparison results to JSON file."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            export_data = self._prepare_for_export(comparison_results)
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Results exported to {file_path}")
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
    
    def _prepare_for_export(self, results: Dict) -> Dict:
        """Prepare results for JSON export by converting numpy arrays."""
        export_data = {}
        
        for key, value in results.items():
            if isinstance(value, dict):
                export_data[key] = self._prepare_for_export(value)
            elif isinstance(value, np.ndarray):
                export_data[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                export_data[key] = value.item()
            else:
                export_data[key] = value
        
        return export_data 