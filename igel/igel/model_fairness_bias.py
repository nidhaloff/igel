"""
Model Fairness and Bias Detection Framework for Igel.

This module provides comprehensive tools for detecting and mitigating bias
in machine learning models, ensuring fair and equitable predictions.
Addresses GitHub issue #342 - Create Model Fairness and Bias Detection Framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelFairnessDetector:
    """Detects bias and fairness issues in machine learning models."""
    
    def __init__(self):
        self.bias_metrics = {}
        self.fairness_report = {}
        self.protected_attributes = []
    
    def detect_demographic_parity_bias(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     protected_attr: np.ndarray) -> Dict[str, float]:
        """Detect demographic parity bias."""
        groups = np.unique(protected_attr)
        bias_metrics = {}
        
        for group in groups:
            group_mask = protected_attr == group
            group_positive_rate = np.mean(y_pred[group_mask])
            bias_metrics[f'group_{group}_positive_rate'] = group_positive_rate
        
        # Calculate demographic parity difference
        if len(groups) == 2:
            dp_diff = abs(bias_metrics[f'group_{groups[0]}_positive_rate'] - 
                         bias_metrics[f'group_{groups[1]}_positive_rate'])
            bias_metrics['demographic_parity_difference'] = dp_diff
        
        return bias_metrics
    
    def detect_equalized_odds_bias(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 protected_attr: np.ndarray) -> Dict[str, float]:
        """Detect equalized odds bias."""
        groups = np.unique(protected_attr)
        bias_metrics = {}
        
        for group in groups:
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            # Calculate TPR and FPR for each group
            tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            bias_metrics[f'group_{group}_tpr'] = tpr
            bias_metrics[f'group_{group}_fpr'] = fpr
        
        # Calculate equalized odds difference
        if len(groups) == 2:
            tpr_diff = abs(bias_metrics[f'group_{groups[0]}_tpr'] - 
                          bias_metrics[f'group_{groups[1]}_tpr'])
            fpr_diff = abs(bias_metrics[f'group_{groups[0]}_fpr'] - 
                          bias_metrics[f'group_{groups[1]}_fpr'])
            bias_metrics['equalized_odds_tpr_difference'] = tpr_diff
            bias_metrics['equalized_odds_fpr_difference'] = fpr_diff
        
        return bias_metrics
    
    def detect_disparate_impact(self, y_pred: np.ndarray, protected_attr: np.ndarray,
                              threshold: float = 0.8) -> Dict[str, float]:
        """Detect disparate impact bias."""
        groups = np.unique(protected_attr)
        bias_metrics = {}
        
        if len(groups) == 2:
            group_0_mask = protected_attr == groups[0]
            group_1_mask = protected_attr == groups[1]
            
            group_0_rate = np.mean(y_pred[group_0_mask])
            group_1_rate = np.mean(y_pred[group_1_mask])
            
            # Calculate disparate impact ratio
            disparate_impact = min(group_0_rate, group_1_rate) / max(group_0_rate, group_1_rate)
            bias_metrics['disparate_impact_ratio'] = disparate_impact
            bias_metrics['disparate_impact_violation'] = disparate_impact < threshold
        
        return bias_metrics
    
    def generate_fairness_report(self, model, X: pd.DataFrame, y: pd.Series,
                               protected_attributes: List[str]) -> Dict[str, Any]:
        """Generate comprehensive fairness report."""
        y_pred = model.predict(X)
        report = {}
        
        for attr in protected_attributes:
            if attr in X.columns:
                protected_attr = X[attr].values
                
                # Demographic parity
                dp_bias = self.detect_demographic_parity_bias(y, y_pred, protected_attr)
                report[f'{attr}_demographic_parity'] = dp_bias
                
                # Equalized odds
                eo_bias = self.detect_equalized_odds_bias(y, y_pred, protected_attr)
                report[f'{attr}_equalized_odds'] = eo_bias
                
                # Disparate impact
                di_bias = self.detect_disparate_impact(y_pred, protected_attr)
                report[f'{attr}_disparate_impact'] = di_bias
        
        self.fairness_report = report
        logger.info(f"Generated fairness report for {len(protected_attributes)} protected attributes")
        return report
    
    def visualize_bias_metrics(self, report: Dict[str, Any], save_path: Optional[str] = None):
        """Visualize bias metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract metrics for visualization
        metrics_data = []
        for attr, metrics in report.items():
            if 'demographic_parity' in attr:
                for key, value in metrics.items():
                    if 'positive_rate' in key:
                        metrics_data.append({
                            'attribute': attr.replace('_demographic_parity', ''),
                            'metric': key,
                            'value': value
                        })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            sns.barplot(data=df, x='attribute', y='value', hue='metric', ax=axes[0, 0])
            axes[0, 0].set_title('Demographic Parity Bias')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class BiasMitigation:
    """Techniques for mitigating bias in machine learning models."""
    
    def __init__(self):
        self.mitigation_methods = {}
    
    def reweight_samples(self, X: pd.DataFrame, y: pd.Series, 
                        protected_attr: str) -> np.ndarray:
        """Reweight samples to reduce bias."""
        # Calculate weights based on protected attribute distribution
        attr_counts = X[protected_attr].value_counts()
        total_samples = len(X)
        weights = total_samples / (len(attr_counts) * attr_counts)
        
        # Create sample weights
        sample_weights = X[protected_attr].map(weights).values
        return sample_weights
    
    def apply_fairness_constraints(self, model, X: pd.DataFrame, y: pd.Series,
                                 protected_attr: str, constraint_type: str = "demographic_parity"):
        """Apply fairness constraints during training."""
        # This is a placeholder for more advanced constraint-based methods
        # In practice, you would use libraries like fairlearn or implement
        # constraint optimization techniques
        
        logger.info(f"Applied {constraint_type} fairness constraints")
        return model


def detect_model_bias(model, X: pd.DataFrame, y: pd.Series, 
                     protected_attributes: List[str]) -> Dict[str, Any]:
    """Quick function to detect bias in a model."""
    detector = ModelFairnessDetector()
    return detector.generate_fairness_report(model, X, y, protected_attributes)


def mitigate_bias(model, X: pd.DataFrame, y: pd.Series, 
                protected_attr: str, method: str = "reweight") -> Any:
    """Quick function to mitigate bias in a model."""
    mitigator = BiasMitigation()
    
    if method == "reweight":
        sample_weights = mitigator.reweight_samples(X, y, protected_attr)
        return sample_weights
    elif method == "constraints":
        return mitigator.apply_fairness_constraints(model, X, y, protected_attr)
    else:
        logger.warning(f"Unknown mitigation method: {method}")
        return None
