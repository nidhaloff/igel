"""
Model Comparison Leaderboard for igel.
Provides ranking and comparison of multiple trained models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score
import joblib
import json
import logging
from datetime import datetime
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelLeaderboard:
    """
    Leaderboard system for comparing and ranking machine learning models.
    """
    
    def __init__(self, leaderboard_name: str = "default"):
        """
        Initialize the model leaderboard.
        
        Args:
            leaderboard_name: Name of the leaderboard
        """
        self.leaderboard_name = leaderboard_name
        self.models = {}
        self.rankings = {}
        self.metrics_history = []
        
    def add_model(self, model_id: str, model_path: str, model_name: str = None, 
                  metadata: Dict[str, Any] = None) -> bool:
        """
        Add a model to the leaderboard.
        
        Args:
            model_id: Unique identifier for the model
            model_path: Path to the saved model file
            model_name: Human-readable name for the model
            metadata: Additional metadata about the model
            
        Returns:
            True if model was added successfully
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            model_info = {
                'model_id': model_id,
                'model_path': model_path,
                'model_name': model_name or model_id,
                'metadata': metadata or {},
                'added_date': datetime.now().isoformat(),
                'metrics': {}
            }
            
            self.models[model_id] = model_info
            logger.info(f"Added model {model_id} to leaderboard")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add model {model_id}: {e}")
            return False
    
    def evaluate_model(self, model_id: str, X_test: np.ndarray, y_test: np.ndarray, 
                      problem_type: str = "classification") -> Dict[str, float]:
        """
        Evaluate a model and update its metrics.
        
        Args:
            model_id: ID of the model to evaluate
            X_test: Test features
            y_test: Test labels
            problem_type: Type of problem ('classification' or 'regression')
            
        Returns:
            Dictionary of evaluation metrics
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in leaderboard")
        
        try:
            # Load model
            model = joblib.load(self.models[model_id]['model_path'])
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Calculate metrics based on problem type
            metrics = {}
            if problem_type == "classification":
                metrics['accuracy'] = accuracy_score(y_test, predictions)
                metrics['f1_score'] = f1_score(y_test, predictions, average='weighted')
                
                # Add probability-based metrics if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_test)
                    metrics['log_loss'] = self._calculate_log_loss(y_test, probabilities)
            else:
                metrics['mse'] = mean_squared_error(y_test, predictions)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['r2'] = r2_score(y_test, predictions)
                metrics['mae'] = np.mean(np.abs(y_test - predictions))
            
            # Update model metrics
            self.models[model_id]['metrics'] = metrics
            self.models[model_id]['last_evaluated'] = datetime.now().isoformat()
            
            # Store evaluation history
            evaluation_record = {
                'model_id': model_id,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'test_size': len(X_test)
            }
            self.metrics_history.append(evaluation_record)
            
            logger.info(f"Evaluated model {model_id}: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate model {model_id}: {e}")
            return {}
    
    def _calculate_log_loss(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate log loss for classification."""
        from sklearn.metrics import log_loss
        try:
            return log_loss(y_true, y_prob)
        except:
            return float('inf')
    
    def rank_models(self, metric: str = None, ascending: bool = None) -> List[Dict[str, Any]]:
        """
        Rank models based on their performance metrics.
        
        Args:
            metric: Specific metric to rank by (if None, uses composite score)
            ascending: Whether to sort in ascending order (None for auto-detect)
            
        Returns:
            List of ranked models with their scores
        """
        if not self.models:
            return []
        
        rankings = []
        
        for model_id, model_info in self.models.items():
            if not model_info['metrics']:
                continue
            
            metrics = model_info['metrics']
            
            if metric and metric in metrics:
                # Rank by specific metric
                score = metrics[metric]
                if ascending is None:
                    # Auto-detect: higher is better for accuracy, r2, f1; lower is better for mse, rmse, mae
                    ascending = metric in ['mse', 'rmse', 'mae', 'log_loss']
            else:
                # Use composite score
                score = self._calculate_composite_score(metrics)
                ascending = False  # Higher composite score is better
            
            rankings.append({
                'model_id': model_id,
                'model_name': model_info['model_name'],
                'score': score,
                'metrics': metrics,
                'rank': 0  # Will be set after sorting
            })
        
        # Sort by score
        rankings.sort(key=lambda x: x['score'], reverse=not ascending)
        
        # Assign ranks
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        self.rankings = rankings
        return rankings
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate a composite score from multiple metrics."""
        score = 0.0
        weight_sum = 0.0
        
        # Weight different metrics
        metric_weights = {
            'accuracy': 0.3,
            'f1_score': 0.2,
            'r2': 0.3,
            'mse': -0.1,  # Negative weight (lower is better)
            'rmse': -0.1,
            'mae': -0.1,
            'log_loss': -0.1
        }
        
        for metric, value in metrics.items():
            if metric in metric_weights:
                weight = metric_weights[metric]
                score += weight * value
                weight_sum += abs(weight)
        
        # Normalize by total weight
        if weight_sum > 0:
            score = score / weight_sum
        
        return score
    
    def get_leaderboard_table(self, top_n: int = None) -> pd.DataFrame:
        """
        Get leaderboard as a pandas DataFrame.
        
        Args:
            top_n: Number of top models to include (None for all)
            
        Returns:
            DataFrame with model rankings and metrics
        """
        if not self.rankings:
            self.rank_models()
        
        rankings = self.rankings[:top_n] if top_n else self.rankings
        
        if not rankings:
            return pd.DataFrame()
        
        # Create DataFrame
        data = []
        for ranking in rankings:
            row = {
                'Rank': ranking['rank'],
                'Model ID': ranking['model_id'],
                'Model Name': ranking['model_name'],
                'Score': ranking['score']
            }
            
            # Add individual metrics
            for metric, value in ranking['metrics'].items():
                row[metric.title()] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_model_comparison(self, model_ids: List[str]) -> Dict[str, Any]:
        """
        Compare specific models side by side.
        
        Args:
            model_ids: List of model IDs to compare
            
        Returns:
            Comparison results
        """
        comparison = {
            'models': [],
            'metrics_comparison': {},
            'best_per_metric': {}
        }
        
        for model_id in model_ids:
            if model_id in self.models:
                model_info = self.models[model_id]
                comparison['models'].append({
                    'model_id': model_id,
                    'model_name': model_info['model_name'],
                    'metrics': model_info['metrics']
                })
        
        # Compare metrics across models
        all_metrics = set()
        for model in comparison['models']:
            all_metrics.update(model['metrics'].keys())
        
        for metric in all_metrics:
            metric_values = []
            for model in comparison['models']:
                if metric in model['metrics']:
                    metric_values.append((model['model_id'], model['metrics'][metric]))
            
            if metric_values:
                # Sort by metric value
                ascending = metric in ['mse', 'rmse', 'mae', 'log_loss']
                metric_values.sort(key=lambda x: x[1], reverse=not ascending)
                
                comparison['metrics_comparison'][metric] = metric_values
                comparison['best_per_metric'][metric] = metric_values[0][0]
        
        return comparison
    
    def generate_leaderboard_report(self) -> str:
        """Generate a comprehensive leaderboard report."""
        report = []
        report.append(f"Model Leaderboard: {self.leaderboard_name}")
        report.append("=" * 40)
        report.append(f"Total Models: {len(self.models)}")
        report.append(f"Evaluated Models: {len([m for m in self.models.values() if m['metrics']])}")
        
        if self.rankings:
            report.append(f"\nTop 5 Models:")
            report.append("-" * 15)
            
            for i, ranking in enumerate(self.rankings[:5]):
                report.append(f"{i+1}. {ranking['model_name']} (Score: {ranking['score']:.4f})")
                for metric, value in ranking['metrics'].items():
                    report.append(f"   {metric}: {value:.4f}")
                report.append("")
        
        return "\n".join(report)
    
    def save_leaderboard(self, filepath: str):
        """Save leaderboard data to file."""
        leaderboard_data = {
            'leaderboard_name': self.leaderboard_name,
            'models': self.models,
            'rankings': self.rankings,
            'metrics_history': self.metrics_history,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{filepath}_leaderboard.json", 'w') as f:
            json.dump(leaderboard_data, f, indent=2, default=str)
        
        logger.info(f"Leaderboard saved to {filepath}_leaderboard.json")
    
    @classmethod
    def load_leaderboard(cls, filepath: str):
        """Load leaderboard from file."""
        with open(f"{filepath}_leaderboard.json", 'r') as f:
            data = json.load(f)
        
        leaderboard = cls(data['leaderboard_name'])
        leaderboard.models = data['models']
        leaderboard.rankings = data['rankings']
        leaderboard.metrics_history = data['metrics_history']
        
        return leaderboard
