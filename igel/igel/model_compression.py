"""
Model Compression and Optimization Framework for igel.
Provides various techniques for compressing and optimizing machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ModelCompressor:
    """
    Framework for compressing and optimizing machine learning models.
    """
    
    def __init__(self, compression_method: str = "pruning", target_compression_ratio: float = 0.5):
        """
        Initialize the model compressor.
        
        Args:
            compression_method: Method to use ('pruning', 'quantization', 'knowledge_distillation', 'feature_selection')
            target_compression_ratio: Target compression ratio (0.0 to 1.0)
        """
        self.compression_method = compression_method
        self.target_compression_ratio = target_compression_ratio
        self.original_model = None
        self.compressed_model = None
        self.compression_stats = {}
        self.performance_comparison = {}
        
    def compress_model(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray, 
                      validation_data: Optional[Tuple] = None, **kwargs) -> BaseEstimator:
        """
        Compress a model using the specified method.
        
        Args:
            model: The model to compress
            X: Training features
            y: Training labels
            validation_data: Optional validation data (X_val, y_val)
            **kwargs: Additional parameters for compression
            
        Returns:
            Compressed model
        """
        self.original_model = model
        original_size = self._get_model_size(model)
        
        if self.compression_method == "pruning":
            self.compressed_model = self._prune_model(model, X, y, **kwargs)
        elif self.compression_method == "quantization":
            self.compressed_model = self._quantize_model(model, X, y, **kwargs)
        elif self.compression_method == "knowledge_distillation":
            self.compressed_model = self._distill_model(model, X, y, **kwargs)
        elif self.compression_method == "feature_selection":
            self.compressed_model = self._select_features(model, X, y, **kwargs)
        else:
            raise ValueError(f"Unsupported compression method: {self.compression_method}")
        
        compressed_size = self._get_model_size(self.compressed_model)
        compression_ratio = compressed_size / original_size
        
        self.compression_stats = {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'size_reduction': (original_size - compressed_size) / original_size,
            'method': self.compression_method
        }
        
        # Compare performance if validation data provided
        if validation_data is not None:
            self._compare_performance(validation_data)
        
        logger.info(f"Model compressed using {self.compression_method}. "
                   f"Size reduction: {self.compression_stats['size_reduction']:.2%}")
        
        return self.compressed_model
    
    def _prune_model(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray, **kwargs) -> BaseEstimator:
        """Prune a model to reduce complexity."""
        if hasattr(model, 'tree_') and hasattr(model.tree_, 'prune'):
            # For decision trees, use built-in pruning
            pruned_model = model.__class__(**model.get_params())
            pruned_model.fit(X, y)
            return pruned_model
        elif hasattr(model, 'estimators_'):
            # For ensemble models, reduce number of estimators
            n_estimators = max(1, int(len(model.estimators_) * self.target_compression_ratio))
            pruned_model = model.__class__(n_estimators=n_estimators, **model.get_params())
            pruned_model.fit(X, y)
            return pruned_model
        else:
            # For other models, create a simpler version
            return self._create_simpler_model(model, X, y, **kwargs)
    
    def _quantize_model(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray, **kwargs) -> BaseEstimator:
        """Quantize model parameters to reduce precision."""
        # Create a copy of the model
        quantized_model = model.__class__(**model.get_params())
        quantized_model.fit(X, y)
        
        # Quantize parameters if possible
        if hasattr(quantized_model, 'coef_'):
            # For linear models, quantize coefficients
            precision = kwargs.get('precision', 8)  # Number of bits
            max_val = np.max(np.abs(quantized_model.coef_))
            scale = (2 ** (precision - 1) - 1) / max_val
            quantized_model.coef_ = np.round(quantized_model.coef_ * scale) / scale
        
        return quantized_model
    
    def _distill_model(self, teacher_model: BaseEstimator, X: np.ndarray, y: np.ndarray, **kwargs) -> BaseEstimator:
        """Create a smaller student model using knowledge distillation."""
        # Create a simpler student model
        if isinstance(teacher_model, (RandomForestClassifier, RandomForestRegressor)):
            student_model = DecisionTreeClassifier() if isinstance(teacher_model, RandomForestClassifier) else DecisionTreeRegressor()
        elif isinstance(teacher_model, (MLPClassifier, MLPRegressor)):
            # Create a smaller neural network
            hidden_layers = kwargs.get('student_hidden_layers', (50,))
            if isinstance(teacher_model, MLPClassifier):
                student_model = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=1000)
            else:
                student_model = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=1000)
        else:
            # Default to linear model
            if isinstance(teacher_model, ClassifierMixin):
                student_model = LogisticRegression()
            else:
                student_model = LinearRegression()
        
        # Get teacher predictions
        if hasattr(teacher_model, 'predict_proba') and isinstance(teacher_model, ClassifierMixin):
            teacher_predictions = teacher_model.predict_proba(X)
        else:
            teacher_predictions = teacher_model.predict(X)
        
        # Train student model on teacher predictions
        student_model.fit(X, teacher_predictions)
        
        return student_model
    
    def _select_features(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray, **kwargs) -> BaseEstimator:
        """Select most important features to reduce model complexity."""
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
        from sklearn.pipeline import Pipeline
        
        # Determine number of features to select
        n_features = max(1, int(X.shape[1] * self.target_compression_ratio))
        
        # Choose scoring function based on problem type
        if isinstance(model, ClassifierMixin):
            scoring_func = f_classif
        else:
            scoring_func = f_regression
        
        # Create feature selector
        selector = SelectKBest(score_func=scoring_func, k=n_features)
        
        # Create pipeline with feature selection
        compressed_model = Pipeline([
            ('feature_selection', selector),
            ('model', model)
        ])
        
        compressed_model.fit(X, y)
        
        return compressed_model
    
    def _create_simpler_model(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray, **kwargs) -> BaseEstimator:
        """Create a simpler version of the model."""
        if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            # Reduce number of trees and depth
            n_estimators = max(1, int(model.n_estimators * self.target_compression_ratio))
            max_depth = kwargs.get('max_depth', 3)
            
            if isinstance(model, RandomForestClassifier):
                simpler_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            else:
                simpler_model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
        elif isinstance(model, (MLPClassifier, MLPRegressor)):
            # Reduce hidden layer size
            original_layers = model.hidden_layer_sizes
            if isinstance(original_layers, int):
                new_layers = max(1, int(original_layers * self.target_compression_ratio))
            else:
                new_layers = tuple(max(1, int(size * self.target_compression_ratio)) for size in original_layers)
            
            if isinstance(model, MLPClassifier):
                simpler_model = MLPClassifier(hidden_layer_sizes=new_layers, max_iter=1000)
            else:
                simpler_model = MLPRegressor(hidden_layer_sizes=new_layers, max_iter=1000)
        else:
            # For other models, try to reduce complexity parameters
            simpler_model = model.__class__(**model.get_params())
        
        simpler_model.fit(X, y)
        return simpler_model
    
    def _get_model_size(self, model: BaseEstimator) -> int:
        """Estimate model size in bytes."""
        try:
            # Save model to temporary file and get size
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                joblib.dump(model, tmp_file.name)
                size = os.path.getsize(tmp_file.name)
                os.unlink(tmp_file.name)
                return size
        except:
            # Fallback: estimate based on model type
            if hasattr(model, 'n_estimators'):
                return model.n_estimators * 1000  # Rough estimate
            elif hasattr(model, 'coef_'):
                return model.coef_.nbytes
            else:
                return 10000  # Default estimate
    
    def _compare_performance(self, validation_data: Tuple):
        """Compare performance between original and compressed models."""
        X_val, y_val = validation_data
        
        # Original model performance
        orig_pred = self.original_model.predict(X_val)
        if isinstance(self.original_model, ClassifierMixin):
            orig_score = accuracy_score(y_val, orig_pred)
        else:
            orig_score = r2_score(y_val, orig_pred)
        
        # Compressed model performance
        comp_pred = self.compressed_model.predict(X_val)
        if isinstance(self.compressed_model, ClassifierMixin):
            comp_score = accuracy_score(y_val, comp_pred)
        else:
            comp_score = r2_score(y_val, comp_pred)
        
        self.performance_comparison = {
            'original_score': orig_score,
            'compressed_score': comp_score,
            'performance_drop': orig_score - comp_score,
            'relative_performance': comp_score / orig_score if orig_score != 0 else 0
        }
    
    def get_compression_report(self) -> str:
        """Generate a compression report."""
        report = []
        report.append("Model Compression Report")
        report.append("=" * 25)
        report.append(f"Compression Method: {self.compression_method}")
        report.append(f"Original Size: {self.compression_stats['original_size']:,} bytes")
        report.append(f"Compressed Size: {self.compression_stats['compressed_size']:,} bytes")
        report.append(f"Compression Ratio: {self.compression_stats['compression_ratio']:.2%}")
        report.append(f"Size Reduction: {self.compression_stats['size_reduction']:.2%}")
        
        if self.performance_comparison:
            report.append("\nPerformance Comparison:")
            report.append("-" * 22)
            report.append(f"Original Score: {self.performance_comparison['original_score']:.4f}")
            report.append(f"Compressed Score: {self.performance_comparison['compressed_score']:.4f}")
            report.append(f"Performance Drop: {self.performance_comparison['performance_drop']:.4f}")
            report.append(f"Relative Performance: {self.performance_comparison['relative_performance']:.2%}")
        
        return "\n".join(report)
    
    def save_compressed_model(self, filepath: str):
        """Save the compressed model and compression statistics."""
        # Save compressed model
        joblib.dump(self.compressed_model, f"{filepath}_compressed.joblib")
        
        # Save compression statistics
        compression_data = {
            'compression_stats': self.compression_stats,
            'performance_comparison': self.performance_comparison,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{filepath}_compression_stats.json", 'w') as f:
            json.dump(compression_data, f, indent=2)
        
        logger.info(f"Compressed model saved to {filepath}")


class ModelOptimizer:
    """
    Framework for optimizing model performance and efficiency.
    """
    
    def __init__(self, optimization_goal: str = "accuracy"):
        """
        Initialize the model optimizer.
        
        Args:
            optimization_goal: Goal to optimize for ('accuracy', 'speed', 'memory', 'balanced')
        """
        self.optimization_goal = optimization_goal
        self.optimization_results = {}
        
    def optimize_model(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray, 
                      validation_data: Optional[Tuple] = None, **kwargs) -> BaseEstimator:
        """
        Optimize a model based on the specified goal.
        
        Args:
            model: The model to optimize
            X: Training features
            y: Training labels
            validation_data: Optional validation data
            **kwargs: Additional optimization parameters
            
        Returns:
            Optimized model
        """
        if self.optimization_goal == "accuracy":
            return self._optimize_for_accuracy(model, X, y, validation_data, **kwargs)
        elif self.optimization_goal == "speed":
            return self._optimize_for_speed(model, X, y, **kwargs)
        elif self.optimization_goal == "memory":
            return self._optimize_for_memory(model, X, y, **kwargs)
        elif self.optimization_goal == "balanced":
            return self._optimize_balanced(model, X, y, validation_data, **kwargs)
        else:
            raise ValueError(f"Unsupported optimization goal: {self.optimization_goal}")
    
    def _optimize_for_accuracy(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray, 
                              validation_data: Optional[Tuple], **kwargs) -> BaseEstimator:
        """Optimize model for maximum accuracy."""
        from sklearn.model_selection import GridSearchCV
        
        # Define parameter grid based on model type
        param_grid = self._get_accuracy_param_grid(model)
        
        if validation_data is not None:
            X_val, y_val = validation_data
            scoring = 'accuracy' if isinstance(model, ClassifierMixin) else 'r2'
        else:
            scoring = 'accuracy' if isinstance(model, ClassifierMixin) else 'r2'
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring=scoring, n_jobs=-1
        )
        grid_search.fit(X, y)
        
        self.optimization_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'optimization_goal': self.optimization_goal
        }
        
        return grid_search.best_estimator_
    
    def _optimize_for_speed(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray, **kwargs) -> BaseEstimator:
        """Optimize model for maximum speed."""
        # Create a faster version of the model
        if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            # Reduce number of trees and use parallel processing
            n_estimators = min(10, model.n_estimators)
            max_depth = 5
            
            if isinstance(model, RandomForestClassifier):
                optimized_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    n_jobs=-1,
                    random_state=42
                )
            else:
                optimized_model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    n_jobs=-1,
                    random_state=42
                )
        else:
            # For other models, use default parameters optimized for speed
            optimized_model = model.__class__(**model.get_params())
        
        optimized_model.fit(X, y)
        
        self.optimization_results = {
            'optimization_goal': self.optimization_goal,
            'speed_optimizations': 'Reduced complexity, parallel processing enabled'
        }
        
        return optimized_model
    
    def _optimize_for_memory(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray, **kwargs) -> BaseEstimator:
        """Optimize model for minimum memory usage."""
        # Create a memory-efficient version
        if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            # Use fewer trees and smaller depth
            n_estimators = max(1, model.n_estimators // 4)
            max_depth = 3
            
            if isinstance(model, RandomForestClassifier):
                optimized_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            else:
                optimized_model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
        else:
            # For other models, use simpler parameters
            optimized_model = model.__class__(**model.get_params())
        
        optimized_model.fit(X, y)
        
        self.optimization_results = {
            'optimization_goal': self.optimization_goal,
            'memory_optimizations': 'Reduced model complexity for lower memory usage'
        }
        
        return optimized_model
    
    def _optimize_balanced(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray, 
                          validation_data: Optional[Tuple], **kwargs) -> BaseEstimator:
        """Optimize model for balanced performance across all metrics."""
        # Use moderate complexity for balanced performance
        if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            n_estimators = max(10, model.n_estimators // 2)
            max_depth = 8
            
            if isinstance(model, RandomForestClassifier):
                optimized_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            else:
                optimized_model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
        else:
            optimized_model = model.__class__(**model.get_params())
        
        optimized_model.fit(X, y)
        
        self.optimization_results = {
            'optimization_goal': self.optimization_goal,
            'balanced_optimizations': 'Balanced complexity for good performance across all metrics'
        }
        
        return optimized_model
    
    def _get_accuracy_param_grid(self, model: BaseEstimator) -> Dict:
        """Get parameter grid for accuracy optimization."""
        if isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        elif isinstance(model, (MLPClassifier, MLPRegressor)):
            return {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        else:
            return {}
    
    def get_optimization_report(self) -> str:
        """Generate an optimization report."""
        report = []
        report.append("Model Optimization Report")
        report.append("=" * 28)
        report.append(f"Optimization Goal: {self.optimization_goal}")
        
        if 'best_params' in self.optimization_results:
            report.append(f"Best Parameters: {self.optimization_results['best_params']}")
            report.append(f"Best Score: {self.optimization_results['best_score']:.4f}")
        
        if 'speed_optimizations' in self.optimization_results:
            report.append(f"Optimizations: {self.optimization_results['speed_optimizations']}")
        
        return "\n".join(report)
