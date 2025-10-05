"""
Advanced Ensemble Framework for igel.
Provides comprehensive ensemble learning capabilities beyond basic sklearn ensembles.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
import logging
import joblib
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class AdvancedEnsemble:
    """
    Advanced ensemble framework supporting multiple ensemble strategies.
    """
    
    def __init__(self, ensemble_type: str = "voting", problem_type: str = "classification"):
        """
        Initialize the ensemble framework.
        
        Args:
            ensemble_type: Type of ensemble ('voting', 'stacking', 'blending', 'bagging', 'boosting')
            problem_type: Type of problem ('classification' or 'regression')
        """
        self.ensemble_type = ensemble_type
        self.problem_type = problem_type
        self.base_models = []
        self.ensemble_model = None
        self.meta_model = None
        self.performance_metrics = {}
        self.training_history = []
        
    def add_base_model(self, model: BaseEstimator, name: str = None, weight: float = 1.0):
        """
        Add a base model to the ensemble.
        
        Args:
            model: The base model to add
            name: Name for the model (if None, uses model class name)
            weight: Weight for the model (for weighted voting)
        """
        if name is None:
            name = model.__class__.__name__
        
        self.base_models.append({
            'model': model,
            'name': name,
            'weight': weight
        })
        logger.info(f"Added {name} to ensemble")
    
    def create_ensemble(self, **kwargs):
        """
        Create the ensemble model based on the specified type.
        """
        if not self.base_models:
            raise ValueError("No base models added to ensemble")
        
        if self.ensemble_type == "voting":
            self._create_voting_ensemble(**kwargs)
        elif self.ensemble_type == "stacking":
            self._create_stacking_ensemble(**kwargs)
        elif self.ensemble_type == "blending":
            self._create_blending_ensemble(**kwargs)
        elif self.ensemble_type == "bagging":
            self._create_bagging_ensemble(**kwargs)
        elif self.ensemble_type == "boosting":
            self._create_boosting_ensemble(**kwargs)
        else:
            raise ValueError(f"Unsupported ensemble type: {self.ensemble_type}")
    
    def _create_voting_ensemble(self, voting_type: str = "soft", weights: List[float] = None):
        """Create a voting ensemble."""
        estimators = [(model['name'], model['model']) for model in self.base_models]
        
        if weights is None:
            weights = [model['weight'] for model in self.base_models]
        
        if self.problem_type == "classification":
            self.ensemble_model = VotingClassifier(
                estimators=estimators,
                voting=voting_type,
                weights=weights
            )
        else:
            self.ensemble_model = VotingRegressor(
                estimators=estimators,
                weights=weights
            )
    
    def _create_stacking_ensemble(self, meta_model=None, cv_folds: int = 5, **kwargs):
        """Create a stacking ensemble."""
        estimators = [(model['name'], model['model']) for model in self.base_models]
        
        if meta_model is None:
            if self.problem_type == "classification":
                meta_model = LogisticRegression()
            else:
                meta_model = LinearRegression()
        
        if self.problem_type == "classification":
            self.ensemble_model = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_model,
                cv=cv_folds,
                **kwargs
            )
        else:
            self.ensemble_model = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_model,
                cv=cv_folds,
                **kwargs
            )
    
    def _create_blending_ensemble(self, meta_model=None, validation_split: float = 0.2):
        """Create a blending ensemble (manual stacking)."""
        if meta_model is None:
            if self.problem_type == "classification":
                meta_model = LogisticRegression()
            else:
                meta_model = LinearRegression()
        
        self.meta_model = meta_model
        self.validation_split = validation_split
        logger.info("Blending ensemble created - requires manual training")
    
    def _create_bagging_ensemble(self, base_estimator=None, n_estimators: int = 10, **kwargs):
        """Create a bagging ensemble."""
        if base_estimator is None:
            if self.problem_type == "classification":
                base_estimator = DecisionTreeClassifier()
            else:
                base_estimator = DecisionTreeRegressor()
        
        if self.problem_type == "classification":
            self.ensemble_model = BaggingClassifier(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                **kwargs
            )
        else:
            self.ensemble_model = BaggingRegressor(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                **kwargs
            )
    
    def _create_boosting_ensemble(self, base_estimator=None, n_estimators: int = 50, **kwargs):
        """Create a boosting ensemble."""
        if base_estimator is None:
            if self.problem_type == "classification":
                base_estimator = DecisionTreeClassifier(max_depth=1)
            else:
                base_estimator = DecisionTreeRegressor(max_depth=1)
        
        if self.problem_type == "classification":
            self.ensemble_model = AdaBoostClassifier(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                **kwargs
            )
        else:
            self.ensemble_model = AdaBoostRegressor(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                **kwargs
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Train the ensemble model.
        """
        if self.ensemble_model is not None:
            # Standard ensemble training
            self.ensemble_model.fit(X, y, **kwargs)
            logger.info(f"Ensemble model trained successfully")
        
        elif self.ensemble_type == "blending":
            # Manual blending training
            self._train_blending_ensemble(X, y, **kwargs)
        
        # Evaluate individual models
        self._evaluate_base_models(X, y)
        
        # Store training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'ensemble_type': self.ensemble_type,
            'n_models': len(self.base_models),
            'problem_type': self.problem_type
        })
    
    def _train_blending_ensemble(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Train a blending ensemble manually."""
        from sklearn.model_selection import train_test_split
        
        # Split data for blending
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42
        )
        
        # Train base models on training set
        base_predictions = []
        for model_info in self.base_models:
            model = model_info['model']
            model.fit(X_train, y_train, **kwargs)
            
            # Get predictions on validation set
            if self.problem_type == "classification":
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_val)
                else:
                    pred = model.predict(X_val).reshape(-1, 1)
            else:
                pred = model.predict(X_val).reshape(-1, 1)
            
            base_predictions.append(pred)
        
        # Train meta-model on blended predictions
        X_blend = np.hstack(base_predictions)
        self.meta_model.fit(X_blend, y_val)
        
        logger.info("Blending ensemble trained successfully")
    
    def predict(self, X: np.ndarray):
        """Make predictions using the ensemble."""
        if self.ensemble_model is not None:
            return self.ensemble_model.predict(X)
        elif self.ensemble_type == "blending":
            return self._predict_blending(X)
        else:
            raise ValueError("Ensemble not properly initialized")
    
    def predict_proba(self, X: np.ndarray):
        """Make probability predictions (for classification)."""
        if self.problem_type != "classification":
            raise ValueError("predict_proba only available for classification")
        
        if self.ensemble_model is not None:
            if hasattr(self.ensemble_model, 'predict_proba'):
                return self.ensemble_model.predict_proba(X)
            else:
                # Convert predictions to probabilities
                predictions = self.ensemble_model.predict(X)
                return self._predictions_to_proba(predictions)
        elif self.ensemble_type == "blending":
            return self._predict_blending_proba(X)
        else:
            raise ValueError("Ensemble not properly initialized")
    
    def _predict_blending(self, X: np.ndarray):
        """Make predictions using blending ensemble."""
        base_predictions = []
        for model_info in self.base_models:
            model = model_info['model']
            if self.problem_type == "classification":
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                else:
                    pred = model.predict(X).reshape(-1, 1)
            else:
                pred = model.predict(X).reshape(-1, 1)
            base_predictions.append(pred)
        
        X_blend = np.hstack(base_predictions)
        return self.meta_model.predict(X_blend)
    
    def _predict_blending_proba(self, X: np.ndarray):
        """Make probability predictions using blending ensemble."""
        base_predictions = []
        for model_info in self.base_models:
            model = model_info['model']
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X).reshape(-1, 1)
            base_predictions.append(pred)
        
        X_blend = np.hstack(base_predictions)
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(X_blend)
        else:
            predictions = self.meta_model.predict(X_blend)
            return self._predictions_to_proba(predictions)
    
    def _predictions_to_proba(self, predictions: np.ndarray):
        """Convert predictions to probability format."""
        unique_classes = np.unique(predictions)
        n_classes = len(unique_classes)
        proba = np.zeros((len(predictions), n_classes))
        
        for i, pred in enumerate(predictions):
            class_idx = np.where(unique_classes == pred)[0][0]
            proba[i, class_idx] = 1.0
        
        return proba
    
    def _evaluate_base_models(self, X: np.ndarray, y: np.ndarray):
        """Evaluate individual base models."""
        cv_folds = 5
        if self.problem_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'neg_mean_squared_error'
        
        for model_info in self.base_models:
            model = model_info['model']
            name = model_info['name']
            
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                self.performance_metrics[name] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'scores': scores.tolist()
                }
                logger.info(f"{name} CV score: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
            except Exception as e:
                logger.warning(f"Could not evaluate {name}: {e}")
                self.performance_metrics[name] = {'error': str(e)}
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from the ensemble."""
        if self.ensemble_model is None:
            return {}
        
        importance_dict = {}
        
        if hasattr(self.ensemble_model, 'feature_importances_'):
            importance_dict['ensemble'] = self.ensemble_model.feature_importances_.tolist()
        
        # Get importance from individual models
        for model_info in self.base_models:
            model = model_info['model']
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_info['name']] = model.feature_importances_.tolist()
        
        return importance_dict
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get model weights for weighted ensembles."""
        if self.ensemble_type == "voting" and hasattr(self.ensemble_model, 'weights_'):
            return dict(zip([m['name'] for m in self.base_models], self.ensemble_model.weights_))
        else:
            return {model['name']: model['weight'] for model in self.base_models}
    
    def save_ensemble(self, filepath: str):
        """Save the ensemble model to disk."""
        ensemble_data = {
            'ensemble_type': self.ensemble_type,
            'problem_type': self.problem_type,
            'base_models': [],
            'performance_metrics': self.performance_metrics,
            'training_history': self.training_history
        }
        
        # Save base models
        for i, model_info in enumerate(self.base_models):
            model_path = f"{filepath}_base_model_{i}.joblib"
            joblib.dump(model_info['model'], model_path)
            ensemble_data['base_models'].append({
                'name': model_info['name'],
                'weight': model_info['weight'],
                'model_path': model_path
            })
        
        # Save ensemble model
        if self.ensemble_model is not None:
            ensemble_path = f"{filepath}_ensemble.joblib"
            joblib.dump(self.ensemble_model, ensemble_path)
            ensemble_data['ensemble_path'] = ensemble_path
        
        # Save meta model
        if self.meta_model is not None:
            meta_path = f"{filepath}_meta.joblib"
            joblib.dump(self.meta_model, meta_path)
            ensemble_data['meta_path'] = meta_path
        
        # Save ensemble metadata
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(ensemble_data, f, indent=2)
        
        logger.info(f"Ensemble saved to {filepath}")
    
    @classmethod
    def load_ensemble(cls, filepath: str):
        """Load an ensemble model from disk."""
        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            ensemble_data = json.load(f)
        
        # Create ensemble instance
        ensemble = cls(
            ensemble_type=ensemble_data['ensemble_type'],
            problem_type=ensemble_data['problem_type']
        )
        
        # Load base models
        for model_data in ensemble_data['base_models']:
            model = joblib.load(model_data['model_path'])
            ensemble.add_base_model(model, model_data['name'], model_data['weight'])
        
        # Load ensemble model
        if 'ensemble_path' in ensemble_data:
            ensemble.ensemble_model = joblib.load(ensemble_data['ensemble_path'])
        
        # Load meta model
        if 'meta_path' in ensemble_data:
            ensemble.meta_model = joblib.load(ensemble_data['meta_path'])
        
        # Restore metadata
        ensemble.performance_metrics = ensemble_data['performance_metrics']
        ensemble.training_history = ensemble_data['training_history']
        
        return ensemble
    
    def generate_report(self) -> str:
        """Generate a comprehensive report of the ensemble."""
        report = []
        report.append("Advanced Ensemble Framework Report")
        report.append("=" * 40)
        report.append(f"Ensemble Type: {self.ensemble_type}")
        report.append(f"Problem Type: {self.problem_type}")
        report.append(f"Number of Base Models: {len(self.base_models)}")
        report.append(f"Training Sessions: {len(self.training_history)}")
        
        report.append("\nBase Models Performance:")
        report.append("-" * 30)
        for name, metrics in self.performance_metrics.items():
            if 'error' not in metrics:
                report.append(f"{name}: {metrics['mean_score']:.4f} (+/- {metrics['std_score'] * 2:.4f})")
            else:
                report.append(f"{name}: Error - {metrics['error']}")
        
        if self.ensemble_model is not None:
            report.append(f"\nEnsemble Model: {self.ensemble_model.__class__.__name__}")
        
        if self.meta_model is not None:
            report.append(f"Meta Model: {self.meta_model.__class__.__name__}")
        
        return "\n".join(report)


class EnsembleBuilder:
    """
    Builder class for creating ensembles with predefined configurations.
    """
    
    @staticmethod
    def create_classification_ensemble(ensemble_type: str = "voting") -> AdvancedEnsemble:
        """Create a classification ensemble with common models."""
        ensemble = AdvancedEnsemble(ensemble_type=ensemble_type, problem_type="classification")
        
        # Add common classification models
        ensemble.add_base_model(RandomForestClassifier(n_estimators=100, random_state=42), "RandomForest")
        ensemble.add_base_model(SVC(probability=True, random_state=42), "SVM")
        ensemble.add_base_model(KNeighborsClassifier(), "KNN")
        ensemble.add_base_model(GaussianNB(), "NaiveBayes")
        ensemble.add_base_model(DecisionTreeClassifier(random_state=42), "DecisionTree")
        
        return ensemble
    
    @staticmethod
    def create_regression_ensemble(ensemble_type: str = "voting") -> AdvancedEnsemble:
        """Create a regression ensemble with common models."""
        ensemble = AdvancedEnsemble(ensemble_type=ensemble_type, problem_type="regression")
        
        # Add common regression models
        ensemble.add_base_model(RandomForestRegressor(n_estimators=100, random_state=42), "RandomForest")
        ensemble.add_base_model(SVR(), "SVR")
        ensemble.add_base_model(KNeighborsRegressor(), "KNN")
        ensemble.add_base_model(DecisionTreeRegressor(random_state=42), "DecisionTree")
        
        return ensemble
    
    @staticmethod
    def create_auto_ensemble(X: np.ndarray, y: np.ndarray, problem_type: str = "classification") -> AdvancedEnsemble:
        """Automatically create the best ensemble based on data characteristics."""
        # Simple heuristic-based ensemble selection
        n_samples, n_features = X.shape
        
        if n_samples < 1000:
            ensemble_type = "voting"
        elif n_features > 100:
            ensemble_type = "stacking"
        else:
            ensemble_type = "blending"
        
        if problem_type == "classification":
            ensemble = EnsembleBuilder.create_classification_ensemble(ensemble_type)
        else:
            ensemble = EnsembleBuilder.create_regression_ensemble(ensemble_type)
        
        return ensemble
