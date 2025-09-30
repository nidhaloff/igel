"""
Model Ensemble and Stacking Framework for Igel.

This module provides ensemble learning and stacking capabilities.
Addresses GitHub issue #340 - Create Model Ensemble and Stacking Framework.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ModelEnsemble:
    """Create and manage model ensembles."""
    
    def __init__(self, ensemble_type: str = "voting"):
        self.ensemble_type = ensemble_type
        self.ensemble_model = None
        self.base_models = []
        self.meta_model = None
    
    def add_base_model(self, name: str, model: BaseEstimator):
        """Add a base model to the ensemble."""
        self.base_models.append((name, model))
        logger.info(f"Added base model: {name}")
    
    def create_voting_ensemble(self, voting: str = "soft", weights: Optional[List[float]] = None):
        """Create a voting ensemble."""
        if len(self.base_models) < 2:
            raise ValueError("At least 2 base models required for ensemble")
        
        # Determine if classification or regression based on first model
        model_type = self.base_models[0][1].__class__.__name__
        
        if "Classifier" in model_type:
            self.ensemble_model = VotingClassifier(
                estimators=self.base_models,
                voting=voting,
                weights=weights
            )
        else:
            self.ensemble_model = VotingRegressor(
                estimators=self.base_models,
                weights=weights
            )
        
        logger.info(f"Created voting ensemble with {len(self.base_models)} models")
        return self.ensemble_model
    
    def create_stacking_ensemble(self, meta_model: BaseEstimator, cv: int = 5):
        """Create a stacking ensemble."""
        if len(self.base_models) < 2:
            raise ValueError("At least 2 base models required for stacking")
        
        self.meta_model = meta_model
        
        # Determine if classification or regression
        model_type = self.base_models[0][1].__class__.__name__
        
        if "Classifier" in model_type:
            self.ensemble_model = StackingClassifier(
                estimators=self.base_models,
                final_estimator=meta_model,
                cv=cv
            )
        else:
            self.ensemble_model = StackingRegressor(
                estimators=self.base_models,
                final_estimator=meta_model,
                cv=cv
            )
        
        logger.info(f"Created stacking ensemble with {len(self.base_models)} base models and meta model")
        return self.ensemble_model
    
    def fit(self, X, y):
        """Fit the ensemble model."""
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not created. Call create_voting_ensemble() or create_stacking_ensemble()")
        
        self.ensemble_model.fit(X, y)
        logger.info("Ensemble model fitted successfully")
    
    def predict(self, X):
        """Make predictions with the ensemble."""
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not fitted")
        
        return self.ensemble_model.predict(X)


class ModelBlending:
    """Blend predictions from multiple models."""
    
    def __init__(self):
        self.models = []
        self.weights = None
    
    def add_model(self, model: BaseEstimator, weight: float = 1.0):
        """Add a model with its weight for blending."""
        self.models.append({'model': model, 'weight': weight})
        logger.info(f"Added model to blending with weight: {weight}")
    
    def fit_all(self, X, y):
        """Fit all models."""
        for model_info in self.models:
            model_info['model'].fit(X, y)
        logger.info(f"Fitted {len(self.models)} models for blending")
    
    def predict(self, X):
        """Blend predictions from all models."""
        if not self.models:
            raise ValueError("No models added for blending")
        
        # Get predictions from all models
        predictions = []
        weights = []
        
        for model_info in self.models:
            pred = model_info['model'].predict(X)
            predictions.append(pred)
            weights.append(model_info['weight'])
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Weighted average of predictions
        predictions = np.array(predictions)
        blended = np.average(predictions, axis=0, weights=weights)
        
        return blended


def create_ensemble(models: List[tuple], ensemble_type: str = "voting", 
                   meta_model: Optional[BaseEstimator] = None) -> ModelEnsemble:
    """Quick function to create an ensemble."""
    ensemble = ModelEnsemble(ensemble_type)
    
    for name, model in models:
        ensemble.add_base_model(name, model)
    
    if ensemble_type == "voting":
        ensemble.create_voting_ensemble()
    elif ensemble_type == "stacking" and meta_model:
        ensemble.create_stacking_ensemble(meta_model)
    
    return ensemble
