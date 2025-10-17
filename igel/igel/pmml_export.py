"""
PMML Export functionality for igel.
Export trained models to PMML format.
"""

import joblib
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PMMLExporter:
    """
    Export sklearn models to PMML format.
    """
    
    def __init__(self):
        """Initialize PMML exporter."""
        self.supported_models = [
            'RandomForestClassifier', 'RandomForestRegressor',
            'LogisticRegression', 'LinearRegression',
            'DecisionTreeClassifier', 'DecisionTreeRegressor'
        ]
    
    def export_to_pmml(self, model, output_path: str, model_name: str = "Model") -> bool:
        """
        Export model to PMML format.
        
        Args:
            model: Trained sklearn model
            output_path: Path to save PMML file
            model_name: Name for the model in PMML
            
        Returns:
            True if export successful
        """
        try:
            from sklearn2pmml import sklearn2pmml
            from sklearn2pmml.pipeline import PMMLPipeline
            
            # Create PMML pipeline
            pipeline = PMMLPipeline([("model", model)])
            
            # Export to PMML
            sklearn2pmml(pipeline, output_path, with_repr=True)
            
            logger.info(f"Model exported to PMML: {output_path}")
            return True
            
        except ImportError:
            logger.error("sklearn2pmml not available. Install with: pip install sklearn2pmml")
            return False
        except Exception as e:
            logger.error(f"PMML export failed: {e}")
            return False
    
    def is_model_supported(self, model) -> bool:
        """Check if model is supported for PMML export."""
        model_type = type(model).__name__
        return model_type in self.supported_models
    
    def get_supported_models(self) -> list:
        """Get list of supported model types."""
        return self.supported_models.copy()
