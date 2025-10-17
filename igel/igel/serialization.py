"""
Efficient model serialization for igel.
Support for multiple serialization formats.
"""

import joblib
import pickle
import json
import numpy as np
from typing import Any, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)

class ModelSerializer:
    """
    Efficient model serialization with multiple format support.
    """
    
    def __init__(self):
        """Initialize model serializer."""
        self.supported_formats = ['joblib', 'pickle', 'json', 'numpy']
        
    def save_model(self, model: Any, path: str, format: str = 'joblib', 
                   compress: bool = True) -> bool:
        """
        Save model in specified format.
        
        Args:
            model: Model to save
            path: Path to save model
            format: Serialization format
            compress: Whether to compress the file
            
        Returns:
            True if successful
        """
        try:
            if format == 'joblib':
                if compress:
                    joblib.dump(model, path, compress=3)
                else:
                    joblib.dump(model, path)
                    
            elif format == 'pickle':
                with open(path, 'wb') as f:
                    if compress:
                        import gzip
                        with gzip.open(path + '.gz', 'wb') as gz:
                            pickle.dump(model, gz)
                    else:
                        pickle.dump(model, f)
                        
            elif format == 'json':
                # For simple models that can be JSON serialized
                model_dict = self._model_to_dict(model)
                with open(path, 'w') as f:
                    json.dump(model_dict, f, indent=2)
                    
            elif format == 'numpy':
                # Save as numpy array if possible
                if hasattr(model, 'coef_'):
                    np.save(path, model.coef_)
                else:
                    raise ValueError("Model cannot be saved as numpy array")
            
            logger.info(f"Model saved successfully: {path} ({format})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, path: str, format: str = 'joblib') -> Any:
        """
        Load model from specified format.
        
        Args:
            path: Path to model file
            format: Serialization format
            
        Returns:
            Loaded model
        """
        try:
            if format == 'joblib':
                return joblib.load(path)
                
            elif format == 'pickle':
                if path.endswith('.gz'):
                    import gzip
                    with gzip.open(path, 'rb') as f:
                        return pickle.load(f)
                else:
                    with open(path, 'rb') as f:
                        return pickle.load(f)
                        
            elif format == 'json':
                with open(path, 'r') as f:
                    model_dict = json.load(f)
                return self._dict_to_model(model_dict)
                
            elif format == 'numpy':
                return np.load(path)
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _model_to_dict(self, model: Any) -> Dict:
        """Convert model to dictionary for JSON serialization."""
        model_dict = {
            'type': type(model).__name__,
            'params': model.get_params() if hasattr(model, 'get_params') else {}
        }
        
        # Add model-specific attributes
        if hasattr(model, 'coef_'):
            model_dict['coef_'] = model.coef_.tolist()
        if hasattr(model, 'intercept_'):
            model_dict['intercept_'] = model.intercept_.tolist()
        if hasattr(model, 'feature_importances_'):
            model_dict['feature_importances_'] = model.feature_importances_.tolist()
            
        return model_dict
    
    def _dict_to_model(self, model_dict: Dict) -> Any:
        """Convert dictionary back to model."""
        # This is a simplified implementation
        # In practice, you'd need to handle different model types
        model_type = model_dict.get('type')
        params = model_dict.get('params', {})
        
        # Import appropriate model class
        if model_type == 'LinearRegression':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression(**params)
            if 'coef_' in model_dict:
                model.coef_ = np.array(model_dict['coef_'])
            if 'intercept_' in model_dict:
                model.intercept_ = np.array(model_dict['intercept_'])
            return model
        
        # Add more model types as needed
        raise ValueError(f"Unsupported model type: {model_type}")
    
    def get_file_size(self, path: str) -> int:
        """Get file size in bytes."""
        return os.path.getsize(path)
    
    def compare_formats(self, model: Any, base_path: str) -> Dict[str, Dict]:
        """
        Compare different serialization formats.
        
        Args:
            model: Model to serialize
            base_path: Base path for saving files
            
        Returns:
            Comparison results
        """
        results = {}
        
        for fmt in self.supported_formats:
            try:
                path = f"{base_path}.{fmt}"
                success = self.save_model(model, path, fmt)
                
                if success:
                    size = self.get_file_size(path)
                    results[fmt] = {
                        'success': True,
                        'size_bytes': size,
                        'size_mb': size / (1024 * 1024)
                    }
                else:
                    results[fmt] = {'success': False}
                    
            except Exception as e:
                results[fmt] = {'success': False, 'error': str(e)}
        
        return results
