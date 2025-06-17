"""
Model Registry System for Igel.

This module provides functionality for storing, versioning, and managing trained models.
"""

import os
import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil
import hashlib

class ModelRegistry:
    """A class to manage model versions and metadata."""
    
    def __init__(self, registry_path: str = "model_registry"):
        """
        Initialize the model registry.
        
        Args:
            registry_path (str): Path to store the model registry
        """
        self.registry_path = Path(registry_path)
        self.metadata_path = self.registry_path / "metadata"
        self.models_path = self.registry_path / "models"
        
        # Create necessary directories
        self.registry_path.mkdir(exist_ok=True)
        self.metadata_path.mkdir(exist_ok=True)
        self.models_path.mkdir(exist_ok=True)
        
        # Initialize metadata index
        self.metadata_index_path = self.metadata_path / "index.json"
        if not self.metadata_index_path.exists():
            self._save_metadata_index({})
    
    def _save_metadata_index(self, index: Dict[str, Any]) -> None:
        """Save the metadata index to disk."""
        with open(self.metadata_index_path, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _load_metadata_index(self) -> Dict[str, Any]:
        """Load the metadata index from disk."""
        if not self.metadata_index_path.exists():
            return {}
        with open(self.metadata_index_path, 'r') as f:
            return json.load(f)
    
    def register_model(self, 
                      model_path: str,
                      model_name: str,
                      version: str,
                      metadata: Dict[str, Any]) -> str:
        """
        Register a new model version.
        
        Args:
            model_path (str): Path to the model file
            model_name (str): Name of the model
            version (str): Version identifier
            metadata (Dict[str, Any]): Additional metadata about the model
            
        Returns:
            str: Model ID
        """
        # Generate unique model ID
        timestamp = datetime.datetime.now().isoformat()
        model_id = f"{model_name}_{version}_{hashlib.md5(timestamp.encode()).hexdigest()[:8]}"
        
        # Create model directory
        model_dir = self.models_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Copy model file
        shutil.copy2(model_path, model_dir / "model.pkl")
        
        # Prepare metadata
        model_metadata = {
            "model_id": model_id,
            "model_name": model_name,
            "version": version,
            "timestamp": timestamp,
            "path": str(model_dir),
            **metadata
        }
        
        # Save metadata
        metadata_file = self.metadata_path / f"{model_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Update index
        index = self._load_metadata_index()
        if model_name not in index:
            index[model_name] = []
        index[model_name].append(model_id)
        self._save_metadata_index(index)
        
        return model_id
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve model information by ID.
        
        Args:
            model_id (str): Model ID to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Model metadata if found, None otherwise
        """
        metadata_file = self.metadata_path / f"{model_id}.json"
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def list_models(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered models or models with a specific name.
        
        Args:
            model_name (Optional[str]): Filter by model name
            
        Returns:
            List[Dict[str, Any]]: List of model metadata
        """
        index = self._load_metadata_index()
        models = []
        
        if model_name:
            if model_name not in index:
                return []
            model_ids = index[model_name]
        else:
            model_ids = [mid for ids in index.values() for mid in ids]
        
        for model_id in model_ids:
            model_info = self.get_model(model_id)
            if model_info:
                models.append(model_info)
        
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model and its metadata.
        
        Args:
            model_id (str): Model ID to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        model_info = self.get_model(model_id)
        if not model_info:
            return False
        
        # Remove model files
        model_dir = Path(model_info["path"])
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove metadata
        metadata_file = self.metadata_path / f"{model_id}.json"
        if metadata_file.exists():
            metadata_file.unlink()
        
        # Update index
        index = self._load_metadata_index()
        for model_name, model_ids in index.items():
            if model_id in model_ids:
                index[model_name].remove(model_id)
                if not index[model_name]:
                    del index[model_name]
                break
        self._save_metadata_index(index)
        
        return True
    
    def update_metadata(self, model_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a registered model.
        
        Args:
            model_id (str): Model ID to update
            metadata (Dict[str, Any]): New metadata to add/update
            
        Returns:
            bool: True if successful, False otherwise
        """
        model_info = self.get_model(model_id)
        if not model_info:
            return False
        
        # Update metadata
        model_info.update(metadata)
        
        # Save updated metadata
        metadata_file = self.metadata_path / f"{model_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return True 