"""
Enhanced Model Versioning System for Igel.

This module provides advanced model versioning capabilities including:
- Model lineage tracking
- Metadata management
- Experiment integration
- Model comparison and visualization
- Deployment tracking
"""

import os
import json
import datetime
import uuid
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import joblib
import logging
from dataclasses import dataclass, asdict
import sqlite3
import shutil

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Model version information."""
    version_id: str
    model_name: str
    version: str
    created_at: str
    created_by: str
    description: str
    tags: Dict[str, str]
    model_path: str
    model_type: str
    model_algorithm: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    dataset_info: Dict[str, Any]
    dependencies: Dict[str, str]
    lineage: Dict[str, Any]
    status: str  # DRAFT, STAGING, PRODUCTION, ARCHIVED


@dataclass
class ModelDeployment:
    """Model deployment information."""
    deployment_id: str
    model_version_id: str
    environment: str
    deployed_at: str
    deployed_by: str
    status: str  # ACTIVE, INACTIVE, FAILED
    endpoint: Optional[str]
    performance_metrics: Dict[str, float]


class ModelVersioning:
    """
    Enhanced model versioning system with lineage tracking.
    
    Provides functionality for:
    - Model versioning with semantic versioning
    - Lineage tracking from experiments
    - Metadata management
    - Deployment tracking
    - Model comparison and visualization
    """
    
    def __init__(self, versioning_path: str = "model_versions"):
        """
        Initialize the model versioning system.
        
        Args:
            versioning_path (str): Path to store versioned models
        """
        self.versioning_path = Path(versioning_path)
        self.versioning_path.mkdir(exist_ok=True)
        
        # Database for storing version metadata
        self.db_path = self.versioning_path / "versions.db"
        self._init_database()
        
        # Model storage
        self.models_path = self.versioning_path / "models"
        self.models_path.mkdir(exist_ok=True)
        
        # Deployments
        self.deployments_path = self.versioning_path / "deployments"
        self.deployments_path.mkdir(exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database for model versioning."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create model versions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                version_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                created_by TEXT NOT NULL,
                description TEXT,
                tags TEXT,
                model_path TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_algorithm TEXT NOT NULL,
                parameters TEXT,
                metrics TEXT,
                dataset_info TEXT,
                dependencies TEXT,
                lineage TEXT,
                status TEXT NOT NULL,
                UNIQUE(model_name, version)
            )
        ''')
        
        # Create deployments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS deployments (
                deployment_id TEXT PRIMARY KEY,
                model_version_id TEXT NOT NULL,
                environment TEXT NOT NULL,
                deployed_at TEXT NOT NULL,
                deployed_by TEXT NOT NULL,
                status TEXT NOT NULL,
                endpoint TEXT,
                performance_metrics TEXT,
                FOREIGN KEY (model_version_id) REFERENCES model_versions (version_id)
            )
        ''')
        
        # Create model lineage table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_lineage (
                version_id TEXT NOT NULL,
                parent_version_id TEXT,
                experiment_id TEXT,
                run_id TEXT,
                lineage_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (version_id) REFERENCES model_versions (version_id),
                FOREIGN KEY (parent_version_id) REFERENCES model_versions (version_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_version(self,
                      model_name: str,
                      model,
                      model_type: str,
                      model_algorithm: str,
                      parameters: Dict[str, Any],
                      metrics: Dict[str, float],
                      dataset_info: Dict[str, Any],
                      description: str = "",
                      tags: Optional[Dict[str, str]] = None,
                      dependencies: Optional[Dict[str, str]] = None,
                      parent_version_id: Optional[str] = None,
                      experiment_id: Optional[str] = None,
                      run_id: Optional[str] = None,
                      created_by: str = "user") -> str:
        """
        Create a new model version.
        
        Args:
            model_name (str): Name of the model
            model: The model object to version
            model_type (str): Type of model (regression, classification, etc.)
            model_algorithm (str): Algorithm used
            parameters (Dict[str, Any]): Model parameters
            metrics (Dict[str, float]): Model performance metrics
            dataset_info (Dict[str, Any]): Information about the training dataset
            description (str): Version description
            tags (Optional[Dict[str, str]]): Version tags
            dependencies (Optional[Dict[str, str]]): Model dependencies
            parent_version_id (Optional[str]): Parent version ID for lineage
            experiment_id (Optional[str]): Associated experiment ID
            run_id (Optional[str]): Associated run ID
            created_by (str): User who created the version
            
        Returns:
            str: Version ID
        """
        # Generate version ID and determine version number
        version_id = str(uuid.uuid4())
        version = self._get_next_version(model_name)
        created_at = datetime.datetime.now().isoformat()
        
        # Create model directory
        model_dir = self.models_path / version_id
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        # Prepare lineage information
        lineage = {
            "parent_version_id": parent_version_id,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "lineage_type": "experiment" if experiment_id else "manual"
        }
        
        # Create model version record
        model_version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            version=version,
            created_at=created_at,
            created_by=created_by,
            description=description,
            tags=tags or {},
            model_path=str(model_path),
            model_type=model_type,
            model_algorithm=model_algorithm,
            parameters=parameters,
            metrics=metrics,
            dataset_info=dataset_info,
            dependencies=dependencies or {},
            lineage=lineage,
            status="DRAFT"
        )
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_versions 
            (version_id, model_name, version, created_at, created_by, description,
             tags, model_path, model_type, model_algorithm, parameters, metrics,
             dataset_info, dependencies, lineage, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_version.version_id,
            model_version.model_name,
            model_version.version,
            model_version.created_at,
            model_version.created_by,
            model_version.description,
            json.dumps(model_version.tags),
            model_version.model_path,
            model_version.model_type,
            model_version.model_algorithm,
            json.dumps(model_version.parameters),
            json.dumps(model_version.metrics),
            json.dumps(model_version.dataset_info),
            json.dumps(model_version.dependencies),
            json.dumps(model_version.lineage),
            model_version.status
        ))
        
        # Save lineage information
        if parent_version_id or experiment_id or run_id:
            cursor.execute('''
                INSERT INTO model_lineage 
                (version_id, parent_version_id, experiment_id, run_id, lineage_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                version_id,
                parent_version_id,
                experiment_id,
                run_id,
                lineage["lineage_type"],
                created_at
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created model version {version} for {model_name} with ID: {version_id}")
        return version_id
    
    def _get_next_version(self, model_name: str) -> str:
        """
        Get the next version number for a model.
        
        Args:
            model_name (str): Model name
            
        Returns:
            str: Next version number (e.g., "1.0.0", "1.1.0")
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT version FROM model_versions 
            WHERE model_name = ? ORDER BY version DESC LIMIT 1
        ''', (model_name,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            current_version = row[0]
            # Simple version increment - in production, use semantic versioning
            try:
                major, minor, patch = map(int, current_version.split('.'))
                return f"{major}.{minor}.{patch + 1}"
            except:
                return f"{current_version}.1"
        else:
            return "1.0.0"
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """
        Get model version by ID.
        
        Args:
            version_id (str): Version ID
            
        Returns:
            Optional[ModelVersion]: Model version if found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT version_id, model_name, version, created_at, created_by, description,
                   tags, model_path, model_type, model_algorithm, parameters, metrics,
                   dataset_info, dependencies, lineage, status
            FROM model_versions WHERE version_id = ?
        ''', (version_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return ModelVersion(
                version_id=row[0],
                model_name=row[1],
                version=row[2],
                created_at=row[3],
                created_by=row[4],
                description=row[5],
                tags=json.loads(row[6]) if row[6] else {},
                model_path=row[7],
                model_type=row[8],
                model_algorithm=row[9],
                parameters=json.loads(row[10]) if row[10] else {},
                metrics=json.loads(row[11]) if row[11] else {},
                dataset_info=json.loads(row[12]) if row[12] else {},
                dependencies=json.loads(row[13]) if row[13] else {},
                lineage=json.loads(row[14]) if row[14] else {},
                status=row[15]
            )
        return None
    
    def list_versions(self, model_name: Optional[str] = None) -> List[ModelVersion]:
        """
        List model versions, optionally filtered by model name.
        
        Args:
            model_name (Optional[str]): Filter by model name
            
        Returns:
            List[ModelVersion]: List of model versions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if model_name:
            cursor.execute('''
                SELECT version_id, model_name, version, created_at, created_by, description,
                       tags, model_path, model_type, model_algorithm, parameters, metrics,
                       dataset_info, dependencies, lineage, status
                FROM model_versions WHERE model_name = ? ORDER BY created_at DESC
            ''', (model_name,))
        else:
            cursor.execute('''
                SELECT version_id, model_name, version, created_at, created_by, description,
                       tags, model_path, model_type, model_algorithm, parameters, metrics,
                       dataset_info, dependencies, lineage, status
                FROM model_versions ORDER BY created_at DESC
            ''')
        
        versions = []
        for row in cursor.fetchall():
            versions.append(ModelVersion(
                version_id=row[0],
                model_name=row[1],
                version=row[2],
                created_at=row[3],
                created_by=row[4],
                description=row[5],
                tags=json.loads(row[6]) if row[6] else {},
                model_path=row[7],
                model_type=row[8],
                model_algorithm=row[9],
                parameters=json.loads(row[10]) if row[10] else {},
                metrics=json.loads(row[11]) if row[11] else {},
                dataset_info=json.loads(row[12]) if row[12] else {},
                dependencies=json.loads(row[13]) if row[13] else {},
                lineage=json.loads(row[14]) if row[14] else {},
                status=row[15]
            ))
        
        conn.close()
        return versions
    
    def load_model(self, version_id: str):
        """
        Load a model from version ID.
        
        Args:
            version_id (str): Version ID
            
        Returns:
            The loaded model
        """
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"Model version {version_id} not found")
        
        if not Path(version.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {version.model_path}")
        
        return joblib.load(version.model_path)
    
    def update_status(self, version_id: str, status: str):
        """
        Update the status of a model version.
        
        Args:
            version_id (str): Version ID
            status (str): New status (DRAFT, STAGING, PRODUCTION, ARCHIVED)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE model_versions SET status = ? WHERE version_id = ?
        ''', (status, version_id))
        conn.commit()
        conn.close()
        
        logger.info(f"Updated model version {version_id} status to {status}")
    
    def deploy_model(self,
                    version_id: str,
                    environment: str,
                    deployed_by: str = "user",
                    endpoint: Optional[str] = None) -> str:
        """
        Deploy a model version to an environment.
        
        Args:
            version_id (str): Version ID to deploy
            environment (str): Target environment
            deployed_by (str): User who deployed the model
            endpoint (Optional[str]): Model endpoint URL
            
        Returns:
            str: Deployment ID
        """
        deployment_id = str(uuid.uuid4())
        deployed_at = datetime.datetime.now().isoformat()
        
        deployment = ModelDeployment(
            deployment_id=deployment_id,
            model_version_id=version_id,
            environment=environment,
            deployed_at=deployed_at,
            deployed_by=deployed_by,
            status="ACTIVE",
            endpoint=endpoint,
            performance_metrics={}
        )
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO deployments 
            (deployment_id, model_version_id, environment, deployed_at, deployed_by, status, endpoint, performance_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            deployment.deployment_id,
            deployment.model_version_id,
            deployment.environment,
            deployment.deployed_at,
            deployment.deployed_by,
            deployment.status,
            deployment.endpoint,
            json.dumps(deployment.performance_metrics)
        ))
        conn.commit()
        conn.close()
        
        # Update model status to PRODUCTION if deploying to production
        if environment.lower() == "production":
            self.update_status(version_id, "PRODUCTION")
        
        logger.info(f"Deployed model version {version_id} to {environment}")
        return deployment_id
    
    def get_deployments(self, version_id: Optional[str] = None) -> List[ModelDeployment]:
        """
        Get model deployments.
        
        Args:
            version_id (Optional[str]): Filter by version ID
            
        Returns:
            List[ModelDeployment]: List of deployments
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if version_id:
            cursor.execute('''
                SELECT deployment_id, model_version_id, environment, deployed_at, deployed_by, status, endpoint, performance_metrics
                FROM deployments WHERE model_version_id = ? ORDER BY deployed_at DESC
            ''', (version_id,))
        else:
            cursor.execute('''
                SELECT deployment_id, model_version_id, environment, deployed_at, deployed_by, status, endpoint, performance_metrics
                FROM deployments ORDER BY deployed_at DESC
            ''')
        
        deployments = []
        for row in cursor.fetchall():
            deployments.append(ModelDeployment(
                deployment_id=row[0],
                model_version_id=row[1],
                environment=row[2],
                deployed_at=row[3],
                deployed_by=row[4],
                status=row[5],
                endpoint=row[6],
                performance_metrics=json.loads(row[7]) if row[7] else {}
            ))
        
        conn.close()
        return deployments
    
    def compare_versions(self, version_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple model versions.
        
        Args:
            version_ids (List[str]): List of version IDs to compare
            
        Returns:
            pd.DataFrame: Comparison table
        """
        versions = []
        for version_id in version_ids:
            version = self.get_version(version_id)
            if version:
                versions.append(version)
        
        if not versions:
            return pd.DataFrame()
        
        # Prepare comparison data
        comparison_data = []
        for version in versions:
            data = {
                "version_id": version.version_id,
                "model_name": version.model_name,
                "version": version.version,
                "created_at": version.created_at,
                "created_by": version.created_by,
                "status": version.status,
                "model_type": version.model_type,
                "model_algorithm": version.model_algorithm,
                **version.metrics,
                **{f"param_{k}": v for k, v in version.parameters.items()}
            }
            comparison_data.append(data)
        
        return pd.DataFrame(comparison_data)
    
    def get_lineage(self, version_id: str) -> Dict[str, Any]:
        """
        Get the lineage of a model version.
        
        Args:
            version_id (str): Version ID
            
        Returns:
            Dict[str, Any]: Lineage information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT parent_version_id, experiment_id, run_id, lineage_type, created_at
            FROM model_lineage WHERE version_id = ?
        ''', (version_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "parent_version_id": row[0],
                "experiment_id": row[1],
                "run_id": row[2],
                "lineage_type": row[3],
                "created_at": row[4]
            }
        return {}
    
    def export_version(self, version_id: str, output_path: str):
        """
        Export a model version to a file.
        
        Args:
            version_id (str): Version ID to export
            output_path (str): Output file path
        """
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"Model version {version_id} not found")
        
        # Copy model file
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True)
        
        model_file = output_dir / f"{version.model_name}_v{version.version}.joblib"
        shutil.copy2(version.model_path, model_file)
        
        # Export metadata
        metadata_file = output_dir / f"{version.model_name}_v{version.version}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(version), f, indent=2)
        
        logger.info(f"Exported model version {version_id} to {output_path}")
    
    def delete_version(self, version_id: str) -> bool:
        """
        Delete a model version.
        
        Args:
            version_id (str): Version ID to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        version = self.get_version(version_id)
        if not version:
            return False
        
        # Check if version is deployed
        deployments = self.get_deployments(version_id)
        if deployments:
            logger.warning(f"Cannot delete version {version_id} - it has active deployments")
            return False
        
        # Remove model files
        model_dir = Path(version.model_path).parent
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM model_versions WHERE version_id = ?', (version_id,))
        cursor.execute('DELETE FROM model_lineage WHERE version_id = ?', (version_id,))
        conn.commit()
        conn.close()
        
        logger.info(f"Deleted model version {version_id}")
        return True 