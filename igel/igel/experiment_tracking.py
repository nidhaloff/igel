"""
Experiment Tracking System for Igel.

This module provides MLflow-like experiment tracking functionality including:
- Experiment management
- Run tracking with metrics and parameters
- Model versioning with lineage
- Experiment comparison and visualization
"""

import os
import json
import datetime
import uuid
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import logging
import joblib
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """Experiment configuration and metadata."""
    experiment_id: str
    name: str
    description: str
    created_at: str
    tags: Dict[str, str]
    artifact_location: str


@dataclass
class Run:
    """Run configuration and metadata."""
    run_id: str
    experiment_id: str
    name: str
    status: str  # RUNNING, FINISHED, FAILED
    start_time: str
    end_time: Optional[str]
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    tags: Dict[str, str]
    model_path: Optional[str]
    model_metadata: Optional[Dict[str, Any]]


class ExperimentTracker:
    """
    MLflow-like experiment tracking system for igel.
    
    Provides functionality for:
    - Creating and managing experiments
    - Tracking runs with metrics and parameters
    - Model versioning with lineage
    - Experiment comparison and visualization
    """
    
    def __init__(self, tracking_uri: str = "experiments"):
        """
        Initialize the experiment tracker.
        
        Args:
            tracking_uri (str): Path to store experiment data
        """
        self.tracking_uri = Path(tracking_uri)
        self.tracking_uri.mkdir(exist_ok=True)
        
        # Database for storing experiment metadata
        self.db_path = self.tracking_uri / "experiments.db"
        self._init_database()
        
        # Artifacts directory
        self.artifacts_path = self.tracking_uri / "artifacts"
        self.artifacts_path.mkdir(exist_ok=True)
        
        # Models directory
        self.models_path = self.tracking_uri / "models"
        self.models_path.mkdir(exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database for experiment tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                tags TEXT,
                artifact_location TEXT NOT NULL
            )
        ''')
        
        # Create runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                name TEXT,
                status TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                metrics TEXT,
                parameters TEXT,
                tags TEXT,
                model_path TEXT,
                model_metadata TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            )
        ''')
        
        # Create metrics table for detailed metric tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value REAL NOT NULL,
                step INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                PRIMARY KEY (run_id, key, step),
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_experiment(self, 
                         name: str, 
                         description: str = "",
                         tags: Optional[Dict[str, str]] = None) -> str:
        """
        Create a new experiment.
        
        Args:
            name (str): Experiment name
            description (str): Experiment description
            tags (Optional[Dict[str, str]]): Experiment tags
            
        Returns:
            str: Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        created_at = datetime.datetime.now().isoformat()
        artifact_location = str(self.artifacts_path / experiment_id)
        
        # Create artifact directory
        Path(artifact_location).mkdir(exist_ok=True)
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            created_at=created_at,
            tags=tags or {},
            artifact_location=artifact_location
        )
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO experiments 
            (experiment_id, name, description, created_at, tags, artifact_location)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            experiment.experiment_id,
            experiment.name,
            experiment.description,
            experiment.created_at,
            json.dumps(experiment.tags),
            experiment.artifact_location
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"Created experiment '{name}' with ID: {experiment_id}")
        return experiment_id
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get experiment by ID.
        
        Args:
            experiment_id (str): Experiment ID
            
        Returns:
            Optional[Experiment]: Experiment if found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT experiment_id, name, description, created_at, tags, artifact_location
            FROM experiments WHERE experiment_id = ?
        ''', (experiment_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Experiment(
                experiment_id=row[0],
                name=row[1],
                description=row[2],
                created_at=row[3],
                tags=json.loads(row[4]) if row[4] else {},
                artifact_location=row[5]
            )
        return None
    
    def list_experiments(self) -> List[Experiment]:
        """
        List all experiments.
        
        Returns:
            List[Experiment]: List of experiments
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT experiment_id, name, description, created_at, tags, artifact_location
            FROM experiments ORDER BY created_at DESC
        ''')
        
        experiments = []
        for row in cursor.fetchall():
            experiments.append(Experiment(
                experiment_id=row[0],
                name=row[1],
                description=row[2],
                created_at=row[3],
                tags=json.loads(row[4]) if row[4] else {},
                artifact_location=row[5]
            ))
        
        conn.close()
        return experiments
    
    def start_run(self, 
                  experiment_id: str, 
                  name: str = "",
                  tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new run in an experiment.
        
        Args:
            experiment_id (str): Experiment ID
            name (str): Run name
            tags (Optional[Dict[str, str]]): Run tags
            
        Returns:
            str: Run ID
        """
        run_id = str(uuid.uuid4())
        start_time = datetime.datetime.now().isoformat()
        
        run = Run(
            run_id=run_id,
            experiment_id=experiment_id,
            name=name,
            status="RUNNING",
            start_time=start_time,
            end_time=None,
            metrics={},
            parameters={},
            tags=tags or {},
            model_path=None,
            model_metadata=None
        )
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO runs 
            (run_id, experiment_id, name, status, start_time, end_time, 
             metrics, parameters, tags, model_path, model_metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run.run_id,
            run.experiment_id,
            run.name,
            run.status,
            run.start_time,
            run.end_time,
            json.dumps(run.metrics),
            json.dumps(run.parameters),
            json.dumps(run.tags),
            run.model_path,
            json.dumps(run.model_metadata) if run.model_metadata else None
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"Started run '{name}' with ID: {run_id}")
        return run_id
    
    def log_metric(self, run_id: str, key: str, value: float, step: int = 0):
        """
        Log a metric for a run.
        
        Args:
            run_id (str): Run ID
            key (str): Metric name
            value (float): Metric value
            step (int): Step number (for iterative metrics)
        """
        timestamp = datetime.datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert or update metric
        cursor.execute('''
            INSERT OR REPLACE INTO metrics (run_id, key, value, step, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (run_id, key, value, step, timestamp))
        
        # Update run metrics summary
        cursor.execute('''
            SELECT metrics FROM runs WHERE run_id = ?
        ''', (run_id,))
        
        row = cursor.fetchone()
        if row:
            metrics = json.loads(row[0]) if row[0] else {}
            metrics[key] = value
            cursor.execute('''
                UPDATE runs SET metrics = ? WHERE run_id = ?
            ''', (json.dumps(metrics), run_id))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Logged metric {key}={value} for run {run_id}")
    
    def log_parameter(self, run_id: str, key: str, value: Any):
        """
        Log a parameter for a run.
        
        Args:
            run_id (str): Run ID
            key (str): Parameter name
            value (Any): Parameter value
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT parameters FROM runs WHERE run_id = ?
        ''', (run_id,))
        
        row = cursor.fetchone()
        if row:
            parameters = json.loads(row[0]) if row[0] else {}
            parameters[key] = value
            cursor.execute('''
                UPDATE runs SET parameters = ? WHERE run_id = ?
            ''', (json.dumps(parameters), run_id))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Logged parameter {key}={value} for run {run_id}")
    
    def log_model(self, run_id: str, model, model_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Log a model for a run.
        
        Args:
            run_id (str): Run ID
            model: The model to save
            model_name (str): Name for the model
            metadata (Optional[Dict[str, Any]]): Additional model metadata
        """
        # Create model directory
        model_dir = self.models_path / run_id
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        
        # Prepare metadata
        model_metadata = {
            "model_name": model_name,
            "model_path": str(model_path),
            "model_type": type(model).__name__,
            "saved_at": datetime.datetime.now().isoformat(),
            **(metadata or {})
        }
        
        # Update run with model info
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE runs SET model_path = ?, model_metadata = ? WHERE run_id = ?
        ''', (str(model_path), json.dumps(model_metadata), run_id))
        conn.commit()
        conn.close()
        
        logger.info(f"Logged model {model_name} for run {run_id}")
    
    def end_run(self, run_id: str, status: str = "FINISHED"):
        """
        End a run.
        
        Args:
            run_id (str): Run ID
            status (str): Final status (FINISHED, FAILED)
        """
        end_time = datetime.datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE runs SET status = ?, end_time = ? WHERE run_id = ?
        ''', (status, end_time, run_id))
        conn.commit()
        conn.close()
        
        logger.info(f"Ended run {run_id} with status: {status}")
    
    def get_run(self, run_id: str) -> Optional[Run]:
        """
        Get run by ID.
        
        Args:
            run_id (str): Run ID
            
        Returns:
            Optional[Run]: Run if found, None otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT run_id, experiment_id, name, status, start_time, end_time,
                   metrics, parameters, tags, model_path, model_metadata
            FROM runs WHERE run_id = ?
        ''', (run_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Run(
                run_id=row[0],
                experiment_id=row[1],
                name=row[2],
                status=row[3],
                start_time=row[4],
                end_time=row[5],
                metrics=json.loads(row[6]) if row[6] else {},
                parameters=json.loads(row[7]) if row[7] else {},
                tags=json.loads(row[8]) if row[8] else {},
                model_path=row[9],
                model_metadata=json.loads(row[10]) if row[10] else None
            )
        return None
    
    def list_runs(self, experiment_id: Optional[str] = None) -> List[Run]:
        """
        List runs, optionally filtered by experiment.
        
        Args:
            experiment_id (Optional[str]): Filter by experiment ID
            
        Returns:
            List[Run]: List of runs
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if experiment_id:
            cursor.execute('''
                SELECT run_id, experiment_id, name, status, start_time, end_time,
                       metrics, parameters, tags, model_path, model_metadata
                FROM runs WHERE experiment_id = ? ORDER BY start_time DESC
            ''', (experiment_id,))
        else:
            cursor.execute('''
                SELECT run_id, experiment_id, name, status, start_time, end_time,
                       metrics, parameters, tags, model_path, model_metadata
                FROM runs ORDER BY start_time DESC
            ''')
        
        runs = []
        for row in cursor.fetchall():
            runs.append(Run(
                run_id=row[0],
                experiment_id=row[1],
                name=row[2],
                status=row[3],
                start_time=row[4],
                end_time=row[5],
                metrics=json.loads(row[6]) if row[6] else {},
                parameters=json.loads(row[7]) if row[7] else {},
                tags=json.loads(row[8]) if row[8] else {},
                model_path=row[9],
                model_metadata=json.loads(row[10]) if row[10] else None
            ))
        
        conn.close()
        return runs
    
    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple runs.
        
        Args:
            run_ids (List[str]): List of run IDs to compare
            
        Returns:
            pd.DataFrame: Comparison table
        """
        runs = []
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run:
                runs.append(run)
        
        if not runs:
            return pd.DataFrame()
        
        # Prepare comparison data
        comparison_data = []
        for run in runs:
            data = {
                "run_id": run.run_id,
                "name": run.name,
                "status": run.status,
                "start_time": run.start_time,
                "end_time": run.end_time,
                **run.metrics,
                **{f"param_{k}": v for k, v in run.parameters.items()}
            }
            comparison_data.append(data)
        
        return pd.DataFrame(comparison_data)
    
    def plot_metrics(self, run_ids: List[str], metrics: List[str], figsize: tuple = (12, 8)):
        """
        Plot metrics for multiple runs.
        
        Args:
            run_ids (List[str]): List of run IDs to plot
            metrics (List[str]): List of metrics to plot
            figsize (tuple): Figure size
        """
        fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            for run_id in run_ids:
                # Get metric history
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT step, value FROM metrics 
                    WHERE run_id = ? AND key = ? 
                    ORDER BY step
                ''', (run_id, metric))
                
                steps, values = [], []
                for row in cursor.fetchall():
                    steps.append(row[0])
                    values.append(row[1])
                
                conn.close()
                
                if steps:
                    run = self.get_run(run_id)
                    label = run.name if run and run.name else run_id
                    ax.plot(steps, values, label=label, marker='o')
            
            ax.set_title(f'{metric} over time')
            ax.set_xlabel('Step')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def export_experiment(self, experiment_id: str, output_path: str):
        """
        Export experiment data to a file.
        
        Args:
            experiment_id (str): Experiment ID to export
            output_path (str): Output file path
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        runs = self.list_runs(experiment_id)
        
        export_data = {
            "experiment": asdict(experiment),
            "runs": [asdict(run) for run in runs]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported experiment {experiment_id} to {output_path}")
    
    def get_model_lineage(self, model_path: str) -> Dict[str, Any]:
        """
        Get model lineage information.
        
        Args:
            model_path (str): Path to the model
            
        Returns:
            Dict[str, Any]: Lineage information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT run_id, experiment_id, name, start_time, end_time,
                   metrics, parameters, tags, model_metadata
            FROM runs WHERE model_path = ?
        ''', (model_path,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "run_id": row[0],
                "experiment_id": row[1],
                "run_name": row[2],
                "start_time": row[3],
                "end_time": row[4],
                "metrics": json.loads(row[5]) if row[5] else {},
                "parameters": json.loads(row[6]) if row[6] else {},
                "tags": json.loads(row[7]) if row[7] else {},
                "model_metadata": json.loads(row[8]) if row[8] else {}
            }
        return {}


class ExperimentContext:
    """
    Context manager for experiment runs.
    
    Usage:
        with ExperimentContext(tracker, experiment_id, run_name) as run_id:
            tracker.log_parameter(run_id, "learning_rate", 0.01)
            # ... training code ...
            tracker.log_metric(run_id, "accuracy", 0.95)
    """
    
    def __init__(self, tracker: ExperimentTracker, experiment_id: str, run_name: str = ""):
        self.tracker = tracker
        self.experiment_id = experiment_id
        self.run_name = run_name
        self.run_id = None
    
    def __enter__(self):
        self.run_id = self.tracker.start_run(self.experiment_id, self.run_name)
        return self.run_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.tracker.end_run(self.run_id, "FAILED")
        else:
            self.tracker.end_run(self.run_id, "FINISHED") 