"""
Federated Learning Framework for igel.
Provides distributed machine learning capabilities for privacy-preserving model training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib
import logging
from datetime import datetime
import json
import threading
import time
import socket
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class FederatedClient:
    """
    Client for federated learning that handles local training and model updates.
    """
    
    def __init__(self, client_id: str, model: BaseEstimator, data: Tuple[np.ndarray, np.ndarray]):
        """
        Initialize a federated learning client.
        
        Args:
            client_id: Unique identifier for the client
            model: Local model instance
            data: Local training data (X, y)
        """
        self.client_id = client_id
        self.model = model
        self.X, self.y = data
        self.model_weights = None
        self.training_history = []
        self.is_training = False
        
    def train_local_model(self, epochs: int = 1, learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Train the local model on local data.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            
        Returns:
            Training results and model weights
        """
        self.is_training = True
        start_time = time.time()
        
        try:
            # Train the model
            if hasattr(self.model, 'fit'):
                self.model.fit(self.X, self.y)
            
            # Extract model weights/parameters
            self.model_weights = self._extract_model_weights()
            
            # Calculate local metrics
            predictions = self.model.predict(self.X)
            if isinstance(self.model, ClassifierMixin):
                accuracy = accuracy_score(self.y, predictions)
                metrics = {'accuracy': accuracy}
            else:
                mse = mean_squared_error(self.y, predictions)
                r2 = r2_score(self.y, predictions)
                metrics = {'mse': mse, 'r2': r2}
            
            training_time = time.time() - start_time
            
            # Store training history
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'epochs': epochs,
                'learning_rate': learning_rate,
                'metrics': metrics,
                'training_time': training_time,
                'data_size': len(self.X)
            }
            self.training_history.append(training_record)
            
            logger.info(f"Client {self.client_id} trained successfully. Metrics: {metrics}")
            
            return {
                'client_id': self.client_id,
                'model_weights': self.model_weights,
                'metrics': metrics,
                'training_time': training_time,
                'data_size': len(self.X)
            }
            
        except Exception as e:
            logger.error(f"Training failed for client {self.client_id}: {e}")
            return {'client_id': self.client_id, 'error': str(e)}
        finally:
            self.is_training = False
    
    def _extract_model_weights(self) -> Dict[str, Any]:
        """Extract weights/parameters from the model."""
        weights = {}
        
        if hasattr(self.model, 'coef_'):
            weights['coefficients'] = self.model.coef_.tolist() if hasattr(self.model.coef_, 'tolist') else self.model.coef_
        
        if hasattr(self.model, 'intercept_'):
            weights['intercept'] = self.model.intercept_.tolist() if hasattr(self.model.intercept_, 'tolist') else self.model.intercept_
        
        if hasattr(self.model, 'feature_importances_'):
            weights['feature_importances'] = self.model.feature_importances_.tolist()
        
        if hasattr(self.model, 'estimators_'):
            weights['n_estimators'] = len(self.model.estimators_)
        
        return weights
    
    def update_model_weights(self, global_weights: Dict[str, Any], alpha: float = 0.1):
        """
        Update local model with global weights using weighted averaging.
        
        Args:
            global_weights: Global model weights from server
            alpha: Weight for local vs global model (0 = fully global, 1 = fully local)
        """
        if self.model_weights is None:
            self.model_weights = self._extract_model_weights()
        
        # Weighted average of local and global weights
        for key in global_weights:
            if key in self.model_weights:
                if isinstance(self.model_weights[key], list):
                    self.model_weights[key] = [
                        alpha * local + (1 - alpha) * global_val
                        for local, global_val in zip(self.model_weights[key], global_weights[key])
                    ]
                else:
                    self.model_weights[key] = alpha * self.model_weights[key] + (1 - alpha) * global_weights[key]
        
        # Update model with new weights
        self._update_model_with_weights()
    
    def _update_model_with_weights(self):
        """Update the model with current weights."""
        # This is a simplified implementation
        # In practice, you'd need to implement model-specific weight updates
        if hasattr(self.model, 'coef_') and 'coefficients' in self.model_weights:
            self.model.coef_ = np.array(self.model_weights['coefficients'])
        
        if hasattr(self.model, 'intercept_') and 'intercept' in self.model_weights:
            self.model.intercept_ = np.array(self.model_weights['intercept'])


class FederatedServer:
    """
    Server for federated learning that coordinates model aggregation and distribution.
    """
    
    def __init__(self, global_model: BaseEstimator, aggregation_method: str = "fedavg"):
        """
        Initialize the federated learning server.
        
        Args:
            global_model: Global model instance
            aggregation_method: Method for aggregating client updates ('fedavg', 'fedprox', 'scaffold')
        """
        self.global_model = global_model
        self.aggregation_method = aggregation_method
        self.clients = {}
        self.global_weights = None
        self.training_rounds = []
        self.server_history = []
        
    def add_client(self, client: FederatedClient):
        """Add a client to the federation."""
        self.clients[client.client_id] = client
        logger.info(f"Added client {client.client_id} to federation")
    
    def remove_client(self, client_id: str):
        """Remove a client from the federation."""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Removed client {client_id} from federation")
    
    def federated_training_round(self, epochs: int = 1, learning_rate: float = 0.01, 
                                client_fraction: float = 1.0) -> Dict[str, Any]:
        """
        Perform one round of federated training.
        
        Args:
            epochs: Number of local training epochs
            learning_rate: Learning rate for local training
            client_fraction: Fraction of clients to participate in this round
            
        Returns:
            Training round results
        """
        start_time = time.time()
        
        # Select participating clients
        participating_clients = self._select_clients(client_fraction)
        
        if not participating_clients:
            return {'error': 'No clients available for training'}
        
        logger.info(f"Starting federated training round with {len(participating_clients)} clients")
        
        # Train clients in parallel
        client_results = self._train_clients_parallel(participating_clients, epochs, learning_rate)
        
        # Aggregate model updates
        aggregated_weights = self._aggregate_client_updates(client_results)
        
        # Update global model
        self._update_global_model(aggregated_weights)
        
        # Distribute global model to clients
        self._distribute_global_model(aggregated_weights)
        
        # Calculate round metrics
        round_time = time.time() - start_time
        round_metrics = self._calculate_round_metrics(client_results)
        
        # Store round history
        round_record = {
            'timestamp': datetime.now().isoformat(),
            'participating_clients': len(participating_clients),
            'client_results': client_results,
            'round_metrics': round_metrics,
            'round_time': round_time
        }
        self.training_rounds.append(round_record)
        
        logger.info(f"Federated training round completed in {round_time:.2f}s")
        
        return round_record
    
    def _select_clients(self, client_fraction: float) -> List[FederatedClient]:
        """Select clients to participate in the training round."""
        available_clients = list(self.clients.values())
        n_selected = max(1, int(len(available_clients) * client_fraction))
        return np.random.choice(available_clients, size=n_selected, replace=False).tolist()
    
    def _train_clients_parallel(self, clients: List[FederatedClient], epochs: int, 
                               learning_rate: float) -> List[Dict[str, Any]]:
        """Train multiple clients in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=min(len(clients), 4)) as executor:
            # Submit training tasks
            future_to_client = {
                executor.submit(client.train_local_model, epochs, learning_rate): client
                for client in clients
            }
            
            # Collect results
            for future in as_completed(future_to_client):
                client = future_to_client[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Client {client.client_id} training failed: {e}")
                    results.append({'client_id': client.client_id, 'error': str(e)})
        
        return results
    
    def _aggregate_client_updates(self, client_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate client model updates using the specified method."""
        if self.aggregation_method == "fedavg":
            return self._fedavg_aggregation(client_results)
        elif self.aggregation_method == "fedprox":
            return self._fedprox_aggregation(client_results)
        else:
            return self._fedavg_aggregation(client_results)
    
    def _fedavg_aggregation(self, client_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Federated Averaging aggregation."""
        valid_results = [r for r in client_results if 'error' not in r and 'model_weights' in r]
        
        if not valid_results:
            return self.global_weights or {}
        
        # Calculate weighted average based on data size
        total_data_size = sum(r['data_size'] for r in valid_results)
        
        aggregated_weights = {}
        for key in valid_results[0]['model_weights'].keys():
            weighted_sum = np.zeros_like(valid_results[0]['model_weights'][key])
            
            for result in valid_results:
                weight = result['data_size'] / total_data_size
                if isinstance(result['model_weights'][key], list):
                    weighted_sum += weight * np.array(result['model_weights'][key])
                else:
                    weighted_sum += weight * result['model_weights'][key]
            
            aggregated_weights[key] = weighted_sum.tolist() if hasattr(weighted_sum, 'tolist') else weighted_sum
        
        return aggregated_weights
    
    def _fedprox_aggregation(self, client_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """FedProx aggregation with proximal term."""
        # Simplified FedProx implementation
        # In practice, this would include a proximal term to prevent divergence
        return self._fedavg_aggregation(client_results)
    
    def _update_global_model(self, aggregated_weights: Dict[str, Any]):
        """Update the global model with aggregated weights."""
        self.global_weights = aggregated_weights
        
        # Update global model parameters
        if hasattr(self.global_model, 'coef_') and 'coefficients' in aggregated_weights:
            self.global_model.coef_ = np.array(aggregated_weights['coefficients'])
        
        if hasattr(self.global_model, 'intercept_') and 'intercept' in aggregated_weights:
            self.global_model.intercept_ = np.array(aggregated_weights['intercept'])
    
    def _distribute_global_model(self, global_weights: Dict[str, Any]):
        """Distribute global model to all clients."""
        for client in self.clients.values():
            client.update_model_weights(global_weights)
    
    def _calculate_round_metrics(self, client_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics for the training round."""
        valid_results = [r for r in client_results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid client results'}
        
        # Calculate average metrics
        metrics = {}
        for key in ['accuracy', 'mse', 'r2']:
            values = [r['metrics'].get(key) for r in valid_results if key in r['metrics']]
            if values:
                metrics[f'avg_{key}'] = np.mean(values)
                metrics[f'std_{key}'] = np.std(values)
        
        metrics['total_clients'] = len(client_results)
        metrics['successful_clients'] = len(valid_results)
        metrics['success_rate'] = len(valid_results) / len(client_results)
        
        return metrics
    
    def run_federated_training(self, num_rounds: int = 10, epochs_per_round: int = 1, 
                              learning_rate: float = 0.01, client_fraction: float = 1.0) -> Dict[str, Any]:
        """
        Run complete federated training process.
        
        Args:
            num_rounds: Number of federated training rounds
            epochs_per_round: Number of local epochs per round
            learning_rate: Learning rate for local training
            client_fraction: Fraction of clients participating per round
            
        Returns:
            Complete training results
        """
        logger.info(f"Starting federated training: {num_rounds} rounds, {len(self.clients)} clients")
        
        training_start = time.time()
        
        for round_num in range(num_rounds):
            logger.info(f"Training round {round_num + 1}/{num_rounds}")
            
            round_result = self.federated_training_round(
                epochs=epochs_per_round,
                learning_rate=learning_rate,
                client_fraction=client_fraction
            )
            
            if 'error' in round_result:
                logger.error(f"Round {round_num + 1} failed: {round_result['error']}")
                break
        
        total_time = time.time() - training_start
        
        # Calculate final metrics
        final_metrics = self._evaluate_global_model()
        
        training_summary = {
            'total_rounds': num_rounds,
            'completed_rounds': len(self.training_rounds),
            'total_time': total_time,
            'final_metrics': final_metrics,
            'round_history': self.training_rounds
        }
        
        self.server_history.append(training_summary)
        
        logger.info(f"Federated training completed in {total_time:.2f}s")
        
        return training_summary
    
    def _evaluate_global_model(self) -> Dict[str, Any]:
        """Evaluate the global model on all client data."""
        all_predictions = []
        all_labels = []
        
        for client in self.clients.values():
            predictions = self.global_model.predict(client.X)
            all_predictions.extend(predictions)
            all_labels.extend(client.y)
        
        if isinstance(self.global_model, ClassifierMixin):
            accuracy = accuracy_score(all_labels, all_predictions)
            return {'accuracy': accuracy}
        else:
            mse = mean_squared_error(all_labels, all_predictions)
            r2 = r2_score(all_labels, all_predictions)
            return {'mse': mse, 'r2': r2}
    
    def get_training_report(self) -> str:
        """Generate a comprehensive training report."""
        report = []
        report.append("Federated Learning Training Report")
        report.append("=" * 35)
        report.append(f"Total Clients: {len(self.clients)}")
        report.append(f"Completed Rounds: {len(self.training_rounds)}")
        report.append(f"Aggregation Method: {self.aggregation_method}")
        
        if self.training_rounds:
            report.append("\nRound-by-Round Results:")
            report.append("-" * 25)
            
            for i, round_data in enumerate(self.training_rounds):
                report.append(f"\nRound {i + 1}:")
                report.append(f"  Participating Clients: {round_data['participating_clients']}")
                report.append(f"  Round Time: {round_data['round_time']:.2f}s")
                
                if 'round_metrics' in round_data:
                    metrics = round_data['round_metrics']
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            report.append(f"  {key}: {value:.4f}")
        
        return "\n".join(report)
    
    def save_federated_model(self, filepath: str):
        """Save the federated model and training history."""
        # Save global model
        joblib.dump(self.global_model, f"{filepath}_global_model.joblib")
        
        # Save training history
        history_data = {
            'aggregation_method': self.aggregation_method,
            'training_rounds': self.training_rounds,
            'server_history': self.server_history,
            'global_weights': self.global_weights,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{filepath}_federated_history.json", 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        logger.info(f"Federated model saved to {filepath}")


class FederatedLearningManager:
    """
    High-level manager for federated learning operations.
    """
    
    def __init__(self, problem_type: str = "classification"):
        """
        Initialize the federated learning manager.
        
        Args:
            problem_type: Type of problem ('classification' or 'regression')
        """
        self.problem_type = problem_type
        self.server = None
        self.clients = {}
        
    def create_federation(self, global_model: BaseEstimator, aggregation_method: str = "fedavg"):
        """Create a new federation with a global model."""
        self.server = FederatedServer(global_model, aggregation_method)
        logger.info(f"Created federation with {aggregation_method} aggregation")
    
    def add_client_data(self, client_id: str, X: np.ndarray, y: np.ndarray, 
                        model_type: str = None) -> FederatedClient:
        """Add client data to the federation."""
        if model_type is None:
            if self.problem_type == "classification":
                model = LogisticRegression()
            else:
                model = LinearRegression()
        else:
            # Create model based on type
            if model_type == "random_forest":
                model = RandomForestClassifier() if self.problem_type == "classification" else RandomForestRegressor()
            else:
                model = LogisticRegression() if self.problem_type == "classification" else LinearRegression()
        
        client = FederatedClient(client_id, model, (X, y))
        self.clients[client_id] = client
        
        if self.server:
            self.server.add_client(client)
        
        return client
    
    def run_federated_training(self, num_rounds: int = 10, **kwargs) -> Dict[str, Any]:
        """Run federated training with all clients."""
        if not self.server:
            raise ValueError("No federation created. Call create_federation() first.")
        
        return self.server.run_federated_training(num_rounds, **kwargs)
    
    def get_federation_status(self) -> Dict[str, Any]:
        """Get current status of the federation."""
        return {
            'total_clients': len(self.clients),
            'server_created': self.server is not None,
            'aggregation_method': self.server.aggregation_method if self.server else None,
            'completed_rounds': len(self.server.training_rounds) if self.server else 0
        }


