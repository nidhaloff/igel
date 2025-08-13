"""
Tests for few-shot learning functionality in igel.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from igel.few_shot_learning import (
    MAMLClassifier,
    PrototypicalNetwork,
    DomainAdaptation,
    TransferLearning,
    create_few_shot_dataset,
    evaluate_few_shot_model
)


class TestMAMLClassifier:
    """Test cases for MAMLClassifier."""
    
    def setup_method(self):
        """Set up test data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=42
        )
        self.X = X
        self.y = y
    
    def test_maml_initialization(self):
        """Test MAML classifier initialization."""
        maml = MAMLClassifier(
            inner_lr=0.01,
            outer_lr=0.001,
            num_tasks=5,
            shots_per_task=3,
            inner_steps=3,
            meta_epochs=10
        )
        
        assert maml.inner_lr == 0.01
        assert maml.outer_lr == 0.001
        assert maml.num_tasks == 5
        assert maml.shots_per_task == 3
        assert maml.inner_steps == 3
        assert maml.meta_epochs == 10
        assert not maml.is_fitted
    
    def test_maml_fit(self):
        """Test MAML training."""
        maml = MAMLClassifier(
            num_tasks=3,
            shots_per_task=2,
            inner_steps=2,
            meta_epochs=5  # Small number for testing
        )
        
        maml.fit(self.X, self.y)
        
        assert maml.is_fitted
        assert hasattr(maml, 'classes_')
        assert hasattr(maml, 'n_classes_')
        assert len(maml.classes_) == 4
    
    def test_maml_predict(self):
        """Test MAML prediction."""
        maml = MAMLClassifier(
            num_tasks=3,
            shots_per_task=2,
            inner_steps=2,
            meta_epochs=5
        )
        
        maml.fit(self.X, self.y)
        
        # Test prediction
        X_test = self.X[:10]
        predictions = maml.predict(X_test)
        
        assert len(predictions) == 10
        assert all(pred in maml.classes_ for pred in predictions)
    
    def test_maml_predict_proba(self):
        """Test MAML probability prediction."""
        maml = MAMLClassifier(
            num_tasks=3,
            shots_per_task=2,
            inner_steps=2,
            meta_epochs=5
        )
        
        maml.fit(self.X, self.y)
        
        # Test probability prediction
        X_test = self.X[:10]
        probas = maml.predict_proba(X_test)
        
        assert probas.shape == (10, 4)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)


class TestPrototypicalNetwork:
    """Test cases for PrototypicalNetwork."""
    
    def setup_method(self):
        """Set up test data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=42
        )
        self.X = X
        self.y = y
    
    def test_prototypical_network_initialization(self):
        """Test Prototypical Network initialization."""
        proto_net = PrototypicalNetwork(
            embedding_dim=32,
            num_tasks=5,
            shots_per_task=3,
            meta_epochs=10
        )
        
        assert proto_net.embedding_dim == 32
        assert proto_net.num_tasks == 5
        assert proto_net.shots_per_task == 3
        assert proto_net.meta_epochs == 10
        assert not proto_net.is_fitted
    
    def test_prototypical_network_fit(self):
        """Test Prototypical Network training."""
        proto_net = PrototypicalNetwork(
            embedding_dim=16,
            num_tasks=3,
            shots_per_task=2,
            meta_epochs=5  # Small number for testing
        )
        
        proto_net.fit(self.X, self.y)
        
        assert proto_net.is_fitted
        assert hasattr(proto_net, 'classes_')
        assert hasattr(proto_net, 'n_classes_')
        assert len(proto_net.classes_) == 4
    
    def test_prototypical_network_predict(self):
        """Test Prototypical Network prediction."""
        proto_net = PrototypicalNetwork(
            embedding_dim=16,
            num_tasks=3,
            shots_per_task=2,
            meta_epochs=5
        )
        
        proto_net.fit(self.X, self.y)
        
        # Test prediction
        X_test = self.X[:10]
        predictions = proto_net.predict(X_test)
        
        assert len(predictions) == 10


class TestDomainAdaptation:
    """Test cases for DomainAdaptation."""
    
    def setup_method(self):
        """Set up test data."""
        # Create source domain data
        X_source, y_source = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=3,
            random_state=42
        )
        
        # Create target domain data with some shift
        X_target, y_target = make_classification(
            n_samples=50,
            n_features=10,
            n_classes=3,
            random_state=123
        )
        
        self.X_source = X_source
        self.y_source = y_source
        self.X_target = X_target
        self.y_target = y_target
    
    def test_domain_adaptation_initialization(self):
        """Test DomainAdaptation initialization."""
        from sklearn.ensemble import RandomForestClassifier
        
        base_model = RandomForestClassifier(n_estimators=10, random_state=42)
        adapter = DomainAdaptation(base_model)
        
        assert adapter.base_model == base_model
        assert not adapter.is_adapted
    
    def test_fine_tune_adaptation(self):
        """Test fine-tuning domain adaptation."""
        from sklearn.ensemble import RandomForestClassifier
        
        base_model = RandomForestClassifier(n_estimators=10, random_state=42)
        adapter = DomainAdaptation(base_model)
        
        # Train base model on source
        base_model.fit(self.X_source, self.y_source)
        
        # Perform adaptation
        adapted_model = adapter.adapt_model(
            self.X_source, self.y_source,
            self.X_target, self.y_target,
            adaptation_method='fine_tune'
        )
        
        assert adapter.is_adapted
        assert adapted_model is not None
    
    def test_maml_adaptation(self):
        """Test MAML-based domain adaptation."""
        from sklearn.ensemble import RandomForestClassifier
        
        base_model = RandomForestClassifier(n_estimators=10, random_state=42)
        adapter = DomainAdaptation(base_model)
        
        # Perform MAML adaptation
        adapted_model = adapter.adapt_model(
            self.X_source, self.y_source,
            self.X_target, self.y_target,
            adaptation_method='maml'
        )
        
        assert adapter.is_adapted
        assert adapted_model is not None


class TestTransferLearning:
    """Test cases for TransferLearning."""
    
    def setup_method(self):
        """Set up test data."""
        # Create source data
        X_source, y_source = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=4,
            random_state=42
        )
        
        # Create target data
        X_target, y_target = make_classification(
            n_samples=50,
            n_features=10,
            n_classes=3,
            random_state=123
        )
        
        self.X_source = X_source
        self.y_source = y_source
        self.X_target = X_target
        self.y_target = y_target
    
    def test_transfer_learning_initialization(self):
        """Test TransferLearning initialization."""
        from sklearn.ensemble import RandomForestClassifier
        
        source_model = RandomForestClassifier(n_estimators=10, random_state=42)
        transfer = TransferLearning(source_model)
        
        assert transfer.base_model == source_model
    
    def test_feature_extraction_transfer(self):
        """Test feature extraction transfer learning."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Train source model
        source_model = RandomForestClassifier(n_estimators=10, random_state=42)
        source_model.fit(self.X_source, self.y_source)
        
        transfer = TransferLearning(source_model)
        
        # Perform feature extraction transfer
        transfer_model = transfer.create_transfer_model(
            source_model, self.X_target, self.y_target,
            method='feature_extraction'
        )
        
        assert transfer_model is not None
        
        # Test prediction
        predictions = transfer_model.predict(self.X_target[:10])
        assert len(predictions) == 10
    
    def test_fine_tuning_transfer(self):
        """Test fine-tuning transfer learning."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Train source model
        source_model = RandomForestClassifier(n_estimators=10, random_state=42)
        source_model.fit(self.X_source, self.y_source)
        
        transfer = TransferLearning(source_model)
        
        # Perform fine-tuning transfer
        transfer_model = transfer.create_transfer_model(
            source_model, self.X_target, self.y_target,
            method='fine_tuning'
        )
        
        assert transfer_model is not None


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def setup_method(self):
        """Set up test data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=6,
            random_state=42
        )
        self.X = X
        self.y = y
    
    def test_create_few_shot_dataset(self):
        """Test few-shot dataset creation."""
        tasks = create_few_shot_dataset(
            self.X, self.y,
            n_way=3,
            k_shot=5,
            n_query=5
        )
        
        assert len(tasks) > 0
        
        # Check task structure
        support_X, support_y, query_X, query_y = tasks[0]
        
        assert support_X.shape[0] == 15  # 3 classes * 5 shots
        assert query_X.shape[0] == 15   # 3 classes * 5 queries
        assert len(support_y) == 15
        assert len(query_y) == 15
        
        # Check that classes are relabeled as 0, 1, 2
        assert set(support_y) == {0, 1, 2}
        assert set(query_y) == {0, 1, 2}
    
    def test_evaluate_few_shot_model(self):
        """Test few-shot model evaluation."""
        # Create a simple mock model
        class MockModel:
            def __init__(self):
                self.is_fitted = True
            
            def predict(self, X):
                return np.random.randint(0, 3, len(X))
        
        model = MockModel()
        
        # Create tasks
        tasks = create_few_shot_dataset(
            self.X, self.y,
            n_way=2,
            k_shot=3,
            n_query=3
        )
        
        # Evaluate model
        results = evaluate_few_shot_model(model, tasks)
        
        assert 'mean_accuracy' in results
        assert 'std_accuracy' in results
        assert 'accuracies' in results
        assert 0 <= results['mean_accuracy'] <= 1
        assert len(results['accuracies']) == len(tasks)


class TestIntegration:
    """Integration tests for few-shot learning."""
    
    def test_end_to_end_maml(self):
        """Test end-to-end MAML workflow."""
        # Create data
        X, y = make_classification(
            n_samples=300,
            n_features=15,
            n_classes=5,
            random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train MAML
        maml = MAMLClassifier(
            num_tasks=5,
            shots_per_task=3,
            inner_steps=3,
            meta_epochs=10
        )
        
        maml.fit(X_train, y_train)
        
        # Create evaluation tasks
        tasks = create_few_shot_dataset(X_test, y_test, n_way=2, k_shot=3, n_query=3)
        
        # Evaluate
        results = evaluate_few_shot_model(maml, tasks)
        
        assert results['mean_accuracy'] >= 0
        assert results['std_accuracy'] >= 0
    
    def test_end_to_end_prototypical_network(self):
        """Test end-to-end Prototypical Network workflow."""
        # Create data
        X, y = make_classification(
            n_samples=300,
            n_features=15,
            n_classes=5,
            random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Prototypical Network
        proto_net = PrototypicalNetwork(
            embedding_dim=32,
            num_tasks=5,
            shots_per_task=3,
            meta_epochs=10
        )
        
        proto_net.fit(X_train, y_train)
        
        # Create evaluation tasks
        tasks = create_few_shot_dataset(X_test, y_test, n_way=2, k_shot=3, n_query=3)
        
        # Evaluate
        results = evaluate_few_shot_model(proto_net, tasks)
        
        assert results['mean_accuracy'] >= 0
        assert results['std_accuracy'] >= 0


if __name__ == "__main__":
    pytest.main([__file__]) 