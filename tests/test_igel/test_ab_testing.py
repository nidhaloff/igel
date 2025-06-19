"""Tests for the A/B testing functionality."""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pytest

from igel.ab_testing import ModelComparison

@pytest.fixture
def classification_data():
    """Generate sample classification data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def regression_data():
    """Generate sample regression data."""
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_classification_comparison(classification_data):
    """Test model comparison for classification problems."""
    X_train, X_test, y_train, y_test = classification_data
    
    # Create two models with different parameters
    model_a = RandomForestClassifier(n_estimators=100, random_state=42)
    model_b = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # Train models
    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)
    
    # Compare models
    comparison = ModelComparison(model_a, model_b, test_type="classification")
    results = comparison.compare_predictions(X_test, y_test)
    
    # Basic assertions
    assert "model_a_metrics" in results
    assert "model_b_metrics" in results
    assert "statistical_test" in results
    assert "accuracy" in results["model_a_metrics"]
    assert "accuracy" in results["model_b_metrics"]
    assert results["model_a_metrics"]["accuracy"] > 0
    assert results["model_b_metrics"]["accuracy"] > 0

def test_regression_comparison(regression_data):
    """Test model comparison for regression problems."""
    X_train, X_test, y_train, y_test = regression_data
    
    # Create two models with different parameters
    model_a = RandomForestRegressor(n_estimators=100, random_state=42)
    model_b = RandomForestRegressor(n_estimators=50, random_state=42)
    
    # Train models
    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)
    
    # Compare models
    comparison = ModelComparison(model_a, model_b, test_type="regression")
    results = comparison.compare_predictions(X_test, y_test)
    
    # Basic assertions
    assert "model_a_metrics" in results
    assert "model_b_metrics" in results
    assert "statistical_test" in results
    assert "mse" in results["model_a_metrics"]
    assert "r2" in results["model_a_metrics"]
    assert "mse" in results["model_b_metrics"]
    assert "r2" in results["model_b_metrics"]

def test_report_generation(classification_data):
    """Test report generation functionality."""
    X_train, X_test, y_train, y_test = classification_data
    
    # Create and train models
    model_a = RandomForestClassifier(n_estimators=100, random_state=42)
    model_b = RandomForestClassifier(n_estimators=50, random_state=42)
    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)
    
    # Generate report
    comparison = ModelComparison(model_a, model_b, test_type="classification")
    results = comparison.compare_predictions(X_test, y_test)
    report = comparison.generate_report(results)
    
    # Check report content
    assert isinstance(report, str)
    assert "Model A/B Testing Results" in report
    assert "Statistical Test Results" in report
    assert "Model A Accuracy" in report
    assert "Model B Accuracy" in report
    assert "P-value" in report 