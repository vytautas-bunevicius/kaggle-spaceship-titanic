"""Unit tests for the model_utils module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.spaceship_titanic.model_utils import evaluate_model, extract_feature_importances

@pytest.fixture(name="mock_data")
def fixture_mock_data():
    """Create mock data for testing."""
    x, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

@pytest.fixture(name="mock_model")
def fixture_mock_model(mock_data):
    """Create and train a mock model for testing."""
    x_train, _, y_train, _ = mock_data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model

def test_evaluate_model_basic(mock_model, mock_data):
    """Test basic functionality of evaluate_model."""
    _, x_test, _, y_test = mock_data
    result = evaluate_model(mock_model, x_test, y_test)

    assert isinstance(result, dict)
    assert all(metric in result for metric in [
        'roc_auc', 'pr_auc', 'f1', 'precision', 'recall', 'balanced_accuracy',
        'threshold', 'y_pred', 'y_pred_proba'
    ])
    assert 0 <= result['roc_auc'] <= 1
    assert 0 <= result['pr_auc'] <= 1
    assert 0 <= result['f1'] <= 1
    assert 0 <= result['precision'] <= 1
    assert 0 <= result['recall'] <= 1
    assert 0 <= result['balanced_accuracy'] <= 1
    assert result['threshold'] == 0.5
    assert isinstance(result['y_pred'], np.ndarray)
    assert isinstance(result['y_pred_proba'], np.ndarray)

def test_evaluate_model_custom_threshold(mock_model, mock_data):
    """Test evaluate_model with a custom threshold."""
    _, x_test, _, y_test = mock_data
    result = evaluate_model(mock_model, x_test, y_test, threshold=0.7)

    assert result['threshold'] == 0.7
    assert all(pred in [0, 1] for pred in result['y_pred'])

def test_evaluate_model_target_recall(mock_model, mock_data):
    """Test evaluate_model with a target recall."""
    _, x_test, _, y_test = mock_data
    target_recall = 0.8
    result = evaluate_model(mock_model, x_test, y_test, target_recall=target_recall)

    assert abs(result['recall'] - target_recall) < 0.1  # Allow small deviation

def test_extract_feature_importances_random_forest(mock_model, mock_data):
    """Test extract_feature_importances with a RandomForestClassifier."""
    x_train, _, y_train, _ = mock_data
    importances = extract_feature_importances(mock_model, pd.DataFrame(x_train), y_train)

    assert isinstance(importances, np.ndarray)
    assert len(importances) == x_train.shape[1]
    assert all(importance >= 0 for importance in importances)

def test_extract_feature_importances_non_feature_importance_model(mock_data):
    """Test extract_feature_importances with a model that doesn't have feature_importances_."""
    x_train, _, y_train, _ = mock_data

    class DummyModel:
        """A dummy model for testing purposes."""

        def __init__(self):
            """Initialize the DummyModel."""
            self.x = None
            self.y = None

        def fit(self, x, y):
            """Dummy fit method."""
            self.x = x
            self.y = y
            return self

        def predict(self, x):
            """Dummy predict method."""
            return np.random.randint(0, 2, size=len(x))

        def score(self, x, y):
            """Dummy score method."""
            predictions = self.predict(x)
            return accuracy_score(y, predictions)

    dummy_model = DummyModel()
    dummy_model.fit(x_train, y_train)
    importances = extract_feature_importances(dummy_model, pd.DataFrame(x_train), y_train)

    assert isinstance(importances, np.ndarray)
    assert len(importances) == x_train.shape[1]
    assert all(importance >= 0 for importance in importances)
if __name__ == "__main__":
    pytest.main()
