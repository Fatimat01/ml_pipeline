import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'workclass': ['Private', 'Private', 'Self-emp', 'Private', 'Government'],
        'education': ['Bachelors', 'HS-grad', 'Masters', 'Bachelors', 'Doctorate'],
        'salary': ['<=50K', '>50K', '>50K', '<=50K', '>50K']
    })
    return data


@pytest.fixture
def processed_data(sample_data):
    """Process sample data for testing."""
    cat_features = ['workclass', 'education']
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label='salary',
        training=True
    )
    return X, y, encoder, lb


def test_train_model_returns_correct_type(processed_data):
    """Test that train_model returns a RandomForestClassifier instance."""
    X, y, _, _ = processed_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "Model should be RandomForestClassifier"


def test_compute_model_metrics_returns_correct_types(processed_data):
    """Test that compute_model_metrics returns three floats."""
    X, y, _, _ = processed_data
    model = train_model(X, y)
    preds = inference(model, X)
    
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    assert isinstance(precision, (float, np.floating)), "Precision should be float"
    assert isinstance(recall, (float, np.floating)), "Recall should be float"
    assert isinstance(fbeta, (float, np.floating)), "F-beta should be float"


def test_compute_model_metrics_values_in_valid_range(processed_data):
    """Test that metrics are in the valid range [0, 1]."""
    X, y, _, _ = processed_data
    model = train_model(X, y)
    preds = inference(model, X)
    
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= fbeta <= 1, "F-beta should be between 0 and 1"


def test_inference_returns_correct_shape(processed_data):
    """Test that inference returns predictions with correct shape."""
    X, y, _, _ = processed_data
    model = train_model(X, y)
    preds = inference(model, X)
    
    assert preds.shape[0] == X.shape[0], "Number of predictions should match number of samples"


def test_process_data_returns_correct_types(sample_data):
    """Test that process_data returns correct types."""
    cat_features = ['workclass', 'education']
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label='salary',
        training=True
    )
    
    assert isinstance(X, np.ndarray), "X should be numpy array"
    assert isinstance(y, np.ndarray), "y should be numpy array"
    assert isinstance(encoder, OneHotEncoder), "encoder should be OneHotEncoder"
    assert isinstance(lb, LabelBinarizer), "lb should be LabelBinarizer"


def test_process_data_training_mode(sample_data):
    """Test process_data in training mode creates and fits encoders."""
    cat_features = ['workclass', 'education']
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label='salary',
        training=True
    )
    
    # Check that encoder was fitted
    assert hasattr(encoder, 'categories_'), "Encoder should be fitted"
    # Check that label binarizer was fitted
    assert hasattr(lb, 'classes_'), "Label binarizer should be fitted"

