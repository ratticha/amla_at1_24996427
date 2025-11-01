import pytest
import pandas as pd
import numpy as np
from amla_at1_24996427.features.FeatureSelection import *

@pytest.fixture
def sample_data():
    """Create a small synthetic regression dataset for testing."""
    np.random.seed(42)
    X = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "feature_2": np.random.randn(100) * 0.5,
        "feature_3": np.random.rand(100) * 10
    })
    y = 3 * X["feature_1"] + 0.5 * X["feature_2"] + np.random.randn(100)
    return X, y

def test_output_is_dataframe(sample_data):
    """Ensure the function returns a pandas DataFrame."""
    X, y = sample_data
    result = compute_feature_scores(X, y)
    assert isinstance(result, pd.DataFrame)

def test_dataframe_columns(sample_data):
    """Check that output DataFrame has correct columns."""
    X, y = sample_data
    result = compute_feature_scores(X, y)
    expected_cols = {"Feature", "F_score", "Mutual_Info", "Correlation"}
    assert expected_cols.issubset(result.columns)

def test_dataframe_length(sample_data):
    """Ensure number of rows matches number of features."""
    X, y = sample_data
    result = compute_feature_scores(X, y)
    assert len(result) == X.shape[1]

def test_feature_names_match(sample_data):
    """Verify that feature names are preserved correctly."""
    X, y = sample_data
    result = compute_feature_scores(X, y)
    assert set(result["Feature"]) == set(X.columns)

def test_sorted_by_mutual_info(sample_data):
    """Ensure the output is sorted by Mutual_Info in descending order."""
    X, y = sample_data
    result = compute_feature_scores(X, y)
    assert result["Mutual_Info"].is_monotonic_decreasing
