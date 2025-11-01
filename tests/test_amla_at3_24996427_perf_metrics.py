import pytest
import numpy as np
from io import StringIO
import sys
import warnings

# Import metrics for reference
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# Import your function
from amla_at1_24996427.models.PerformanceMetrics import *


# ---------------------------------------------------------------------
# 1. TEST: basic correctness of metrics
# ---------------------------------------------------------------------
def test_evaluate_regression_basic(capsys):
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    result = evaluate_regression(y_true, y_pred, label='Validation')

    # Check returned dict keys
    assert set(result.keys()) == {'MAE', 'RMSE', 'MAPE', 'R2'}

    # Check values against sklearn metrics
    expected_mae = mean_absolute_error(y_true, y_pred)
    expected_rmse = root_mean_squared_error(y_true, y_pred)
    expected_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    expected_r2 = r2_score(y_true, y_pred)

    assert result['MAE'] == pytest.approx(expected_mae)
    assert result['RMSE'] == pytest.approx(expected_rmse)
    assert result['MAPE'] == pytest.approx(expected_mape)
    assert result['R2'] == pytest.approx(expected_r2)

    # Verify print output
    captured = capsys.readouterr().out
    assert "Validation" in captured
    assert "MAE=" in captured
    assert "R²=" in captured


# ---------------------------------------------------------------------
# 2. TEST: edge case — perfect prediction
# ---------------------------------------------------------------------
def test_evaluate_regression_perfect():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])

    result = evaluate_regression(y_true, y_pred, label='Perfect')
    assert result['MAE'] == 0
    assert result['RMSE'] == 0
    assert result['MAPE'] == 0
    assert result['R2'] == 1


# ---------------------------------------------------------------------
# 3. TEST: edge case — all zeros in y_true
# ---------------------------------------------------------------------
def test_evaluate_regression_zero_true():
    y_true = np.zeros(4)
    y_pred = np.array([0.1, -0.2, 0.3, -0.4])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = evaluate_regression(y_true, y_pred, label='ZeroTrue')

    assert np.isnan(result['MAPE']) or np.isinf(result['MAPE'])
    assert result['MAE'] >= 0
    assert result['RMSE'] >= 0


# ---------------------------------------------------------------------
# 4. TEST: mismatched lengths (should raise ValueError)
# ---------------------------------------------------------------------
def test_evaluate_regression_mismatched_length():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2])
    with pytest.raises(ValueError):
        evaluate_regression(y_true, y_pred)


# ---------------------------------------------------------------------
# 5. TEST: handles pandas Series as input
# ---------------------------------------------------------------------
import pandas as pd

def test_evaluate_regression_pandas_input():
    y_true = pd.Series([10, 20, 30])
    y_pred = pd.Series([12, 18, 33])
    result = evaluate_regression(y_true, y_pred)
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values())
