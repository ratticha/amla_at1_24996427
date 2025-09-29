import pytest
import numpy as np
import matplotlib.pyplot as plt
from amla_at1_24996427.models.PerformanceMetrics import *

# --- Fixtures ---
@pytest.fixture
def simple_data():
    # Binary classification ground truth and predicted probabilities
    y_true = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    return y_true, p


# --- Tests for pick_threshold_by_f1 ---
def test_pick_threshold_returns_float(simple_data):
    y_true, p = simple_data
    tau = pick_threshold_by_f1(y_true, p)
    assert isinstance(tau, float)

def test_pick_threshold_value_range(simple_data):
    y_true, p = simple_data
    tau = pick_threshold_by_f1(y_true, p)
    assert 0.0 <= tau <= 1.0

def test_pick_threshold_all_same_predictions():
    y_true = np.array([0, 1, 1, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    tau = pick_threshold_by_f1(y_true, p)
    # No useful threshold, should fall back gracefully
    assert tau == 0.5


# --- Tests for report_all ---
def test_report_all_keys_exist(simple_data):
    y_true, p = simple_data
    tau = 0.4
    report = report_all(y_true, p, tau)
    expected_keys = {
        "auroc", "prauc", "brier", "precision", "recall", "f1", "tp", "fp", "fn", "tn"
    }
    assert expected_keys.issubset(report.keys())

def test_report_all_metrics_valid_range(simple_data):
    y_true, p = simple_data
    tau = 0.4
    report = report_all(y_true, p, tau)
    assert 0.0 <= report["auroc"] <= 1.0
    assert 0.0 <= report["prauc"] <= 1.0
    assert 0.0 <= report["precision"] <= 1.0
    assert 0.0 <= report["recall"] <= 1.0
    assert 0.0 <= report["f1"] <= 1.0
    assert report["tp"] >= 0 and report["tn"] >= 0
    assert report["fp"] >= 0 and report["fn"] >= 0

def test_report_all_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.1, 0.9, 1.0])
    tau = 0.5
    report = report_all(y_true, p, tau)
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["f1"] == 1.0
    assert report["fp"] == 0
    assert report["fn"] == 0


# --- Tests for plot_roc_pr ---
def test_plot_roc_pr_runs_without_error(simple_data):
    y_true, p = simple_data
    # Disable interactive mode for matplotlib to avoid blocking
    plt.ioff()
    plot_roc_pr(y_true, p, title_prefix="Test")
    plt.close("all")  # Cleanup figures

def test_plot_roc_pr_creates_two_axes(simple_data, monkeypatch):
    y_true, p = simple_data

    # Monkeypatch plt.show to prevent closing
    monkeypatch.setattr(plt, "show", lambda: None)

    plt.ioff()
    plot_roc_pr(y_true, p, title_prefix="Check Axes")
    fig = plt.gcf()
    assert len(fig.axes) == 2
    plt.close("all")
