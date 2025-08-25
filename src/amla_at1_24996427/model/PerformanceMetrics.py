import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score

def display_roc_auc_curve(model, X, y):
    # Predict probabilities
    y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of positive class

    # Calculate ROC metrics
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    roc_auc = roc_auc_score(y, y_pred_proba)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X, y)
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.title(f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

