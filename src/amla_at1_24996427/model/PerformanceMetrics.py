import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score, RocCurveDisplay
)

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

def pick_threshold_by_f1(y_true, p_val):
    """Return tau that maximizes F1 on validation."""
    prec, rec, thr = precision_recall_curve(y_true, p_val)
    f1 = (2 * prec * rec) / np.clip(prec + rec, 1e-12, None)
    best_idx = int(np.nanargmax(f1))
    # precision_recall_curve returns thresholds one shorter than prec/rec
    tau = thr[max(0, best_idx - 1)] if len(thr) > 0 else 0.5
    return float(tau)

def report_all(y_true, p, tau):
    """Compute AUROC, PR-AUC, Brier, and confusion-derived metrics at tau."""
    auroc  = roc_auc_score(y_true, p)
    prauc  = average_precision_score(y_true, p)
    brier  = brier_score_loss(y_true, p)
    yhat   = (p >= tau).astype(int)
    prec   = precision_score(y_true, yhat, zero_division=0)
    rec    = recall_score(y_true, yhat, zero_division=0)
    f1     = f1_score(y_true, yhat, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat).ravel()
    return dict(auroc=auroc, prauc=prauc, brier=brier,
                precision=prec, recall=rec, f1=f1,
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

# --- I.2 Plot helpers (use on any fitted model with predict_proba) ---
def plot_roc_pr(y_true, p, title_prefix=""):
    fpr, tpr, _ = roc_curve(y_true, p)
    prec, rec, _ = precision_recall_curve(y_true, p)
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].plot(fpr, tpr); ax[0].plot([0,1],[0,1],'--',lw=1)
    ax[0].set_title(f"{title_prefix} ROC (AUROC={roc_auc_score(y_true,p):.3f})")
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
    ax[1].plot(rec, prec)
    ax[1].set_title(f"{title_prefix} PR (AP={average_precision_score(y_true,p):.3f})")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    plt.tight_layout(); plt.show()
