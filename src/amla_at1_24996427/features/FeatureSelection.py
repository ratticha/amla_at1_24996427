
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, f_regression

def compute_feature_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute feature importance scores using F-statistics, mutual information, and correlation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.

    Returns
    -------
    pd.DataFrame
        DataFrame containing F-score, Mutual Information, and Correlation for each feature,
        sorted by Mutual Information in descending order.
    """
    # --- Compute F-statistics and Mutual Information ---
    f_stats, _ = f_regression(X, y)
    mi_scores = mutual_info_regression(X, y)
    correlations = X.corrwith(y)

    # --- Combine into a single DataFrame ---
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'F_score': f_stats,
        'Mutual_Info': mi_scores,
        'Correlation': correlations
    }).sort_values('Mutual_Info', ascending=False).reset_index(drop=True)

    return feature_scores
