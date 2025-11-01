import numpy as np
import pandas as pd

class BaselineClassifier:
    """
    Class used as a baseline model for classification problems.

    Attributes
    ----------
    y : array-like
        Target variable seen during fit.
    pred_value : scalar
        Majority-class label used for prediction.
    preds : ndarray
        Predicted labels (majority class).
    classes_ : ndarray
        Unique class labels seen during fit (sorted).
    class_priors_ : ndarray
        Empirical class probabilities P(y=c) from training data, aligned to classes_.

    Methods
    -------
    fit(y)
        Store the target variable and compute majority class and empirical priors.
    predict(y_like)
        Return the majority class for each requested sample.
    predict_proba(y_like)
        Return constant class probabilities equal to training priors.
    fit_predict(y)
        Fit then predict on the training targets' shape.
    """

    def __init__(self):
        self.y = None
        self.pred_value = None
        self.preds = None
        self.classes_ = None
        self.class_priors_ = None
        self._fitted = False

    def fit(self, y):
        # Accept pandas Series/DataFrame or numpy array
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = np.asarray(y).ravel()
        else:
            y_arr = np.asarray(y).ravel()

        if y_arr.size == 0:
            raise ValueError("y must be non-empty")

        # Classes and priors
        classes, counts = np.unique(y_arr, return_counts=True)
        priors = counts / counts.sum()

        # Majority class (tie-breaker: smallest label by np.unique order)
        majority_idx = np.argmax(counts)
        self.pred_value = classes[majority_idx]

        self.y = y_arr
        self.classes_ = classes
        self.class_priors_ = priors
        self._fitted = True
        return self

    def predict(self, y_like):
        if not self._fitted:
            raise RuntimeError("Must call fit before predict.")
        n = len(y_like)
        self.preds = np.full(shape=(n,), fill_value=self.pred_value, dtype=self.classes_.dtype)
        return self.preds

    def predict_proba(self, y_like):
        """
        Return constant probabilities equal to training priors for each sample.

        Shape: (n_samples, n_classes) aligned with self.classes_.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit before predict_proba.")
        n = len(y_like)
        # Tile priors for each requested sample
        return np.tile(self.class_priors_, (n, 1))

    def fit_predict(self, y):
        self.fit(y)
        return self.predict(y)
