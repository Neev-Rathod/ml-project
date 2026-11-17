"""
wp2_synthetic_baselines/baselines.py
======================================
Implements the five supervised baseline classifiers (Proposal §4.2).

Each classifier is wrapped in a thin class that inherits BaseSSLModel so it
plugs into WP5's evaluation suite.

Classifiers
-----------
1. GaussianNaiveBayes
2. LogisticRegressionModel
3. KNNClassifierModel
4. DecisionTreeModel
5. SVMModel

Training modes (each class supports all three):
  gold_only    – fit on D_L only
  naive_pool   – fit on D_L ∪ D_U, treating noisy labels as clean
  self_training – iterative self-labelling from D_L

TODO for Arnav (WP2)
---------------------
1. Implement the fit() method in each class (follow the TODO comments).
2. Implement run_breakdown_analysis() in run_wp2.py.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.base_model import BaseSSLModel


# ── Helper: self-training loop ────────────────────────────────────────────

def self_training_fit(
    base_clf,
    X_L: np.ndarray,
    y_L: np.ndarray,
    X_U: np.ndarray,
    threshold: float = 0.90,
    max_iter: int = 20,
):
    """
    Generic self-training wrapper.
    In each round, adds unlabeled samples with max predicted probability
    above `threshold` to the labeled pool.
    Returns the fitted classifier.
    """
    # TODO: Implement.
    # Steps:
    #   1. Fit base_clf on (X_L, y_L).
    #   2. Predict probabilities on X_U.
    #   3. Identify rows where max(proba) > threshold.
    #   4. Move those rows to the labeled pool with their predicted label.
    #   5. Repeat until no new samples added or max_iter reached.
    raise NotImplementedError("self_training_fit not yet implemented")


# ── 1. Gaussian Naïve Bayes ────────────────────────────────────────────────

class GaussianNaiveBayes(BaseSSLModel):
    """Gaussian Naïve Bayes with three training modes."""

    def __init__(self, mode: str = "gold_only"):
        super().__init__(name=f"GNB-{mode}")
        assert mode in ("gold_only", "naive_pool", "self_training")
        self.mode = mode
        self._clf = GaussianNB()

    def fit(self, X_L, y_L, X_U, y_U_noisy=None):
        # TODO: Implement the three modes using self._clf.
        # gold_only    → fit on X_L, y_L
        # naive_pool   → fit on vstack([X_L, X_U]), hstack([y_L, y_U_noisy])
        # self_training → call self_training_fit(self._clf, ...)
        raise NotImplementedError
        self._is_fitted = True
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)


# ── 2. Logistic Regression ─────────────────────────────────────────────────

class LogisticRegressionModel(BaseSSLModel):

    def __init__(self, mode: str = "gold_only", C: float = 1.0):
        super().__init__(name=f"LR-{mode}")
        self.mode = mode
        self._clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=1000,
                                      multi_class="multinomial",
                                      solver="lbfgs")),
        ])

    def fit(self, X_L, y_L, X_U, y_U_noisy=None):
        # TODO: Same three-mode pattern as GaussianNaiveBayes.
        raise NotImplementedError
        self._is_fitted = True
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)


# ── 3. k-NN ───────────────────────────────────────────────────────────────

class KNNClassifierModel(BaseSSLModel):

    def __init__(self, mode: str = "gold_only", k: int = 7):
        super().__init__(name=f"kNN(k={k})-{mode}")
        self.mode = mode
        self._clf = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ])

    def fit(self, X_L, y_L, X_U, y_U_noisy=None):
        # TODO
        raise NotImplementedError
        self._is_fitted = True
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)


# ── 4. Decision Tree ──────────────────────────────────────────────────────

class DecisionTreeModel(BaseSSLModel):

    def __init__(self, mode: str = "gold_only",
                 max_depth: int | None = None):
        super().__init__(name=f"DT(depth={max_depth})-{mode}")
        self.mode = mode
        self._clf = DecisionTreeClassifier(max_depth=max_depth,
                                           random_state=42)

    def fit(self, X_L, y_L, X_U, y_U_noisy=None):
        # TODO: Also record tree depth and n_leaves for the analysis.
        raise NotImplementedError
        self._is_fitted = True
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)


# ── 5. SVM ────────────────────────────────────────────────────────────────

class SVMModel(BaseSSLModel):

    def __init__(self, mode: str = "gold_only", C: float = 1.0):
        super().__init__(name=f"SVM(C={C})-{mode}")
        self.mode = mode
        # probability=True required for predict_proba
        self._clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=C, kernel="rbf", probability=True)),
        ])

    def fit(self, X_L, y_L, X_U, y_U_noisy=None):
        # TODO: Same three-mode pattern.
        # TIP: self-training with SVC is slow; limit max_iter to 5–10.
        raise NotImplementedError
        self._is_fitted = True
        return self

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        return self._clf.predict_proba(X)
