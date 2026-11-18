"""
wp4_cotraining_tsvm/cotraining.py
==================================
Implements Multi-View Co-Training (Proposal §4.4).

The two views are defined by WP1's feature layout:
  View 1 → feature indices 0:42  (time-domain / morphological)
  View 2 → feature indices 42:84 (frequency-domain / wavelet)

Algorithm (Blum & Mitchell 1998)
---------------------------------
1. Train classifier h1 on (X_L[:, V1], y_L) and h2 on (X_L[:, V2], y_L).
2. Each round:
   a. h1 picks the top-p unlabeled samples it predicts with highest confidence.
   b. Add those to h2's labeled pool (with h1's label).
   c. h2 does the same for h1.
3. Repeat for T rounds or until no new samples are added.
"""

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.optimize import OptimizeWarning

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.base_model import BaseSSLModel

# View index slices (from WP1 feature layout)
VIEW1 = slice(0, 42)
VIEW2 = slice(42, 84)


def _make_base_clf(C: float = 1.0, random_state: int = 42, class_weight=None):
    """Default base classifier for each view: logistic regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs",
            class_weight=class_weight,
            random_state=random_state,
        )),
    ])


def _align_proba(clf, X: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """Safely obtain probability estimates aligned to classes 0..n_classes-1."""
    raw_proba = clf.predict_proba(X)
    classes = getattr(clf, "classes_", None)
    if classes is None and hasattr(clf, "named_steps") and "lr" in clf.named_steps:
        classes = clf.named_steps["lr"].classes_

    if classes is not None and len(classes) == n_classes and np.array_equal(classes, np.arange(n_classes)):
        return raw_proba

    aligned = np.zeros((len(X), n_classes), dtype=float)
    if classes is not None:
        for col_idx, c in enumerate(classes):
            if 0 <= c < n_classes:
                aligned[:, c] = raw_proba[:, col_idx]
    return aligned


class CoTrainer(BaseSSLModel):
    """
    Two-view Co-Training classifier.

    Parameters
    ----------
    n_rounds     : number of co-training iterations
    p            : number of samples to add per class per round
    threshold    : minimum confidence to add a sample
    base_clf_fn  : callable that returns a fresh sklearn estimator (optional)
    class_weight : class weights for base logistic regression (e.g. 'balanced' or None)
    random_state : random seed for reproducibility
    """

    def __init__(
        self,
        n_rounds: int = 30,
        p: int = 5,
        threshold: float = 0.80,
        base_clf_fn=None,
        class_weight=None,
        random_state: int = 42,
    ):
        super().__init__(name=f"CoTrainer(r={n_rounds},p={p},t={threshold})")
        self.n_rounds     = n_rounds
        self.p            = p
        self.threshold    = threshold
        self.class_weight = class_weight
        self.random_state = random_state
        self._clf_fn      = base_clf_fn or (
            lambda: _make_base_clf(random_state=self.random_state, class_weight=self.class_weight)
        )
        self._h1 = None
        self._h2 = None
        self.history_ = []

    def fit(self, X_L: np.ndarray, y_L: np.ndarray, X_U: np.ndarray, y_U_noisy: np.ndarray | None = None):
        """
        Train two classifiers via co-training.
        y_U_noisy is ignored (co-training discards untrusted labels).
        """
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        # 1. Split features into views
        X_L1 = X_L[:, VIEW1].copy()
        X_L2 = X_L[:, VIEW2].copy()
        X_U1 = X_U[:, VIEW1].copy()
        X_U2 = X_U[:, VIEW2].copy()

        # 2. Keep growing labeled pools for each view
        L1_X, L1_y = X_L1.copy(), y_L.copy()
        L2_X, L2_y = X_L2.copy(), y_L.copy()

        # 3. Unlabeled pool tracking
        remaining = np.arange(len(X_U))

        self._h1 = self._clf_fn()
        self._h2 = self._clf_fn()
        self.history_ = []

        n_classes = len(np.unique(y_L)) if len(y_L) > 0 else 5
        n_classes = max(5, n_classes)

        for r in range(self.n_rounds):
            if len(remaining) == 0:
                break

            self._h1.fit(L1_X, L1_y)
            self._h2.fit(L2_X, L2_y)

            # Record inter-view agreement rate on the full X_U pool
            prob_all_1 = _align_proba(self._h1, X_U[:, VIEW1], n_classes)
            prob_all_2 = _align_proba(self._h2, X_U[:, VIEW2], n_classes)
            pred_all_1 = np.argmax(prob_all_1, axis=1)
            pred_all_2 = np.argmax(prob_all_2, axis=1)
            agreement  = float((pred_all_1 == pred_all_2).mean())

            # Evaluate probabilities on remaining unlabeled samples
            proba1 = _align_proba(self._h1, X_U1[remaining], n_classes)
            proba2 = _align_proba(self._h2, X_U2[remaining], n_classes)

            added_to_L2_indices = []
            added_to_L2_labels  = []
            added_to_L1_indices = []
            added_to_L1_labels  = []

            # Classifier h1 (View 1) selects high-confidence samples to add to L2 (View 2)
            for c in range(n_classes):
                conf = proba1[:, c]
                top_idx = np.argsort(conf)[-self.p:][::-1]
                top_idx = [idx for idx in top_idx if conf[idx] >= self.threshold]
                for local_i in top_idx:
                    global_i = remaining[local_i]
                    added_to_L2_indices.append(global_i)
                    added_to_L2_labels.append(c)

            # Classifier h2 (View 2) selects high-confidence samples to add to L1 (View 1)
            for c in range(n_classes):
                conf = proba2[:, c]
                top_idx = np.argsort(conf)[-self.p:][::-1]
                top_idx = [idx for idx in top_idx if conf[idx] >= self.threshold]
                for local_i in top_idx:
                    global_i = remaining[local_i]
                    added_to_L1_indices.append(global_i)
                    added_to_L1_labels.append(c)

            added_all = list(set(added_to_L2_indices + added_to_L1_indices))

            self.history_.append({
                "round": r + 1,
                "agreement": agreement,
                "added_h1": len(added_to_L2_indices),
                "added_h2": len(added_to_L1_indices),
                "total_added": len(added_all),
                "pool1_size": len(L1_y),
                "pool2_size": len(L2_y),
                "remaining": len(remaining),
            })

            if len(added_all) == 0:
                break

            if len(added_to_L2_indices) > 0:
                L2_X = np.vstack([L2_X, X_U[added_to_L2_indices, VIEW2]])
                L2_y = np.append(L2_y, added_to_L2_labels)

            if len(added_to_L1_indices) > 0:
                L1_X = np.vstack([L1_X, X_U[added_to_L1_indices, VIEW1]])
                L1_y = np.append(L1_y, added_to_L1_labels)

            remaining = np.setdiff1d(remaining, added_all)

        # Final fit of h1 and h2 on full enriched pools
        self._h1.fit(L1_X, L1_y)
        self._h2.fit(L2_X, L2_y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions using averaged ensemble probabilities."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average the probability estimates from both views."""
        p1 = _align_proba(self._h1, X[:, VIEW1])
        p2 = _align_proba(self._h2, X[:, VIEW2])
        return (p1 + p2) / 2.0

    def compute_agreement(self, X: np.ndarray) -> float:
        """Compute the inter-view agreement rate on feature matrix X."""
        p1 = np.argmax(_align_proba(self._h1, X[:, VIEW1]), axis=1)
        p2 = np.argmax(_align_proba(self._h2, X[:, VIEW2]), axis=1)
        return float((p1 == p2).mean())
