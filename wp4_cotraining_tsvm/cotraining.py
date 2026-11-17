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

TODO for Pratyush (WP4)
------------------------
Implement the methods marked with TODO.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.base_model import BaseSSLModel

# View index slices (from WP1 feature layout)
VIEW1 = slice(0, 42)
VIEW2 = slice(42, 84)


def _make_base_clf(C: float = 1.0):
    """Default base classifier for each view: logistic regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=1000,
                                  multi_class="multinomial",
                                  solver="lbfgs")),
    ])


class CoTrainer(BaseSSLModel):
    """
    Two-view Co-Training classifier.

    Parameters
    ----------
    n_rounds    : number of co-training iterations
    p           : number of samples to add per class per round (pool size)
    threshold   : minimum confidence to add a sample
    base_clf_fn : callable that returns a fresh sklearn estimator (optional)
    """

    def __init__(
        self,
        n_rounds: int = 30,
        p: int = 5,
        threshold: float = 0.80,
        base_clf_fn=None,
    ):
        super().__init__(name=f"CoTrainer(r={n_rounds},p={p},t={threshold})")
        self.n_rounds  = n_rounds
        self.p         = p
        self.threshold = threshold
        self._clf_fn   = base_clf_fn or _make_base_clf
        self._h1 = None
        self._h2 = None

    def fit(self, X_L, y_L, X_U, y_U_noisy=None):
        """
        Train two classifiers via co-training.
        y_U_noisy is ignored (co-training discards untrusted labels).
        """
        # TODO: Implement.
        # 1. Split features into views: X_L1 = X_L[:, VIEW1], X_L2 = X_L[:, VIEW2]
        # 2. Keep a growing labeled pool for each view:
        #      L1 = (X_L[:, VIEW1].copy(), y_L.copy())
        #      L2 = (X_L[:, VIEW2].copy(), y_L.copy())
        # 3. Keep an unlabeled pool: U = X_U.copy()
        # 4. For each round:
        #      a. Fit self._h1 on L1; get probabilities on U[:, VIEW1].
        #      b. For each class c, pick top-p U indices where h1 predicts c
        #         with prob > threshold.
        #      c. Add those (feature slice VIEW2, h1-predicted label) to L2.
        #      d. Remove those indices from U.
        #      e. Repeat symmetrically for h2 → L1.
        #      f. Break early if no samples were added.
        # 5. Final fit of h1 and h2 on their full labeled pools.
        raise NotImplementedError("CoTrainer.fit not yet implemented")
        self._is_fitted = True
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        """
        Average the probability estimates from both views.
        """
        # TODO: return (self._h1.predict_proba(X[:, VIEW1]) +
        #               self._h2.predict_proba(X[:, VIEW2])) / 2
        raise NotImplementedError("CoTrainer.predict_proba not yet implemented")
