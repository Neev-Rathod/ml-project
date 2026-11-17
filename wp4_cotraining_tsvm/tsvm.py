"""
wp4_cotraining_tsvm/tsvm.py
=============================
Implements a Transductive SVM (Proposal §4.5) using the switching / iterative
label assignment strategy (Joachims 1999).

The key idea: unlabeled-point labels ŷ_j ∈ {-1, +1} are treated as additional
optimization variables.  We use a one-vs-rest decomposition for multi-class.

Switching algorithm
--------------------
1. Train a standard SVM on D_L.
2. Assign initial labels ŷ_U = predicted classes from step 1.
3. Train SVM on D_L ∪ D_U (with ŷ_U).
4. Find pairs (i, j) where:
     i ∈ D_U with ŷ_i = +1, margin_i < 1
     j ∈ D_U with ŷ_j = -1, margin_j < 1
     swapping labels improves objective
5. Swap those labels and re-train.
6. Repeat until no beneficial swaps exist.

TODO for Pratyush (WP4)
------------------------
Implement the methods marked with TODO.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.base_model import BaseSSLModel


class TSVMClassifier(BaseSSLModel):
    """
    Transductive SVM (multi-class via one-vs-rest).

    Parameters
    ----------
    C_L     : regularization weight for labeled points
    C_U     : regularization weight for unlabeled points (start small, anneal up)
    kernel  : SVM kernel (default 'rbf')
    max_iter: maximum switching iterations
    """

    def __init__(
        self,
        C_L: float = 1.0,
        C_U: float = 0.001,
        C_U_max: float = 1.0,
        kernel: str = "rbf",
        max_iter: int = 10,
    ):
        super().__init__(
            name=f"TSVM(CL={C_L},CU={C_U}→{C_U_max})"
        )
        self.C_L     = C_L
        self.C_U     = C_U
        self.C_U_max = C_U_max
        self.kernel  = kernel
        self.max_iter = max_iter
        self._clf = None
        self._scaler = StandardScaler()

    def fit(self, X_L, y_L, X_U, y_U_noisy=None):
        """
        Fit a transductive SVM by iterative label switching.
        y_U_noisy is ignored.
        """
        # TODO: Implement the switching algorithm.
        # Recommended implementation using C_U annealing:
        #
        # 1. Scale all data with self._scaler.fit(vstack([X_L, X_U])).
        # 2. Train standard SVM (C=C_L) on X_L_scaled to get initial ŷ_U.
        # 3. Set C_U = self.C_U (small initial value).
        # 4. While C_U <= C_U_max:
        #    a. Train SVM on X_L + X_U with sample_weight:
        #         labeled   → weight = 1.0 * C_L
        #         unlabeled → weight = C_U
        #    b. Run switching: for each misclassified unlabeled pair (one +1,
        #       one -1), check if swapping improves the objective;
        #       if yes, swap and restart inner loop.
        #    c. Double C_U.
        # 5. Store final fitted model as self._clf.
        #
        # NOTE: sklearn SVC supports sample_weight in fit().
        # For multi-class, wrap in OneVsRestClassifier.
        raise NotImplementedError("TSVMClassifier.fit not yet implemented")
        self._is_fitted = True
        return self

    def predict(self, X):
        X_s = self._scaler.transform(X)
        return self._clf.predict(X_s)

    def predict_proba(self, X):
        X_s = self._scaler.transform(X)
        # OneVsRestClassifier with probability=True gives decision-function
        # scores that can be softmax-normalised.
        scores = self._clf.decision_function(X_s)
        # Softmax normalisation
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_s  = np.exp(scores)
        return exp_s / exp_s.sum(axis=1, keepdims=True)
