"""
wp4_cotraining_tsvm/tsvm.py
=============================
Implements a Transductive SVM (Proposal §4.5) using the switching / iterative
label assignment strategy (Joachims 1999) with C_U annealing.

The key idea: unlabeled-point labels ŷ_j ∈ {0, ..., C-1} are treated as additional
optimization variables. Margin constraints are applied to both D_L and D_U.

Switching & Annealing Algorithm
-------------------------------
1. Standardize all data using StandardScaler fit on X_L ∪ X_U.
2. Train a standard SVM on X_L to obtain initial pseudo-labels ŷ_U for X_U.
3. Start with a small unlabeled regularization weight C_U.
4. While C_U <= C_U_max:
   a. Fit SVM on X_train = [X_L, X_U] with labels [y_L, ŷ_U] and sample weights
      (C_L for labeled points, C_U for unlabeled points).
   b. Check for label switching on X_U: if the SVM predicts a different label
      for an unlabeled point, update ŷ_U and re-fit (up to max_iter).
   c. Double C_U (annealing) to gradually enforce margin separation on X_U.
5. Return the final fitted model.
"""

import warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.base_model import BaseSSLModel


class TSVMClassifier(BaseSSLModel):
    """
    Transductive SVM (multi-class via one-vs-rest / pairwise SVC).

    Parameters
    ----------
    C_L          : regularization weight for labeled points (default 1.0)
    C_U          : initial regularization weight for unlabeled points (default 0.001)
    C_U_max      : maximum unlabeled regularization weight (default 0.5)
    kernel       : SVM kernel ('rbf', 'linear', etc.)
    max_iter     : maximum switching iterations per annealing step
    class_weight : class weights (e.g. 'balanced' or None)
    random_state : random seed for reproducibility
    """

    def __init__(
        self,
        C_L: float = 1.0,
        C_U: float = 0.001,
        C_U_max: float = 0.5,
        kernel: str = "rbf",
        max_iter: int = 5,
        class_weight=None,
        random_state: int = 42,
    ):
        super().__init__(name=f"TSVM(CL={C_L},CU={C_U}→{C_U_max})")
        self.C_L          = C_L
        self.C_U          = C_U
        self.C_U_max      = C_U_max
        self.kernel       = kernel
        self.max_iter     = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

        self._clf = None
        self._scaler = StandardScaler()
        self.n_support_ = None
        self.total_support_vectors_ = None
        self.history_ = []

    def fit(self, X_L: np.ndarray, y_L: np.ndarray, X_U: np.ndarray, y_U_noisy: np.ndarray | None = None):
        """
        Fit a transductive SVM by iterative label switching with C_U annealing.
        y_U_noisy is ignored (TSVM treats unlabeled labels as latent optimization variables).
        """
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        # 1. Scale data across X_L and X_U
        X_all = np.vstack([X_L, X_U])
        self._scaler.fit(X_all)
        X_L_s = self._scaler.transform(X_L)
        X_U_s = self._scaler.transform(X_U)

        # 2. Initial labels for X_U from standard SVM on D_L
        base_svm = SVC(
            C=self.C_L,
            kernel=self.kernel,
            class_weight=self.class_weight,
            decision_function_shape="ovr",
            random_state=self.random_state,
        )
        base_svm.fit(X_L_s, y_L)
        y_U_hat = base_svm.predict(X_U_s)

        # 3. Annealing loop
        C_U = self.C_U
        X_train = np.vstack([X_L_s, X_U_s])
        clf = None
        self.history_ = []

        while C_U <= self.C_U_max:
            # Build sample weights
            weights = np.hstack([
                np.full(len(X_L), self.C_L),
                np.full(len(X_U), C_U),
            ])

            clf = SVC(
                C=1.0,
                kernel=self.kernel,
                class_weight=self.class_weight,
                decision_function_shape="ovr",
                random_state=self.random_state,
            )
            y_train = np.hstack([y_L, y_U_hat])
            clf.fit(X_train, y_train, sample_weight=weights)

            # Inner switching loop
            swaps_in_step = 0
            for _ in range(self.max_iter):
                new_pred = clf.predict(X_U_s)
                diff = (new_pred != y_U_hat)
                n_diff = int(diff.sum())
                if n_diff == 0:
                    break
                y_U_hat[diff] = new_pred[diff]
                swaps_in_step += n_diff
                y_train = np.hstack([y_L, y_U_hat])
                clf.fit(X_train, y_train, sample_weight=weights)

            n_sv = int(np.sum(clf.n_support_))
            self.history_.append({
                "C_U": float(C_U),
                "n_support": n_sv,
                "swaps": swaps_in_step,
            })

            C_U = min(C_U * 2.0, self.C_U_max + 1e-9)

        self._clf = clf
        self.n_support_ = clf.n_support_
        self.total_support_vectors_ = int(np.sum(clf.n_support_))
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions."""
        X_s = self._scaler.transform(X)
        return self._clf.predict(X_s)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return calibrated class probability estimates via softmax over decision function scores.
        """
        X_s = self._scaler.transform(X)
        scores = self._clf.decision_function(X_s)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])

        n_classes = len(self._clf.classes_)
        # Align to standard 5 classes (0..4)
        if n_classes < 5 or not np.array_equal(self._clf.classes_, np.arange(5)):
            aligned = np.full((len(X), 5), -1e9, dtype=float)
            for idx, c in enumerate(self._clf.classes_):
                if 0 <= c < 5:
                    aligned[:, c] = scores[:, idx]
            scores = aligned

        scores = scores - scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum(axis=1, keepdims=True)
