"""
wp5_latent_noise_em/latent_noise_em.py
=======================================
Implements the Latent-Noise EM algorithm (Proposal §4.6).

Model
-----
- True class z_j ∈ {0,…,C-1} is a hidden variable.
- Noisy label ỹ_j ∈ {0,…,C-1} is an observed corrupted copy.
- Confusion matrix T[c, k] = P(ỹ = k | z = c).
- Classifier P(z = c | x; θ) = softmax output of a logistic regression.

EM iterations
-------------
E-step (Eq. 4.4):
    γ_jc = T[c, ỹ_j] * P(z=c | x_j; θ)
            ─────────────────────────────
            Σ_{c'} T[c', ỹ_j] * P(z=c' | x_j; θ)

M-step (Eq. 4.5):
    θ ← weighted MLE using γ_jc as sample weights
    T[c, k] ← (α_ck - 1 + Σ_{j: ỹ_j=k} γ_jc)
               ──────────────────────────────────
               Σ_{k'} (α_ck' - 1 + Σ_{j: ỹ_j=k'} γ_jc)

Gold Set records always have γ_ic = 1[y_i == c].

TODO for Tanuj (WP5)
---------------------
Implement the two methods marked with TODO.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.base_model import BaseSSLModel


class LatentNoiseEM(BaseSSLModel):
    """
    Latent-Noise EM classifier (Dawid & Skene 1979 / Proposal §4.6).

    Parameters
    ----------
    n_classes    : C = 5 for PTB-XL
    max_iter     : maximum EM iterations
    tol          : convergence threshold on T
    alpha        : Dirichlet smoothing parameter (≥ 1)
    C_lr         : regularization for the logistic regression classifier
    """

    def __init__(
        self,
        n_classes: int = 5,
        max_iter: int = 30,
        tol: float = 1e-4,
        alpha: float = 1.5,
        C_lr: float = 1.0,
    ):
        super().__init__(name=f"LatentNoiseEM(α={alpha})")
        self.n_classes = n_classes
        self.max_iter  = max_iter
        self.tol       = tol
        self.alpha     = alpha
        self.C_lr      = C_lr

        # Learned parameters (set after fit)
        self.T_hat_    = None   # (C, C) confusion matrix
        self._clf      = None   # fitted classifier

    def fit(self, X_L, y_L, X_U, y_U_noisy=None):
        """
        Fit using the EM algorithm.

        Parameters
        ----------
        X_L        : Gold Set features (n_L, d)
        y_L        : Clean Gold Set labels (n_L,) int
        X_U        : Untrusted Pool features (n_U, d)
        y_U_noisy  : Noisy labels for X_U (n_U,) int   ← MUST be provided
        """
        if y_U_noisy is None:
            raise ValueError("LatentNoiseEM requires noisy labels y_U_noisy.")

        C = self.n_classes
        n_L = len(X_L)
        n_U = len(X_U)

        # Scale features
        scaler = StandardScaler()
        X_all  = np.vstack([X_L, X_U])
        X_all  = scaler.fit_transform(X_all)
        X_L_s, X_U_s = X_all[:n_L], X_all[n_L:]
        self._scaler = scaler

        # TODO: Implement the EM algorithm.
        #
        # Initialization:
        #   T = (1 - epsilon_init) * I  +  epsilon_init / (C-1) * (1 - I)
        #   where epsilon_init = 0.1  (slight off-diagonal noise)
        #
        # For each EM iteration:
        #   --- E-step ---
        #   1. Fit logistic regression on (X_L_s, y_L) with weights all=1,
        #      and on (X_U_s, y_U_noisy) with weights = γ_jc (from previous M-step).
        #      In the first iteration, use uniform γ for X_U.
        #   2. Compute P(z=c | x_j; θ) = clf.predict_proba(X_U_s)   shape (n_U, C)
        #   3. For each unlabeled point j with observed ỹ_j = k:
        #        numerator_c = T[c, k] * P(z=c | x_j; θ)
        #        γ_jc = numerator_c / sum_c(numerator_c)
        #
        #   --- M-step ---
        #   4. Update T[c, k]:
        #        sum_jc_k = sum over j where ỹ_j==k of γ_jc
        #        T[c, k] = (alpha - 1 + sum_jc_k) /
        #                  sum_{k'} (alpha - 1 + sum_jc_k')
        #   5. Re-fit logistic regression using sample weights:
        #        Gold points   → weight 1.0 (γ known exactly from one-hot)
        #        Unlabeled pts → weight γ_jc for class c (soft labels)
        #      HINT: Use sklearn's fit(X, y, sample_weight=w) where y is the
        #            hard assignment argmax(γ) and w is the max γ probability.
        #
        #   --- Convergence ---
        #   6. Compute ||T_new - T_old||_F.  Break if < tol.
        #
        # Store T_hat_ and self._clf.
        raise NotImplementedError("LatentNoiseEM.fit not yet implemented")
        self._is_fitted = True
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        X_s = self._scaler.transform(X)
        return self._clf.predict_proba(X_s)

    def get_confusion_matrix(self) -> np.ndarray:
        """Return the learned C×C noise transition matrix T̂."""
        if self.T_hat_ is None:
            raise RuntimeError("Call fit() first.")
        return self.T_hat_
