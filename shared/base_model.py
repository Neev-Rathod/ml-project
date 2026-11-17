"""
shared/base_model.py
====================
Abstract base class that every semi-supervised classifier in WP2–WP5 must implement.
This ensures all models share the same train/predict/evaluate API so WP5's evaluation
suite can call them interchangeably.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseSSLModel(ABC):
    """
    Interface for semi-supervised / noise-robust classifiers.

    Subclass this in each Work Package and fill in the three abstract methods.

    Parameters
    ----------
    name : str
        Human-readable model name used in log messages and plot titles.
    """

    def __init__(self, name: str = "BaseSSLModel"):
        self.name = name
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Abstract methods every subclass must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_L: np.ndarray,
        y_L: np.ndarray,
        X_U: np.ndarray,
        y_U_noisy: np.ndarray | None = None,
    ) -> "BaseSSLModel":
        """
        Train the model.

        Parameters
        ----------
        X_L : ndarray of shape (n_L, d)
            Feature matrix for the Gold Set (clean labels).
        y_L : ndarray of shape (n_L,)
            Clean integer labels for X_L  (0 … C-1).
        X_U : ndarray of shape (n_U, d)
            Feature matrix for the Untrusted Pool.
        y_U_noisy : ndarray of shape (n_U,) or None
            Noisy / machine-generated labels for X_U.
            Pass None if the method discards untrusted labels entirely.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return hard class predictions.

        Parameters
        ----------
        X : ndarray of shape (n, d)

        Returns
        -------
        y_pred : ndarray of shape (n,)  with integer class indices 0 … C-1
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability estimates.

        Parameters
        ----------
        X : ndarray of shape (n, d)

        Returns
        -------
        proba : ndarray of shape (n, C)  — rows sum to 1.
        """

    # ------------------------------------------------------------------
    # Concrete helpers available to every subclass
    # ------------------------------------------------------------------

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Compute standard evaluation metrics on a held-out test set.

        Returns a dict with keys:
            macro_f1, per_class_f1, roc_auc, pr_auc, confusion_matrix
        """
        from shared.metrics import evaluate_model
        return evaluate_model(self, X_test, y_test)

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"
