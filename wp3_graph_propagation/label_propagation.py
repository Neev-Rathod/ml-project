"""
wp3_graph_propagation/label_propagation.py
==========================================
Implements Graph Label Propagation (Proposal §4.3).

Algorithm steps
---------------
1. Build a kNN affinity graph with locally-scaled Gaussian kernel weights.
2. Solve the harmonic label spreading system (Eq. 4.2 from the proposal).
3. Return soft label assignments for all unlabeled nodes.

TODO for Neev (WP3)
-------------------
Implement the three classes/functions marked with TODO.
"""

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import NearestNeighbors

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.base_model import BaseSSLModel


# ── Graph construction ─────────────────────────────────────────────────────

def build_knn_graph(
    X: np.ndarray,
    k: int = 10,
    sigma_percentile: float = 50.0,
) -> csr_matrix:
    """
    Build a symmetric kNN affinity matrix with locally-scaled Gaussian kernel.

    W_ij = exp(-d(x_i, x_j)^2 / (sigma_i * sigma_j))

    where sigma_i = distance to the k-th nearest neighbor of x_i.

    Parameters
    ----------
    X               : ndarray (N, d)
    k               : number of nearest neighbors
    sigma_percentile: percentile of kNN distances used as sigma (default = 50)

    Returns
    -------
    W : sparse symmetric affinity matrix (N, N)
    """
    # TODO: Implement.
    # Steps:
    #   1. Use NearestNeighbors to find k+1 nearest neighbors (the first is self).
    #   2. Compute sigma_i = distance to the k-th neighbor.
    #   3. Build W_ij = exp(-d^2 / (sigma_i * sigma_j)) for each (i,j) edge.
    #   4. Symmetrize: W = (W + W.T) / 2.
    #   5. Return as a csr_matrix.
    raise NotImplementedError("build_knn_graph not yet implemented")


# ── Harmonic label spreading ────────────────────────────────────────────────

def harmonic_label_propagation(
    W: csr_matrix,
    labeled_indices: np.ndarray,
    y_L: np.ndarray,
    n_classes: int = 5,
    reg: float = 1e-6,
) -> np.ndarray:
    """
    Solve for the harmonic label assignment on unlabeled nodes.

    Proposal Eq. 4.2:
        F*_U = (D_UU - W_UU)^{-1} W_UL Y_L

    where D is the degree matrix.

    Parameters
    ----------
    W               : affinity matrix (N, N) sparse
    labeled_indices : integer indices of labeled nodes (n_L,)
    y_L             : integer labels for labeled nodes (n_L,)
    n_classes       : C
    reg             : diagonal regularisation added to (D_UU - W_UU)

    Returns
    -------
    F : ndarray (N, n_classes) — soft label assignment for ALL nodes
        (labeled nodes have one-hot rows; unlabeled nodes have propagated posteriors)
    """
    # TODO: Implement.
    # Steps:
    #   1. Build one-hot Y_L of shape (n_L, C).
    #   2. Identify unlabeled indices as the complement of labeled_indices.
    #   3. Extract W_UU (unlabeled-unlabeled block) and W_UL (unlabeled-labeled block).
    #   4. Compute D_UU = diag(rowsum of W restricted to unlabeled rows).
    #   5. Solve the linear system: (D_UU - W_UU + reg*I) @ F_U = W_UL @ Y_L
    #   6. Assemble full F matrix.
    raise NotImplementedError("harmonic_label_propagation not yet implemented")


# ── Full model class ────────────────────────────────────────────────────────

class GraphLabelPropagator(BaseSSLModel):
    """
    Semi-supervised classifier using harmonic label propagation.

    fit() builds the kNN graph and solves the harmonic system.
    predict() / predict_proba() use the propagated soft labels for unlabeled
    points and a kNN lookup for genuinely new test points.

    Parameters
    ----------
    k           : number of nearest neighbors for graph construction
    n_pca       : number of PCA components to apply before building the graph
                  (0 = no PCA)
    distance    : 'euclidean' | 'cosine' | 'mahalanobis'
    """

    def __init__(self, k: int = 10, n_pca: int = 20,
                 distance: str = "euclidean"):
        super().__init__(name=f"GraphLP(k={k},pca={n_pca},{distance})")
        self.k = k
        self.n_pca = n_pca
        self.distance = distance
        self._F = None          # soft label matrix for training nodes
        self._X_train = None    # kept for test-time kNN lookup

    def fit(self, X_L, y_L, X_U, y_U_noisy=None):
        """
        Discard y_U_noisy entirely; propagate clean labels from X_L into X_U.
        """
        # TODO: Implement.
        # 1. Stack X_L and X_U; labeled_indices = np.arange(len(X_L)).
        # 2. Optionally apply PCA + distance transformation.
        # 3. Build the kNN graph with build_knn_graph().
        # 4. Run harmonic_label_propagation().
        # 5. Store F and X_train for predict().
        raise NotImplementedError
        self._is_fitted = True
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        """
        For each test point, return the label distribution of its nearest
        training neighbor (transductive approximation).
        """
        # TODO: Use NearestNeighbors to find the nearest training point,
        #       then return self._F[neighbor_index].
        raise NotImplementedError
