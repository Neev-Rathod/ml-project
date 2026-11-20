"""
wp3_graph_propagation/label_propagation.py
==========================================
Graph Label Propagation (Proposal 4.3).

1. Build a kNN affinity graph with locally-scaled Gaussian kernel weights.
2. Solve the harmonic label spreading system (Eq. 4.2).
3. Predict for new points via a kernel-weighted kNN lookup into the graph.
Untrusted labels (y_U_noisy) are never used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.base_model import BaseSSLModel


# ── Graph construction ─────────────────────────────────────────────────────

def _knn_affinity(Z, k=10, local_scale_k=7, metric="euclidean"):
    """Returns (W, sigma, fitted NearestNeighbors, k_eff, local_scale_eff, m)."""
    n = len(Z)
    k_eff = min(k, n - 1)
    ls = min(local_scale_k, n - 1)
    m = max(k_eff, ls)
    nn = NearestNeighbors(
        n_neighbors=m + 1, metric=metric,
        algorithm="brute" if metric == "cosine" else "auto",
    ).fit(Z)
    dist, idx = nn.kneighbors(Z)
    dist, idx = dist[:, 1:], idx[:, 1:]                       # drop self
    sigma = np.maximum(dist[:, ls - 1], 1e-8)                 # local scale per node

    rows = np.repeat(np.arange(n), k_eff)
    cols = idx[:, :k_eff].ravel()
    d = dist[:, :k_eff].ravel()
    w = np.exp(-(d ** 2) / (sigma[rows] * sigma[cols]))
    W = csr_matrix((w, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T).tocsr()                                # symmetrise
    return W, sigma, nn, k_eff, ls, m


def build_knn_graph(
    X: np.ndarray,
    k: int = 10,
    sigma_percentile: float = 50.0,   # kept for template compatibility; unused
    local_scale_k: int = 7,
    metric: str = "euclidean",
) -> csr_matrix:
    """
    Symmetric kNN affinity matrix, W_ij = exp(-d_ij^2 / (sigma_i sigma_j)),
    sigma_i = distance from x_i to its `local_scale_k`-th nearest neighbour.
    """
    return _knn_affinity(X, k, local_scale_k, metric)[0]


# ── Harmonic label spreading ────────────────────────────────────────────────

def harmonic_label_propagation(
    W: csr_matrix,
    labeled_indices: np.ndarray,
    y_L: np.ndarray,
    n_classes: int = 5,
    reg: float = 1e-6,
    class_mass_norm: bool = False,
) -> np.ndarray:
    """
    Eq. 4.2:  F*_U = (D_UU - W_UU + reg*I)^-1 W_UL Y_L

    Returns F (N, n_classes): one-hot rows for labeled nodes, propagated
    posteriors for unlabeled nodes (rows sum to 1). Unlabeled nodes that no
    labeled node can reach get the (smoothed) class prior.
    """
    W = csr_matrix(W)
    N = W.shape[0]
    labeled_indices = np.asarray(labeled_indices)
    y_L = np.asarray(y_L).astype(int)
    unl = np.setdiff1d(np.arange(N), labeled_indices)

    Y_L = np.eye(n_classes)[y_L]
    prior = (Y_L.sum(0) + 1.0) / (len(y_L) + n_classes)
    F = np.zeros((N, n_classes))
    F[labeled_indices] = Y_L

    if len(unl):
        deg = np.asarray(W.sum(1)).ravel()
        W_UU = W[unl][:, unl]
        W_UL = W[unl][:, labeled_indices]
        A = (diags(deg[unl]) - W_UU + reg * identity(len(unl))).tocsc()
        F_U = np.clip(splu(A).solve(np.asarray(W_UL @ Y_L)), 0.0, None)

        if class_mass_norm:                                   # Zhu et al. 2003
            F_U = F_U * (prior * F_U.sum() / (F_U.sum(0) + 1e-12))

        s = F_U.sum(1, keepdims=True)
        bad = s[:, 0] < 1e-12
        F_U = F_U / np.where(s < 1e-12, 1.0, s)
        F_U[bad] = prior
        F[unl] = F_U
    return F


# ── Full model class ────────────────────────────────────────────────────────

class GraphLabelPropagator(BaseSSLModel):
    """
    Parameters
    ----------
    k           : neighbours per node in the graph
    n_pca       : PCA components before building the graph (0/None = no PCA)
    distance    : 'euclidean' | 'cosine' | 'mahalanobis'
    reg_lambda  : lambda in Eq. 4.1 (relative to mean variance), Mahalanobis only
    local_scale_k : sigma_i = distance to this-th neighbour
    clip        : clip standardised features to +-clip (None = off)
    class_mass_norm : apply class mass normalisation after propagation
    """

    VALID_DISTANCES = ("euclidean", "cosine", "mahalanobis")

    def __init__(self, k: int = 10, n_pca: int = 20, distance: str = "euclidean",
                 reg_lambda: float = 1e-2, local_scale_k: int = 7,
                 clip: float | None = 5.0, class_mass_norm: bool = False,
                 n_classes: int = 5, ridge: float = 1e-6, name: str | None = None):
        if distance not in self.VALID_DISTANCES:
            raise ValueError(f"distance must be one of {self.VALID_DISTANCES}")
        super().__init__(name=name or f"GraphLP(k={k},pca={n_pca},{distance})")
        self.k, self.n_pca, self.distance = k, n_pca, distance
        self.reg_lambda, self.local_scale_k = reg_lambda, local_scale_k
        self.clip, self.class_mass_norm = clip, class_mass_norm
        self.n_classes, self.ridge = n_classes, ridge
        self._F = None
        self._X_train = None

    # -- feature pipeline (scale -> clip -> PCA -> optional whitening) -------
    def _project(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=np.float64))
        if self.clip:
            Xs = np.clip(Xs, -self.clip, self.clip)
        if self.pca_ is not None:
            Xs = self.pca_.transform(Xs)
        return Xs

    def _embed(self, X):
        Z = self._project(X)
        if self.chol_ is not None:       # Mahalanobis == Euclidean after whitening
            Z = solve_triangular(self.chol_, Z.T, lower=True).T
        return Z

    def fit(self, X_L, y_L, X_U, y_U_noisy=None):
        """Discard y_U_noisy; propagate clean labels from X_L into X_U."""
        X_L = np.asarray(X_L, dtype=np.float64)
        X_U = np.asarray(X_U, dtype=np.float64)
        y_L = np.asarray(y_L).astype(int)
        n_L, n = len(X_L), len(X_L) + len(X_U)
        X_all = np.vstack([X_L, X_U])
        labeled_indices = np.arange(n_L)

        self.scaler_ = StandardScaler().fit(X_all)    # features only -> no label leakage
        self.pca_, self.chol_ = None, None
        if self.n_pca:
            pre = self._project(X_all)
            self.pca_ = PCA(n_components=min(self.n_pca, pre.shape[1], n - 1),
                            random_state=0).fit(pre)
        Z = self._project(X_all)

        if self.distance == "mahalanobis":            # Eq. 4.1: (Sigma_L + lambda I)^-1
            cov = np.atleast_2d(np.cov(Z[:n_L].T))
            lam = self.reg_lambda * np.trace(cov) / cov.shape[0]
            self.chol_ = cholesky(cov + lam * np.eye(cov.shape[0]), lower=True)
            Z = solve_triangular(self.chol_, Z.T, lower=True).T

        metric = "cosine" if self.distance == "cosine" else "euclidean"
        W, self.sigma_, self.nn_, self.k_nn_, self.ls_, self.m_ = _knn_affinity(
            Z, self.k, self.local_scale_k, metric)

        # connectivity diagnostics for the graph-connectivity analysis
        n_comp, comp = connected_components(W, directed=False)
        self.n_graph_components_ = int(n_comp)
        self.n_unreachable_ = int(
            (~np.isin(comp[n_L:], np.unique(comp[:n_L]))).sum())

        self._F = harmonic_label_propagation(
            W, labeled_indices, y_L, self.n_classes, self.ridge, self.class_mass_norm)
        self.prior_ = (np.eye(self.n_classes)[y_L].sum(0) + 1.0) / (n_L + self.n_classes)
        self._X_train = Z
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        """Kernel-weighted average of the F rows of the k nearest graph nodes
        (same local scaling as the graph) -- out-of-sample extension."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")
        Zq = self._embed(X)
        dist, idx = self.nn_.kneighbors(Zq, n_neighbors=self.m_)
        sigma_q = np.maximum(dist[:, self.ls_ - 1], 1e-8)
        d, j = dist[:, : self.k_nn_], idx[:, : self.k_nn_]
        w = np.exp(-(d ** 2) / (sigma_q[:, None] * self.sigma_[j]))
        w[w.sum(1) < 1e-12] = 1.0                     # underflow -> uniform weights
        P = np.einsum("nk,nkc->nc", w, self._F[j]) / w.sum(1, keepdims=True)
        return P / P.sum(1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)