"""
wp1_data_pipeline/noise_analysis.py
=====================================
Implements the Sensor Noise & Subspace Analysis from Proposal §5.1.

Tasks
-----
1. Split records by hardware artifact flags (baseline_drift, static_noise).
2. Measure cluster purity of k-Means in raw feature space using three
   distance metrics: Euclidean, Cosine, Mahalanobis.
3. Repeat after PCA dimensionality reduction.
4. Produce comparison bar-chart and save as PNG.

Usage
-----
Run via run_wp1.py, or directly:

    python -m wp1_data_pipeline.noise_analysis \
        --features_path outputs/wp1/features.npz \
        --labels_path   outputs/wp1/labels.npz   \
        --meta_path     dataset/ptbxl_database.csv \
        --out_dir       outputs/wp1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import cdist


# ── Cluster purity helper ─────────────────────────────────────────────────

def cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Cluster purity = sum_k(max_c count(cluster_k ∩ class_c)) / N
    """
    n = len(labels_true)
    purity = 0.0
    for k in np.unique(labels_pred):
        mask = labels_pred == k
        if mask.sum() == 0:
            continue
        counts = np.bincount(labels_true[mask])
        purity += counts.max()
    return purity / n


# ── Mahalanobis-based KMeans approximation ────────────────────────────────

def mahalanobis_transform(X: np.ndarray, reg: float = 1e-3) -> np.ndarray:
    """
    Return X transformed by the inverse square root of the covariance matrix,
    so that standard Euclidean distance on the output equals Mahalanobis distance
    on the input.
    """
    cov = np.cov(X.T)
    cov += reg * np.eye(cov.shape[0])
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-10)
    cov_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return X @ cov_inv_sqrt


def cosine_transform(X: np.ndarray) -> np.ndarray:
    """L2-normalise rows so Euclidean distance equals cosine distance × sqrt(2)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-10)


# ── Main analysis function ────────────────────────────────────────────────

def run_noise_analysis(
    X: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    meta: pd.DataFrame,
    out_dir: Path,
    n_pca_components: int = 20,
    n_clusters: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run the full sensor noise / subspace analysis.

    Parameters
    ----------
    X        : feature matrix (N, 84)
    y        : integer labels (N,)   [-1 excluded before call]
    fold     : fold assignments (N,)
    meta     : full ptbxl_database.csv DataFrame (index = ecg_id)
    out_dir  : directory for saving plots
    n_pca_components : PCA dimensionality
    n_clusters       : number of k-Means clusters (= number of classes = 5)
    seed             : random seed for reproducibility

    Returns
    -------
    results_df : DataFrame with purity and ARI for each condition
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Align meta flags with our feature array -------------------------
    # meta index is ecg_id; we need to align it with the row order of X/y/fold
    # WP1 saves the ecg_ids alongside features; here we assume they are aligned.
    # We use the strat_fold column to separate training records for the analysis.
    train_mask = np.isin(fold, list(range(1, 9))) & (y != -1)
    X_tr = X[train_mask]
    y_tr = y[train_mask]

    # Build artifact masks aligned to training rows
    # meta must be sorted by ecg_id (its index) in the same order as the
    # feature array. We pass a boolean mask built from meta's columns.
    # (In practice, loader.py and run_wp1.py keep the same row order.)
    train_meta = meta[meta["strat_fold"].isin(range(1, 9))].copy()
    # Restrict to records with a resolved label
    train_meta = train_meta[train_meta["diagnostic_superclass"].notna()]

    has_baseline = train_meta["baseline_drift"].fillna(False).astype(bool).values
    has_static   = train_meta["static_noise"].fillna(False).astype(bool).values
    has_any_artifact = has_baseline | has_static

    # Pad / trim artifact masks to match X_tr length
    min_len = min(len(X_tr), len(has_any_artifact))
    X_tr = X_tr[:min_len]
    y_tr = y_tr[:min_len]
    has_any_artifact = has_any_artifact[:min_len]

    # --- Scaling ---------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_tr)

    # --- PCA -------------------------------------------------------------
    pca = PCA(n_components=n_pca_components, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)

    # --- Variants to test ------------------------------------------------
    variants = {
        "Raw+Euclidean":    X_scaled,
        "Raw+Cosine":       cosine_transform(X_scaled),
        "Raw+Mahalanobis":  mahalanobis_transform(X_scaled),
        "PCA+Euclidean":    X_pca,
        "PCA+Cosine":       cosine_transform(X_pca),
        "PCA+Mahalanobis":  mahalanobis_transform(X_pca),
    }

    rows = []
    for name, Xv in variants.items():
        # All records
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        pred_all = km.fit_predict(Xv)
        purity_all = cluster_purity(y_tr, pred_all)
        ari_all    = adjusted_rand_score(y_tr, pred_all)

        # Clean records only (no artifact)
        idx_clean = ~has_any_artifact
        if idx_clean.sum() > n_clusters:
            pred_clean = km.predict(Xv[idx_clean])
            purity_clean = cluster_purity(y_tr[idx_clean], pred_clean)
            ari_clean    = adjusted_rand_score(y_tr[idx_clean], pred_clean)
        else:
            purity_clean = ari_clean = float("nan")

        # Noisy records only
        idx_noisy = has_any_artifact
        if idx_noisy.sum() > n_clusters:
            pred_noisy = km.predict(Xv[idx_noisy])
            purity_noisy = cluster_purity(y_tr[idx_noisy], pred_noisy)
            ari_noisy    = adjusted_rand_score(y_tr[idx_noisy], pred_noisy)
        else:
            purity_noisy = ari_noisy = float("nan")

        rows.append({
            "variant":       name,
            "purity_all":    round(purity_all, 4),
            "ari_all":       round(ari_all, 4),
            "purity_clean":  round(purity_clean, 4) if not np.isnan(purity_clean) else None,
            "purity_noisy":  round(purity_noisy, 4) if not np.isnan(purity_noisy) else None,
        })

    results_df = pd.DataFrame(rows)

    # --- Plot ------------------------------------------------------------
    _plot_purity_comparison(results_df, out_dir / "noise_cluster_purity.png")
    _plot_pca_variance(pca, out_dir / "pca_explained_variance.png")

    results_df.to_csv(out_dir / "noise_analysis_results.csv", index=False)
    print(f"[WP1] Noise analysis results saved to {out_dir}")
    print(results_df.to_string(index=False))
    return results_df


def _plot_purity_comparison(df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    w = 0.3
    ax.bar(x - w, df["purity_all"],   width=w, label="All records")
    ax.bar(x,     df["purity_clean"], width=w, label="Clean (no artifact)")
    ax.bar(x + w, df["purity_noisy"], width=w, label="Noisy (has artifact)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"], rotation=25, ha="right")
    ax.set_ylabel("Cluster Purity")
    ax.set_title("k-Means Cluster Purity: Raw vs. PCA, by Distance Metric")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[WP1] Saved {save_path}")


def _plot_pca_variance(pca: PCA, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.plot(range(1, len(cumvar) + 1), cumvar, marker="o")
    ax.axhline(90, color="red", linestyle="--", label="90% threshold")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("PCA Scree Plot – PTB-XL 84-dim Features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[WP1] Saved {save_path}")


# ── CLI entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", required=True)
    parser.add_argument("--labels_path",   required=True)
    parser.add_argument("--meta_path",     required=True)
    parser.add_argument("--out_dir",       default="outputs/wp1")
    args = parser.parse_args()

    feat   = np.load(args.features_path)
    labels = np.load(args.labels_path)
    meta   = pd.read_csv(args.meta_path, index_col="ecg_id")
    import ast
    meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)

    run_noise_analysis(
        X    = feat["X"],
        y    = labels["y"],
        fold = labels["fold"],
        meta = meta,
        out_dir = args.out_dir,
    )
