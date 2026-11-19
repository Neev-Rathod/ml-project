"""
wp2_synthetic_baselines/run_wp2.py
====================================
Entry point for Work Package 2.

What this script does
---------------------
1. Generates the synthetic dataset (or loads PTB-XL features from WP1).
2. Injects all three noise types at multiple ε values.
3. Trains all five supervised classifiers in all three modes.
4. Plots the Supervised Breakdown Analysis (Proposal §5.2):
   - Macro F1 vs. noise rate ε for each model/mode.
   - Support vector count inflation for the SVM.
   - Decision tree depth / leaf count vs. ε (unpruned vs. pruned).

Usage
-----
    # Synthetic dataset:
    python run_wp2.py --mode synthetic --out_dir ../outputs/wp2

    # PTB-XL features (requires WP1 to have run first):
    python run_wp2.py --mode ptbxl \
        --features_path ../outputs/wp1/features.npz \
        --labels_path   ../outputs/wp1/labels.npz   \
        --out_dir       ../outputs/wp2

Arguments
---------
--mode           'synthetic' or 'ptbxl'
--features_path  (ptbxl mode) path to WP1 features.npz
--labels_path    (ptbxl mode) path to WP1 labels.npz
--out_dir        Output directory for plots and CSVs
--epsilon_values Comma-separated noise rates, e.g. "0,0.15,0.30,0.45,0.60"
--seed           Random seed (default 42)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wp2_synthetic_baselines.synthetic_generator import (
    generate_manifold,
    inject_uniform_noise,
    inject_asymmetric_noise,
    inject_boundary_noise,
)
from wp2_synthetic_baselines.baselines import (
    GaussianNaiveBayes, LogisticRegressionModel,
    KNNClassifierModel, DecisionTreeModel, SVMModel,
)
from shared.data_utils import load_standard_split
from shared.metrics import evaluate_model, print_results


EPSILON_DEFAULT = [0.0, 0.15, 0.30, 0.45, 0.60]
MODES_TRAIN     = ["gold_only", "naive_pool", "self_training"]


def parse_args():
    p = argparse.ArgumentParser(description="WP2 – Synthetic Generator & Supervised Baselines")
    p.add_argument("--mode",          default="synthetic", choices=["synthetic", "ptbxl"])
    p.add_argument("--features_path", default=None)
    p.add_argument("--labels_path",   default=None)
    p.add_argument("--out_dir",       default=str(Path("..") / "outputs" / "wp2"))
    p.add_argument("--epsilon_values",default="0,0.15,0.30,0.45,0.60")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


# ── Classifier constructors ──────────────────────────────────────────────

CLF_CLASSES = {
    "GNB": GaussianNaiveBayes,
    "LR":  LogisticRegressionModel,
    "kNN": KNNClassifierModel,
    "DT":  lambda mode: DecisionTreeModel(mode=mode, max_depth=None),
    "SVM": SVMModel,
}


def _make_clf(name: str, mode: str):
    """Construct a fresh classifier instance by name and training mode."""
    if name == "DT":
        return DecisionTreeModel(mode=mode, max_depth=None)
    cls = CLF_CLASSES[name]
    return cls(mode=mode)


# ── Main analysis function ───────────────────────────────────────────────

def run_breakdown_analysis(
    X_L, y_L,
    X_U, y_U_clean,   # clean labels for injecting noise
    X_val, y_val,
    epsilon_values: list[float],
    out_dir: Path,
    seed: int = 42,
):
    """
    For each epsilon value, inject uniform noise into y_U_clean, train each
    model in each mode, and record macro F1 on the validation set.
    Save results CSV, F1-vs-epsilon plot, SVM support vector plot, and
    decision tree depth plot.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clf_names = ["GNB", "LR", "kNN", "DT", "SVM"]
    rows = []

    # Also track SVM support vector counts and DT depth/leaves
    svm_sv_records = []
    dt_records = []

    total_runs = len(epsilon_values) * len(clf_names) * len(MODES_TRAIN)
    run_count = 0

    for eps in epsilon_values:
        # Inject uniform noise at this epsilon level
        y_U_noisy = inject_uniform_noise(y_U_clean, eps, seed=seed)

        for clf_name in clf_names:
            for mode in MODES_TRAIN:
                run_count += 1
                label = f"{clf_name}-{mode}"
                print(f"  [{run_count}/{total_runs}] ε={eps:.2f}  {label} ...", end=" ", flush=True)

                clf = _make_clf(clf_name, mode)

                try:
                    clf.fit(X_L, y_L, X_U, y_U_noisy)
                    results = evaluate_model(clf, X_val, y_val)
                    macro_f1 = results["macro_f1"]
                    roc_auc = results["roc_auc"]
                    pr_auc = results["pr_auc"]
                except Exception as e:
                    print(f"FAILED ({e})")
                    macro_f1 = float("nan")
                    roc_auc = float("nan")
                    pr_auc = float("nan")
                    results = None

                rows.append({
                    "epsilon": eps,
                    "classifier": clf_name,
                    "mode": mode,
                    "label": label,
                    "macro_f1": macro_f1,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                })

                # Track SVM support vector counts
                if clf_name == "SVM" and hasattr(clf, "n_support_vectors_") and clf.n_support_vectors_ is not None:
                    svm_sv_records.append({
                        "epsilon": eps,
                        "mode": mode,
                        "n_support_vectors": clf.n_support_vectors_,
                    })

                # Track DT depth and leaves
                if clf_name == "DT" and hasattr(clf, "tree_depth_") and clf.tree_depth_ is not None:
                    dt_records.append({
                        "epsilon": eps,
                        "mode": mode,
                        "max_depth_param": "unpruned",
                        "tree_depth": clf.tree_depth_,
                        "n_leaves": clf.n_leaves_,
                    })

                print(f"F1={macro_f1:.4f}")

    # Also train pruned DT (max_depth=5) for comparison
    print("\n── Pruned Decision Tree (max_depth=5) analysis ──")
    for eps in epsilon_values:
        y_U_noisy = inject_uniform_noise(y_U_clean, eps, seed=seed)
        for mode in MODES_TRAIN:
            label = f"DT(d=5)-{mode}"
            print(f"  ε={eps:.2f}  {label} ...", end=" ", flush=True)
            clf = DecisionTreeModel(mode=mode, max_depth=5)
            try:
                clf.fit(X_L, y_L, X_U, y_U_noisy)
                dt_records.append({
                    "epsilon": eps,
                    "mode": mode,
                    "max_depth_param": "pruned(d=5)",
                    "tree_depth": clf.tree_depth_,
                    "n_leaves": clf.n_leaves_,
                })
                print(f"depth={clf.tree_depth_}, leaves={clf.n_leaves_}")
            except Exception as e:
                print(f"FAILED ({e})")

    # ── Save results CSV ──────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = out_dir / "noise_breakdown.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[WP2] Saved results to {csv_path}")

    # ── Plot 1: Macro F1 vs. epsilon ──────────────────────────────────────
    _plot_f1_vs_epsilon(df, out_dir)

    # ── Plot 2: SVM support vector count vs. epsilon ─────────────────────
    if svm_sv_records:
        _plot_svm_support_vectors(svm_sv_records, out_dir)

    # ── Plot 3: Decision tree depth vs. epsilon ───────────────────────────
    if dt_records:
        _plot_tree_depth(dt_records, out_dir)

    return df


# ── Plotting functions ───────────────────────────────────────────────────

def _plot_f1_vs_epsilon(df: pd.DataFrame, out_dir: Path):
    """Main comparison plot: Macro F1 vs ε for each model/mode combination."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define styles for each mode
    mode_styles = {
        "gold_only":     {"linestyle": "-",  "marker": "o"},
        "naive_pool":    {"linestyle": "--", "marker": "s"},
        "self_training": {"linestyle": ":",  "marker": "^"},
    }

    # Color palette for classifiers
    clf_colors = {
        "GNB": "#1f77b4",
        "LR":  "#ff7f0e",
        "kNN": "#2ca02c",
        "DT":  "#d62728",
        "SVM": "#9467bd",
    }

    for _, grp in df.groupby("label"):
        clf_name = grp["classifier"].iloc[0]
        mode = grp["mode"].iloc[0]
        style = mode_styles.get(mode, {})
        color = clf_colors.get(clf_name, None)
        ax.plot(
            grp["epsilon"], grp["macro_f1"],
            label=grp["label"].iloc[0],
            color=color,
            linewidth=2,
            markersize=6,
            **style,
        )

    ax.set_xlabel("Noise Rate ε (uniform)", fontsize=13)
    ax.set_ylabel("Macro F1 (validation)", fontsize=13)
    ax.set_title("WP2 — Supervised Breakdown: Macro F1 vs. Noise Rate ε", fontsize=14, fontweight="bold")
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=8, ncol=1, frameon=True,
    )
    ax.set_ylim(bottom=0)
    ax.set_xticks(df["epsilon"].unique())

    fig.tight_layout()
    path = out_dir / "f1_vs_epsilon.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[WP2] Saved {path}")


def _plot_svm_support_vectors(records: list[dict], out_dir: Path):
    """SVM support vector count vs. ε (shows margin corruption)."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    df_sv = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9, 6))

    mode_markers = {"gold_only": "o", "naive_pool": "s", "self_training": "^"}
    mode_styles = {"gold_only": "-", "naive_pool": "--", "self_training": ":"}

    for mode, grp in df_sv.groupby("mode"):
        grp_sorted = grp.sort_values("epsilon")
        ax.plot(
            grp_sorted["epsilon"],
            grp_sorted["n_support_vectors"],
            label=f"SVM-{mode}",
            marker=mode_markers.get(mode, "o"),
            linestyle=mode_styles.get(mode, "-"),
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Noise Rate ε", fontsize=13)
    ax.set_ylabel("Number of Support Vectors", fontsize=13)
    ax.set_title("WP2 — SVM Support Vector Inflation vs. Noise Rate", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(df_sv["epsilon"].unique())

    fig.tight_layout()
    path = out_dir / "svm_support_vectors.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[WP2] Saved {path}")


def _plot_tree_depth(records: list[dict], out_dir: Path):
    """Decision tree depth and leaf count vs. ε for unpruned vs. pruned."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    df_dt = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left subplot: Tree Depth ---
    ax = axes[0]
    for (variant, mode), grp in df_dt.groupby(["max_depth_param", "mode"]):
        if mode != "naive_pool":
            continue  # Focus on naive_pool for clarity (most affected by noise)
        grp_sorted = grp.sort_values("epsilon")
        marker = "o" if "unpruned" in str(variant) else "s"
        linestyle = "-" if "unpruned" in str(variant) else "--"
        ax.plot(
            grp_sorted["epsilon"],
            grp_sorted["tree_depth"],
            label=f"{variant} ({mode})",
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=7,
        )
    # Also show gold_only unpruned as reference
    for (variant, mode), grp in df_dt.groupby(["max_depth_param", "mode"]):
        if mode != "gold_only" or variant != "unpruned":
            continue
        grp_sorted = grp.sort_values("epsilon")
        ax.plot(
            grp_sorted["epsilon"],
            grp_sorted["tree_depth"],
            label=f"{variant} ({mode})",
            marker="^",
            linestyle="-.",
            linewidth=2,
            markersize=7,
            color="gray",
        )
    ax.set_xlabel("Noise Rate ε", fontsize=12)
    ax.set_ylabel("Tree Depth", fontsize=12)
    ax.set_title("Tree Depth vs. ε", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    # --- Right subplot: Leaf Count ---
    ax = axes[1]
    for (variant, mode), grp in df_dt.groupby(["max_depth_param", "mode"]):
        if mode != "naive_pool":
            continue
        grp_sorted = grp.sort_values("epsilon")
        marker = "o" if "unpruned" in str(variant) else "s"
        linestyle = "-" if "unpruned" in str(variant) else "--"
        ax.plot(
            grp_sorted["epsilon"],
            grp_sorted["n_leaves"],
            label=f"{variant} ({mode})",
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=7,
        )
    for (variant, mode), grp in df_dt.groupby(["max_depth_param", "mode"]):
        if mode != "gold_only" or variant != "unpruned":
            continue
        grp_sorted = grp.sort_values("epsilon")
        ax.plot(
            grp_sorted["epsilon"],
            grp_sorted["n_leaves"],
            label=f"{variant} ({mode})",
            marker="^",
            linestyle="-.",
            linewidth=2,
            markersize=7,
            color="gray",
        )
    ax.set_xlabel("Noise Rate ε", fontsize=12)
    ax.set_ylabel("Number of Leaves", fontsize=12)
    ax.set_title("Leaf Count vs. ε", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("WP2 — Decision Tree Overfitting Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = out_dir / "tree_depth_vs_epsilon.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[WP2] Saved {path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epsilon_values = [float(e) for e in args.epsilon_values.split(",")]

    if args.mode == "synthetic":
        X, y = generate_manifold(n_samples=3000, seed=args.seed)
        # Simple 70/15/15 train/val/test split
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(y))
        n_train = int(0.70 * len(y))
        n_val   = int(0.15 * len(y))
        idx_tr, idx_val, idx_te = (idx[:n_train],
                                   idx[n_train:n_train + n_val],
                                   idx[n_train + n_val:])
        # For synthetic, 20% of training becomes "gold"
        n_gold = int(0.20 * n_train)
        X_L, y_L     = X[idx_tr[:n_gold]], y[idx_tr[:n_gold]]
        X_U, y_U_clean = X[idx_tr[n_gold:]], y[idx_tr[n_gold:]]
        X_val, y_val = X[idx_val], y[idx_val]

    else:   # ptbxl
        if not args.features_path or not args.labels_path:
            print("ERROR: --features_path and --labels_path required for ptbxl mode.")
            sys.exit(1)
        split = load_standard_split(
            args.features_path, args.labels_path, None
        )
        X_L, y_L     = split["X_train_L"], split["y_train_L"]
        X_U, y_U_clean = split["X_train_U"], split["y_train_U"]
        X_val, y_val = split["X_val"], split["y_val"]

    print(f"[WP2] Mode: {args.mode}")
    print(f"[WP2] Gold set:       {X_L.shape[0]} samples")
    print(f"[WP2] Untrusted pool: {X_U.shape[0]} samples")
    print(f"[WP2] Validation set: {X_val.shape[0]} samples")
    print(f"[WP2] Epsilon values: {epsilon_values}")
    print()

    run_breakdown_analysis(
        X_L, y_L,
        X_U, y_U_clean,
        X_val, y_val,
        epsilon_values,
        out_dir,
        seed=args.seed,
    )

    print(f"\n[WP2] Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
