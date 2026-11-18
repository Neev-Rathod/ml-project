"""
wp4_cotraining_tsvm/run_wp4.py
================================
Entry point for Work Package 4: Co-Training & Transductive SVMs.

Experiments
-----------
1. Sweep η (gold fraction) for Multi-View Co-Training:
   - Compares Co-Training against supervised baselines (Full, View 1, View 2).
   - Tracks inter-view agreement rate, macro F1, per-class F1, ROC-AUC, and PR-AUC.
2. Sweep η for Transductive SVM (TSVM):
   - Compares TSVM against supervised SVM.
   - Evaluates margin stability, support vector counts, and class imbalance effects.
3. Track and visualize inter-view agreement dynamics across co-training rounds.
4. Analyze margin stability and support vector inflation under severe class imbalance.

Usage
-----
    python run_wp4.py \\
        --features_path ../outputs/wp1/features.npz \\
        --labels_path   ../outputs/wp1/labels.npz   \\
        --out_dir       ../outputs/wp4

Arguments
---------
--features_path  Path to WP1 features.npz
--labels_path    Path to WP1 labels.npz
--out_dir        Output directory (default: ../outputs/wp4)
--eta_values     Gold-label fractions, e.g. "0.02,0.05,0.10,0.20,0.50"
--seed           Random seed (default: 42)
--max_records    Optional cap on unlabeled records for faster runs
--eval_test      Also evaluate on test set (fold 10) in addition to validation
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.data_utils import SUPERCLASSES, load_standard_split
from shared.metrics import evaluate_model, print_results
from wp4_cotraining_tsvm.cotraining import VIEW1, VIEW2, CoTrainer
from wp4_cotraining_tsvm.tsvm import TSVMClassifier


def parse_args():
    p = argparse.ArgumentParser(description="WP4 – Co-Training & Transductive SVMs")
    p.add_argument("--features_path", required=True, help="Path to features.npz")
    p.add_argument("--labels_path",   required=True, help="Path to labels.npz")
    p.add_argument("--out_dir",       default=str(Path("..") / "outputs" / "wp4"))
    p.add_argument("--eta_values",    default="0.02,0.05,0.10,0.20,0.50")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--max_records",   type=int, default=None, help="Cap unlabeled pool size")
    p.add_argument("--eval_test",     action="store_true", help="Also evaluate on test fold")
    return p.parse_args()


def subsample_gold(X_L: np.ndarray, y_L: np.ndarray, eta: float, seed: int, stratify: bool = True):
    """
    Subsample a fraction eta of the Gold Set.
    Uses stratified sampling to maintain class proportions across small eta.
    """
    n = max(5, int(eta * len(X_L)))
    if n >= len(X_L):
        return X_L.copy(), y_L.copy()

    if stratify:
        try:
            from sklearn.model_selection import train_test_split
            idx, _ = train_test_split(
                np.arange(len(X_L)),
                train_size=n,
                stratify=y_L,
                random_state=seed,
            )
            return X_L[idx], y_L[idx]
        except Exception:
            pass

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_L), size=n, replace=False)
    return X_L[idx], y_L[idx]


def run_eta_sweep_cotraining(
    X_L_full: np.ndarray,
    y_L_full: np.ndarray,
    X_U: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    eta_values: list[float],
    out_dir: Path,
    seed: int = 42,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[float, list[dict]]]:
    """
    Sweep eta (gold fraction) for Co-Training vs Supervised Baselines.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: Co-Training η Sweep & View Agreement Analysis")
    print("=" * 60)

    rows = []
    agreement_curves = {}

    for eta in eta_values:
        t0 = time.time()
        X_L, y_L = subsample_gold(X_L_full, y_L_full, eta, seed=seed)
        n_gold = len(X_L)

        # 1. Supervised Baseline on Full Features (Gold-only)
        base_full = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)),
        ])
        base_full.fit(X_L, y_L)
        res_base_full = evaluate_model(base_full, X_val, y_val)

        # 2. Supervised Baseline on View 1 only
        base_v1 = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)),
        ])
        base_v1.fit(X_L[:, VIEW1], y_L)
        res_base_v1 = evaluate_model(base_v1, X_val[:, VIEW1], y_val)

        # 3. Supervised Baseline on View 2 only
        base_v2 = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)),
        ])
        base_v2.fit(X_L[:, VIEW2], y_L)
        res_base_v2 = evaluate_model(base_v2, X_val[:, VIEW2], y_val)

        # 4. Multi-View Co-Training
        cotrainer = CoTrainer(
            n_rounds=30,
            p=5,
            threshold=0.75,
            random_state=seed,
        )
        cotrainer.fit(X_L, y_L, X_U)
        res_cotrain = evaluate_model(cotrainer, X_val, y_val)
        final_agreement = cotrainer.compute_agreement(X_U)
        agreement_curves[eta] = cotrainer.history_

        row = {
            "eta": eta,
            "n_gold": n_gold,
            "n_unlabeled": len(X_U),
            "cotrain_macro_f1": res_cotrain["macro_f1"],
            "cotrain_roc_auc": res_cotrain["roc_auc"],
            "cotrain_pr_auc": res_cotrain["pr_auc"],
            "cotrain_final_agreement": round(final_agreement, 4),
            "cotrain_rounds_run": len(cotrainer.history_),
            "base_full_macro_f1": res_base_full["macro_f1"],
            "base_full_roc_auc": res_base_full["roc_auc"],
            "base_full_pr_auc": res_base_full["pr_auc"],
            "base_v1_macro_f1": res_base_v1["macro_f1"],
            "base_v2_macro_f1": res_base_v2["macro_f1"],
            "f1_gain_over_base": round(res_cotrain["macro_f1"] - res_base_full["macro_f1"], 4),
        }

        # Add per-class F1
        for cls in SUPERCLASSES:
            row[f"cotrain_f1_{cls}"] = res_cotrain["per_class_f1"].get(cls, 0.0)
            row[f"base_f1_{cls}"] = res_base_full["per_class_f1"].get(cls, 0.0)

        # Optional test set evaluation
        if X_test is not None and y_test is not None:
            res_test = evaluate_model(cotrainer, X_test, y_test)
            row["test_cotrain_macro_f1"] = res_test["macro_f1"]
            row["test_cotrain_roc_auc"] = res_test["roc_auc"]

        rows.append(row)
        elapsed = time.time() - t0
        print(
            f"  [Co-Training] η = {eta:4.2f} ({n_gold:4d} gold) | "
            f"Base F1: {res_base_full['macro_f1']:.4f} | "
            f"CoTrain F1: {res_cotrain['macro_f1']:.4f} (Δ {row['f1_gain_over_base']:+.4f}) | "
            f"Agreement: {final_agreement:.3f} | {elapsed:.1f}s"
        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "cotraining_eta_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[WP4] Saved Co-Training sweep results to: {csv_path}")

    # Plot 1: F1 vs eta for Co-Training
    plot_f1_vs_eta_cotraining(df, out_dir)

    # Plot 2: Agreement rate across rounds
    plot_view_agreement(agreement_curves, out_dir)

    return df, agreement_curves


def run_eta_sweep_tsvm(
    X_L_full: np.ndarray,
    y_L_full: np.ndarray,
    X_U: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    eta_values: list[float],
    out_dir: Path,
    seed: int = 42,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Sweep eta (gold fraction) for TSVM vs Supervised SVM.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: TSVM η Sweep & Margin Stability Analysis")
    print("=" * 60)

    rows = []
    sv_breakdowns = []

    for eta in eta_values:
        t0 = time.time()
        X_L, y_L = subsample_gold(X_L_full, y_L_full, eta, seed=seed)
        n_gold = len(X_L)

        # Standardize using combined scaler
        scaler = StandardScaler()
        scaler.fit(np.vstack([X_L, X_U]))
        X_L_s = scaler.transform(X_L)
        X_val_s = scaler.transform(X_val)

        # 1. Supervised SVM Baseline (on Gold Set only)
        base_svm = SVC(
            C=1.0,
            kernel="rbf",
            decision_function_shape="ovr",
            random_state=seed,
        )
        base_svm.fit(X_L_s, y_L)

        # Evaluate base SVM using wrapper interface
        class SVMWrapper:
            def __init__(self, clf, sc):
                self.clf = clf
                self.sc = sc
            def predict(self, X):
                return self.clf.predict(self.sc.transform(X))
            def predict_proba(self, X):
                sc = self.clf.decision_function(self.sc.transform(X))
                if sc.ndim == 1:
                    sc = np.column_stack([-sc, sc])
                n_c = len(self.clf.classes_)
                if n_c < 5 or not np.array_equal(self.clf.classes_, np.arange(5)):
                    aligned = np.full((len(X), 5), -1e9, dtype=float)
                    for i, c in enumerate(self.clf.classes_):
                        if 0 <= c < 5:
                            aligned[:, c] = sc[:, i]
                    sc = aligned
                sc = sc - sc.max(axis=1, keepdims=True)
                ex = np.exp(sc)
                return ex / ex.sum(axis=1, keepdims=True)

        res_base_svm = evaluate_model(SVMWrapper(base_svm, scaler), X_val, y_val)
        base_sv_count = int(np.sum(base_svm.n_support_))

        # 2. Transductive SVM
        tsvm = TSVMClassifier(
            C_L=1.0,
            C_U=0.005,
            C_U_max=0.5,
            kernel="rbf",
            max_iter=3,
            random_state=seed,
        )
        tsvm.fit(X_L, y_L, X_U)
        res_tsvm = evaluate_model(tsvm, X_val, y_val)
        tsvm_sv_count = tsvm.total_support_vectors_

        row = {
            "eta": eta,
            "n_gold": n_gold,
            "n_unlabeled": len(X_U),
            "tsvm_macro_f1": res_tsvm["macro_f1"],
            "tsvm_roc_auc": res_tsvm["roc_auc"],
            "tsvm_pr_auc": res_tsvm["pr_auc"],
            "tsvm_total_sv": tsvm_sv_count,
            "base_svm_macro_f1": res_base_svm["macro_f1"],
            "base_svm_roc_auc": res_base_svm["roc_auc"],
            "base_svm_pr_auc": res_base_svm["pr_auc"],
            "base_svm_total_sv": base_sv_count,
            "f1_gain_over_base": round(res_tsvm["macro_f1"] - res_base_svm["macro_f1"], 4),
        }

        # Record SV counts per class
        for i, cls in enumerate(SUPERCLASSES):
            n_sv_c = int(tsvm.n_support_[i]) if i < len(tsvm.n_support_) else 0
            row[f"tsvm_sv_{cls}"] = n_sv_c
            row[f"tsvm_f1_{cls}"] = res_tsvm["per_class_f1"].get(cls, 0.0)
            row[f"base_f1_{cls}"] = res_base_svm["per_class_f1"].get(cls, 0.0)

        # Optional test evaluation
        if X_test is not None and y_test is not None:
            res_test = evaluate_model(tsvm, X_test, y_test)
            row["test_tsvm_macro_f1"] = res_test["macro_f1"]
            row["test_tsvm_roc_auc"] = res_test["roc_auc"]

        rows.append(row)
        sv_breakdowns.append({
            "eta": eta,
            "tsvm_n_support": tsvm.n_support_,
            "base_n_support": base_svm.n_support_,
        })

        elapsed = time.time() - t0
        print(
            f"  [TSVM]        η = {eta:4.2f} ({n_gold:4d} gold) | "
            f"Base F1: {res_base_svm['macro_f1']:.4f} | "
            f"TSVM F1: {res_tsvm['macro_f1']:.4f} (Δ {row['f1_gain_over_base']:+.4f}) | "
            f"SVs: {tsvm_sv_count} | {elapsed:.1f}s"
        )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "tsvm_eta_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[WP4] Saved TSVM sweep results to: {csv_path}")

    # Plot 1: F1 vs eta for TSVM
    plot_f1_vs_eta_tsvm(df, out_dir)

    # Plot 2: Margin stability and support vector inflation under class imbalance
    plot_margin_stability_imbalance(df, sv_breakdowns, out_dir)

    return df


def plot_f1_vs_eta_cotraining(df: pd.DataFrame, out_dir: Path):
    """Plot Macro F1 vs Gold Fraction (η) for Co-Training and Baselines."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    eta_pct = df["eta"] * 100

    ax.plot(
        eta_pct, df["cotrain_macro_f1"],
        marker="o", linewidth=2.5, color="#1f77b4",
        label="Co-Training (Two-View)",
    )
    ax.plot(
        eta_pct, df["base_full_macro_f1"],
        marker="s", linewidth=2.0, linestyle="--", color="#2ca02c",
        label="Supervised Baseline (Full Features, Gold)",
    )
    ax.plot(
        eta_pct, df["base_v1_macro_f1"],
        marker="^", linewidth=1.5, linestyle=":", color="#ff7f0e", alpha=0.8,
        label="View 1 Baseline (Temporal / Morphology)",
    )
    ax.plot(
        eta_pct, df["base_v2_macro_f1"],
        marker="v", linewidth=1.5, linestyle=":", color="#9467bd", alpha=0.8,
        label="View 2 Baseline (Spectral / Wavelet)",
    )

    ax.set_title("Co-Training: Macro F1 vs. Gold Set Fraction (η)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Gold Label Fraction η (%)", fontsize=11)
    ax.set_ylabel("Macro F1 Score (Validation Set)", fontsize=11)
    ax.set_xticks(eta_pct)
    ax.set_xticklabels([f"{v:.0f}%" for v in eta_pct])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10, loc="lower right")

    plt.tight_layout()
    plot_path = out_dir / "f1_vs_eta_cotraining.png"
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"[WP4] Generated plot: {plot_path}")


def plot_view_agreement(curves: dict[float, list[dict]], out_dir: Path):
    """Plot Inter-View Agreement Rate across Co-Training Iteration Rounds."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
    for i, (eta, history) in enumerate(curves.items()):
        if not history:
            continue
        rounds = [h["round"] for h in history]
        agreements = [h["agreement"] * 100 for h in history]
        color = colors[i % len(colors)]
        ax.plot(
            rounds, agreements,
            marker="o", markersize=4, linewidth=2, color=color,
            label=f"η = {eta*100:.0f}% ({history[0].get('pool1_size', 0)} gold)",
        )

    ax.set_title("Inter-View Classifier Agreement Dynamics Across Rounds", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Co-Training Round (Iteration)", fontsize=11)
    ax.set_ylabel("Inter-View Agreement Rate on Untrusted Pool (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10, loc="lower right")

    plt.tight_layout()
    plot_path = out_dir / "view_agreement.png"
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"[WP4] Generated plot: {plot_path}")


def plot_f1_vs_eta_tsvm(df: pd.DataFrame, out_dir: Path):
    """Plot Macro F1 vs Gold Fraction (η) for TSVM vs Supervised SVM."""
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

    eta_pct = df["eta"] * 100

    line1 = ax1.plot(
        eta_pct, df["tsvm_macro_f1"],
        marker="o", linewidth=2.5, color="#d62728",
        label="Transductive SVM (TSVM)",
    )
    line2 = ax1.plot(
        eta_pct, df["base_svm_macro_f1"],
        marker="s", linewidth=2.0, linestyle="--", color="#1f77b4",
        label="Supervised SVM Baseline (Gold)",
    )

    ax1.set_title("Transductive SVM: Macro F1 vs. Gold Set Fraction (η)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Gold Label Fraction η (%)", fontsize=11)
    ax1.set_ylabel("Macro F1 Score (Validation Set)", fontsize=11, color="#333333")
    ax1.set_xticks(eta_pct)
    ax1.set_xticklabels([f"{v:.0f}%" for v in eta_pct])
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Twin axis for Support Vectors
    ax2 = ax1.twinx()
    line3 = ax2.plot(
        eta_pct, df["tsvm_total_sv"],
        marker="^", linewidth=1.8, linestyle=":", color="#7f7f7f", alpha=0.85,
        label="TSVM Support Vector Count",
    )
    ax2.set_ylabel("Total Support Vectors", fontsize=11, color="#7f7f7f")
    ax2.tick_params(axis="y", labelcolor="#7f7f7f")

    # Combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=True, fontsize=10, loc="lower right")

    plt.tight_layout()
    plot_path = out_dir / "f1_vs_eta_tsvm.png"
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"[WP4] Generated plot: {plot_path}")


def plot_margin_stability_imbalance(df: pd.DataFrame, sv_breakdowns: list[dict], out_dir: Path):
    """
    Analyse and visualize margin stability and support vector inflation under class imbalance (Proposal §5.2).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot 1: Support vector count per class across eta in TSVM
    colors = ["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e", "#9467bd"]
    eta_pct = df["eta"] * 100

    for i, cls in enumerate(SUPERCLASSES):
        col_name = f"tsvm_sv_{cls}"
        if col_name in df.columns:
            ax1.plot(
                eta_pct, df[col_name],
                marker="o", linewidth=2.0, color=colors[i],
                label=f"{cls}",
            )

    ax1.set_title("TSVM Support Vectors per Superclass vs. η", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Gold Label Fraction η (%)", fontsize=10)
    ax1.set_ylabel("Number of Support Vectors", fontsize=10)
    ax1.set_xticks(eta_pct)
    ax1.set_xticklabels([f"{v:.0f}%" for v in eta_pct])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(title="Superclass", frameon=True, fontsize=9)

    # Subplot 2: Per-Class F1 comparison at η = 0.10 (representative benchmark)
    rep_row = df[df["eta"] == 0.10]
    if rep_row.empty:
        rep_row = df.iloc[len(df) // 2 : len(df) // 2 + 1]

    r = rep_row.iloc[0]
    x = np.arange(len(SUPERCLASSES))
    width = 0.35

    tsvm_f1s = [r.get(f"tsvm_f1_{cls}", 0.0) for cls in SUPERCLASSES]
    base_f1s = [r.get(f"base_f1_{cls}", 0.0) for cls in SUPERCLASSES]

    ax2.bar(x - width/2, base_f1s, width, label="Supervised SVM", color="#1f77b4", alpha=0.85)
    ax2.bar(x + width/2, tsvm_f1s, width, label="TSVM", color="#d62728", alpha=0.85)

    ax2.set_title(f"Per-Class F1 Breakdown under Class Imbalance (η = {r['eta']*100:.0f}%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("ECG Superclass", fontsize=10)
    ax2.set_ylabel("F1 Score", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(SUPERCLASSES, fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.legend(frameon=True, fontsize=10)

    # Annotate class frequencies
    frequencies = ["NORM (62.5%)", "MI (10.4%)", "STTC (13.9%)", "CD (9.9%)", "HYP (3.3%)"]
    for i, freq in enumerate(frequencies):
        ax2.text(i, 0.03, freq, ha="center", fontsize=7.5, rotation=90, color="white", fontweight="bold")

    plt.tight_layout()
    plot_path = out_dir / "margin_stability_imbalance.png"
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"[WP4] Generated plot: {plot_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 65)
    print("  BITS F464: ML Project - WP4 Co-Training & Transductive SVMs")
    print("#" * 65)

    # 1. Load Data Splits
    split = load_standard_split(
        args.features_path,
        args.labels_path,
        None,
    )

    X_L_full = split["X_train_L"]
    y_L_full = split["y_train_L"]
    X_U      = split["X_train_U"]
    X_val    = split["X_val"]
    y_val    = split["y_val"]
    X_test   = split["X_test"] if args.eval_test else None
    y_test   = split["y_test"] if args.eval_test else None

    if args.max_records is not None and args.max_records < len(X_U):
        print(f"[WP4] Subsampling unlabeled pool to max_records={args.max_records}...")
        rng = np.random.default_rng(args.seed)
        sub_u = rng.choice(len(X_U), size=args.max_records, replace=False)
        X_U = X_U[sub_u]

    eta_values = [float(v) for v in args.eta_values.split(",")]

    print(f"[WP4] Loaded dataset successfully:")
    print(f"[WP4]   Gold Set (D_L)    : {len(X_L_full):5d} records (84 features)")
    print(f"[WP4]   Untrusted Pool(D_U): {len(X_U):5d} records")
    print(f"[WP4]   Validation Set    : {len(X_val):5d} records")
    if args.eval_test:
        print(f"[WP4]   Test Set          : {len(X_test):5d} records")
    print(f"[WP4]   Gold Fractions (η): {eta_values}")

    # 2. Run Co-Training Sweep
    cotrain_df, _ = run_eta_sweep_cotraining(
        X_L_full=X_L_full,
        y_L_full=y_L_full,
        X_U=X_U,
        X_val=X_val,
        y_val=y_val,
        eta_values=eta_values,
        out_dir=out_dir,
        seed=args.seed,
        X_test=X_test,
        y_test=y_test,
    )

    # 3. Run TSVM Sweep
    tsvm_df = run_eta_sweep_tsvm(
        X_L_full=X_L_full,
        y_L_full=y_L_full,
        X_U=X_U,
        X_val=X_val,
        y_val=y_val,
        eta_values=eta_values,
        out_dir=out_dir,
        seed=args.seed,
        X_test=X_test,
        y_test=y_test,
    )

    # 4. Print Summary Comparison Table
    print("\n" + "=" * 80)
    print("  WP4 SUMMARY RESULTS TABLE: F1 vs. Gold Fraction (η)")
    print("=" * 80)
    print(f"{'η (%)':<8} {'Gold N':<8} {'CoTrain F1':<12} {'Base LR F1':<12} {'Δ CoTrain':<12} {'TSVM F1':<12} {'Base SVM F1':<12} {'Δ TSVM':<12}")
    print("-" * 80)
    for eta in eta_values:
        c_row = cotrain_df[cotrain_df["eta"] == eta].iloc[0]
        t_row = tsvm_df[tsvm_df["eta"] == eta].iloc[0]
        print(
            f"{eta*100:<7.1f}% {int(c_row['n_gold']):<8d} "
            f"{c_row['cotrain_macro_f1']:<12.4f} {c_row['base_full_macro_f1']:<12.4f} {c_row['f1_gain_over_base']:<+12.4f} "
            f"{t_row['tsvm_macro_f1']:<12.4f} {t_row['base_svm_macro_f1']:<12.4f} {t_row['f1_gain_over_base']:<+12.4f}"
        )
    print("=" * 80)
    print(f"[WP4] All deliverables written to directory: {out_dir}\n")


if __name__ == "__main__":
    main()
