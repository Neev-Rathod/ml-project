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
    python run_wp2.py --mode synthetic --out_dir ..\outputs\wp2

    # PTB-XL features (requires WP1 to have run first):
    python run_wp2.py --mode ptbxl \
        --features_path ..\outputs\wp1\features.npz \
        --labels_path   ..\outputs\wp1\labels.npz   \
        --out_dir       ..\outputs\wp2

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
    Save results CSV and F1-vs-epsilon plot.

    TODO for Arnav (WP2)
    --------------------
    Implement this function. Structure:
      for eps in epsilon_values:
          y_U_noisy = inject_uniform_noise(y_U_clean, eps)
          for clf_class in [GNB, LR, kNN, DT, SVM]:
              for mode in MODES_TRAIN:
                  clf = clf_class(mode=mode)
                  clf.fit(X_L, y_L, X_U, y_U_noisy)
                  results = evaluate_model(clf, X_val, y_val)
                  record results
      plot macro F1 vs. epsilon for each model/mode combination.
    """
    raise NotImplementedError("run_breakdown_analysis not yet implemented")


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

    run_breakdown_analysis(
        X_L, y_L,
        X_U, y_U_clean,
        X_val, y_val,
        epsilon_values,
        out_dir,
        seed=args.seed,
    )

    print(f"[WP2] Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
