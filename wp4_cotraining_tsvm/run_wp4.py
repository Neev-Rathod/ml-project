"""
wp4_cotraining_tsvm/run_wp4.py
================================
Entry point for Work Package 4.

Experiments
-----------
1. Sweep η (gold fraction) for Co-Training with varying k and threshold.
2. Sweep η for TSVM with varying C_U schedule.
3. Compare Co-Training agreement rates between views across rounds.
4. Analyse margin stability under class imbalance.

Usage
-----
    python run_wp4.py \
        --features_path ..\outputs\wp1\features.npz \
        --labels_path   ..\outputs\wp1\labels.npz   \
        --out_dir       ..\outputs\wp4

Arguments
---------
--features_path  Path to WP1 features.npz
--labels_path    Path to WP1 labels.npz
--out_dir        Output directory
--eta_values     Gold-label fractions, e.g. "0.02,0.05,0.10,0.20,0.50"
--seed           Random seed (default 42)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wp4_cotraining_tsvm.cotraining import CoTrainer
from wp4_cotraining_tsvm.tsvm import TSVMClassifier
from shared.data_utils import load_standard_split
from shared.metrics import evaluate_model, print_results


def parse_args():
    p = argparse.ArgumentParser(description="WP4 – Co-Training & Transductive SVMs")
    p.add_argument("--features_path", required=True)
    p.add_argument("--labels_path",   required=True)
    p.add_argument("--out_dir",       default=str(Path("..") / "outputs" / "wp4"))
    p.add_argument("--eta_values",    default="0.02,0.05,0.10,0.20,0.50")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


def subsample_gold(X_L, y_L, eta: float, seed: int):
    rng = np.random.default_rng(seed)
    n = max(1, int(eta * len(X_L)))
    idx = rng.choice(len(X_L), size=n, replace=False)
    return X_L[idx], y_L[idx]


def run_eta_sweep_cotraining(
    X_L_full, y_L_full, X_U, X_val, y_val,
    eta_values, out_dir, seed=42,
):
    """
    TODO for Pratyush (WP4)
    ------------------------
    For each eta, subsample the Gold Set, fit CoTrainer, evaluate on val set.
    Record macro F1 and inter-view agreement rate (fraction of unlabeled samples
    where h1 and h2 agree on the same class).
    Save results to CSV and plot F1 vs eta.
    """
    raise NotImplementedError("run_eta_sweep_cotraining not yet implemented")


def run_eta_sweep_tsvm(
    X_L_full, y_L_full, X_U, X_val, y_val,
    eta_values, out_dir, seed=42,
):
    """
    TODO for Pratyush (WP4)
    ------------------------
    For each eta, fit TSVMClassifier and evaluate on val set.
    Record macro F1 and number of support vectors (from the underlying SVCs).
    Save results to CSV and plot F1 vs eta.
    """
    raise NotImplementedError("run_eta_sweep_tsvm not yet implemented")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = load_standard_split(
        args.features_path,
        args.labels_path,
        None,
    )

    eta_values = [float(v) for v in args.eta_values.split(",")]

    print("[WP4] Data loaded.")
    print(f"[WP4]   Gold Set : {len(split['X_train_L'])} records")
    print(f"[WP4]   Pool     : {len(split['X_train_U'])} records")

    # TODO: Call the two sweep functions once implemented.
    # run_eta_sweep_cotraining(
    #     split["X_train_L"], split["y_train_L"],
    #     split["X_train_U"],
    #     split["X_val"], split["y_val"],
    #     eta_values, out_dir, args.seed,
    # )
    # run_eta_sweep_tsvm(...)
    print("[WP4] TODO: implement sweep functions (see README for guidance).")
    print(f"[WP4] Done. Outputs will be in {out_dir}")


if __name__ == "__main__":
    main()
