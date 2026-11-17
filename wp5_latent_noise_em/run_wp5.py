"""
wp5_latent_noise_em/run_wp5.py
================================
Entry point for Work Package 5.

Experiments
-----------
1. Discard vs. Correct comparison (Proposal §5.3):
   Grid sweep over η ∈ {2%, 5%, 10%, 20%, 50%} and ε ∈ {0%, 15%, 30%, 45%}.
   Compare LatentNoiseEM against GraphLP (WP3) and CoTrainer (WP4).

2. Clinical Annotator Bias Audit (Proposal §5.4):
   On PTB-XL, fit LatentNoiseEM on the real noisy pool and visualise the
   learned confusion matrix T̂ as a 5×5 heatmap.

3. Synthetic Frobenius Norm Recovery (Proposal §5.4):
   On the synthetic dataset (from WP2), verify that T̂ converges to T_true
   as a function of n_U. Report ‖T̂ - T_true‖_F.

Usage
-----
    python run_wp5.py \
        --features_path ..\outputs\wp1\features.npz \
        --labels_path   ..\outputs\wp1\labels.npz   \
        --out_dir       ..\outputs\wp5

Arguments
---------
--features_path  Path to WP1 features.npz
--labels_path    Path to WP1 labels.npz
--out_dir        Output directory
--eta_values     Gold-label fractions
--epsilon_values Noise rates for the grid sweep
--seed           Random seed (default 42)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wp5_latent_noise_em.latent_noise_em import LatentNoiseEM
from shared.data_utils import load_standard_split, SUPERCLASSES
from shared.metrics import evaluate_model, frobenius_error, print_results


def parse_args():
    p = argparse.ArgumentParser(description="WP5 – Latent-Noise EM & Evaluation Suite")
    p.add_argument("--features_path",  required=True)
    p.add_argument("--labels_path",    required=True)
    p.add_argument("--out_dir",        default=str(Path("..") / "outputs" / "wp5"))
    p.add_argument("--eta_values",     default="0.02,0.05,0.10,0.20,0.50")
    p.add_argument("--epsilon_values", default="0.0,0.15,0.30,0.45")
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


def inject_uniform_noise(y: np.ndarray, epsilon: float, n_classes: int = 5,
                         seed: int = 42) -> np.ndarray:
    """Inject uniform noise into labels (standalone copy so WP5 is self-contained)."""
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    for i in range(len(y)):
        if rng.random() < epsilon:
            choices = [c for c in range(n_classes) if c != y[i]]
            y_noisy[i] = rng.choice(choices)
    return y_noisy


def run_discard_vs_correct(
    X_L_full, y_L_full, X_U, y_U_true,
    X_val, y_val,
    eta_values, epsilon_values,
    out_dir: Path, seed: int = 42,
):
    """
    2D grid sweep: for each (η, ε) pair, compare LatentNoiseEM against baselines.

    TODO for Tanuj (WP5)
    --------------------
    For each eta in eta_values:
        For each epsilon in epsilon_values:
            1. Subsample Gold Set to fraction eta.
            2. Inject uniform noise at rate epsilon into y_U.
            3. Fit LatentNoiseEM(X_L, y_L, X_U, y_U_noisy).
            4. Also fit GraphLabelPropagator and CoTrainer (import from WP3/WP4).
            5. Evaluate all three on X_val, y_val.
            6. Record macro F1 for each method.
    Save results to discard_vs_correct.csv.
    Plot: heatmap of EM F1 - GraphLP F1 (positive = EM wins).
    """
    raise NotImplementedError("run_discard_vs_correct not yet implemented")


def run_clinical_audit(
    X_L, y_L, X_U, y_U_noisy,
    out_dir: Path,
):
    """
    Fit LatentNoiseEM on PTB-XL real noisy labels and visualise T̂.

    TODO for Tanuj (WP5)
    --------------------
    1. Fit LatentNoiseEM with the real noisy labels.
    2. Call model.get_confusion_matrix() to retrieve T̂.
    3. Plot T̂ as a seaborn heatmap with class names as axis labels.
    4. Annotate the top-3 most confused class pairs.
    5. Save figure and T_hat as a CSV.
    """
    raise NotImplementedError("run_clinical_audit not yet implemented")


def run_frobenius_recovery(
    X_syn, y_syn, T_true: np.ndarray,
    out_dir: Path, seed: int = 42,
):
    """
    On synthetic data with known T_true, fit LatentNoiseEM for increasing
    n_U and plot ‖T̂ - T_true‖_F vs. n_U.

    TODO for Tanuj (WP5)
    --------------------
    n_U_values = [100, 300, 500, 1000, 2000, 5000]
    for n_U in n_U_values:
        subsample n_U points from X_syn, y_syn as the unlabeled pool
        inject asymmetric noise using T_true
        fit LatentNoiseEM
        record frobenius_error(model.T_hat_, T_true)
    Plot error vs n_U and save.
    """
    raise NotImplementedError("run_frobenius_recovery not yet implemented")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = load_standard_split(
        args.features_path,
        args.labels_path,
        None,
    )

    eta_values     = [float(v) for v in args.eta_values.split(",")]
    epsilon_values = [float(v) for v in args.epsilon_values.split(",")]

    print("[WP5] Data loaded.")
    print(f"[WP5]   Gold Set : {len(split['X_train_L'])} records")
    print(f"[WP5]   Pool     : {len(split['X_train_U'])} records")

    # TODO: Uncomment and implement once the sub-functions are ready.
    # run_discard_vs_correct(...)
    # run_clinical_audit(...)
    # run_frobenius_recovery(...)   ← requires synthetic data from WP2

    print("[WP5] TODO: implement experiment functions (see README for guidance).")
    print(f"[WP5] Done. Outputs will be in {out_dir}")


if __name__ == "__main__":
    main()
