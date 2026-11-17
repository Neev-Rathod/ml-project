"""
wp3_graph_propagation/run_wp3.py
==================================
Entry point for Work Package 3.

Runs the Graph Label Propagation experiments (Proposal §4.3, §5.3).

Usage
-----
    python run_wp3.py \
        --features_path ..\outputs\wp1\features.npz \
        --labels_path   ..\outputs\wp1\labels.npz   \
        --out_dir       ..\outputs\wp3

Arguments
---------
--features_path   Path to WP1 features.npz
--labels_path     Path to WP1 labels.npz
--out_dir         Output directory
--k_values        Comma-separated kNN values to sweep, e.g. "5,10,20"
--pca_components  PCA dims to test, e.g. "10,20,40"
--eta_values      Gold-label fractions, e.g. "0.02,0.05,0.10,0.20,0.50"
--seed            Random seed (default 42)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wp3_graph_propagation.label_propagation import GraphLabelPropagator
from shared.data_utils import load_standard_split
from shared.metrics import evaluate_model, print_results


def parse_args():
    p = argparse.ArgumentParser(description="WP3 – Graph Label Propagation")
    p.add_argument("--features_path",  required=True)
    p.add_argument("--labels_path",    required=True)
    p.add_argument("--out_dir",        default=str(Path("..") / "outputs" / "wp3"))
    p.add_argument("--k_values",       default="5,10,20")
    p.add_argument("--pca_components", default="10,20,40")
    p.add_argument("--eta_values",     default="0.02,0.05,0.10,0.20,0.50")
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


def subsample_gold(X_L, y_L, eta: float, seed: int) -> tuple:
    """Randomly subsample a fraction eta of the Gold Set."""
    rng = np.random.default_rng(seed)
    n = max(1, int(eta * len(X_L)))
    idx = rng.choice(len(X_L), size=n, replace=False)
    return X_L[idx], y_L[idx]


def run_eta_sweep(
    X_L_full: np.ndarray,
    y_L_full: np.ndarray,
    X_U: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    eta_values: list,
    model_kwargs: dict,
    out_dir: Path,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Vary the fraction η of the labeled Gold Set and record macro F1.

    TODO for Neev (WP3)
    --------------------
    Implement:
      for eta in eta_values:
          X_L, y_L = subsample_gold(X_L_full, y_L_full, eta, seed)
          model = GraphLabelPropagator(**model_kwargs)
          model.fit(X_L, y_L, X_U)
          result = evaluate_model(model, X_val, y_val)
          record eta and result["macro_f1"]
      plot F1 vs. eta (line chart)
      save CSV
    """
    raise NotImplementedError("run_eta_sweep not yet implemented")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = load_standard_split(
        args.features_path,
        args.labels_path,
        None,
    )

    k_values   = [int(v)   for v in args.k_values.split(",")]
    pca_values = [int(v)   for v in args.pca_components.split(",")]
    eta_values = [float(v) for v in args.eta_values.split(",")]

    print("[WP3] Data loaded.")
    print(f"[WP3]   Gold Set : {len(split['X_train_L'])} records")
    print(f"[WP3]   Pool     : {len(split['X_train_U'])} records")
    print(f"[WP3]   Val      : {len(split['X_val'])} records")
    print(f"[WP3]   Test     : {len(split['X_test'])} records")

    # TODO: Loop over k_values and pca_values, call run_eta_sweep for each combo.
    # Example for one setting:
    #   run_eta_sweep(
    #       split["X_train_L"], split["y_train_L"],
    #       split["X_train_U"],
    #       split["X_val"], split["y_val"],
    #       eta_values,
    #       model_kwargs={"k": 10, "n_pca": 20, "distance": "euclidean"},
    #       out_dir=out_dir,
    #       seed=args.seed,
    #   )
    print("[WP3] TODO: implement the main sweep loop (see README for guidance).")
    print(f"[WP3] Done. Outputs will be in {out_dir}")


if __name__ == "__main__":
    main()
