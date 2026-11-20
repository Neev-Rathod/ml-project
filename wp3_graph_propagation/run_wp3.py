"""
wp3_graph_propagation/run_wp3.py
================================
Graph Label Propagation experiments (Proposal 4.3, 5.3).
Sweeps distance metric x PCA dims x k x eta (x seeds), evaluates on the
VALIDATION fold only (fold 10 / test is never touched here).

Run from the repo root:
    python wp3_graph_propagation/run_wp3.py \
        --features_path outputs/wp1/features.npz \
        --labels_path   outputs/wp1/labels.npz \
        --out_dir       outputs/wp3
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wp3_graph_propagation.label_propagation import GraphLabelPropagator
from shared.data_utils import load_standard_split, SUPERCLASSES
from shared.metrics import evaluate_model


def parse_args():
    p = argparse.ArgumentParser(description="WP3 - Graph Label Propagation")
    p.add_argument("--features_path", required=True)
    p.add_argument("--labels_path", required=True)
    p.add_argument("--out_dir", default=str(ROOT / "outputs" / "wp3"))
    p.add_argument("--k_values", default="5,10,20")
    p.add_argument("--pca_components", default="10,20,40")
    p.add_argument("--eta_values", default="0.02,0.05,0.10,0.20,0.50")
    p.add_argument("--distances", default="euclidean,cosine,mahalanobis")
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_unused_gold", action="store_true",
                   help="Add the un-sampled Gold records (labels hidden) to the unlabeled pool")
    p.add_argument("--cmn", action="store_true", help="Class mass normalisation")
    return p.parse_args()


def subsample_gold(y_L, eta: float, seed: int) -> np.ndarray:
    """Stratified subsample: fraction eta of EACH class (>=1 per class). Returns indices."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(y_L):
        ids = np.where(y_L == c)[0]
        n = min(len(ids), max(1, int(round(eta * len(ids)))))
        chosen.append(rng.choice(ids, size=n, replace=False))
    return np.sort(np.concatenate(chosen))


def run_eta_sweep(X_L_full, y_L_full, X_U, X_val, y_val, eta_values,
                  model_kwargs, out_dir, seeds=(42,), use_unused_gold=False):
    rows = []
    for eta in eta_values:
        for seed in seeds:
            idx = subsample_gold(y_L_full, eta, seed)
            X_L, y_L = X_L_full[idx], y_L_full[idx]
            X_unl = X_U
            if use_unused_gold:
                rest = np.setdiff1d(np.arange(len(X_L_full)), idx)
                X_unl = np.vstack([X_U, X_L_full[rest]])

            model = GraphLabelPropagator(**model_kwargs)
            t0 = time.time()
            model.fit(X_L, y_L, X_unl)            # noisy labels deliberately NOT passed
            fit_s = time.time() - t0
            res = evaluate_model(model, X_val, y_val)

            row = dict(model_kwargs, eta=eta, seed=seed, n_labeled=len(X_L),
                       n_unlabeled=len(X_unl), macro_f1=float(res["macro_f1"]),
                       roc_auc=float(res.get("roc_auc", np.nan)),
                       pr_auc=float(res.get("pr_auc", np.nan)),
                       graph_components=model.n_graph_components_,
                       unreachable=model.n_unreachable_, fit_seconds=round(fit_s, 2))
            pcf = res.get("per_class_f1")
            if pcf is not None:
                vals = list(pcf.values()) if isinstance(pcf, dict) else np.ravel(pcf)
                for c, v in enumerate(vals):
                    row[f"f1_{SUPERCLASSES[c]}"] = float(v)
            rows.append(row)
    return pd.DataFrame(rows)


def make_plots(df, out_dir, eta_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    ax = axes[0]                                  
    for dist, g in df.groupby("distance"):
        k_best, p_best = g.groupby(["k", "n_pca"])["macro_f1"].mean().idxmax()
        s = g[(g.k == k_best) & (g.n_pca == p_best)].groupby("eta")["macro_f1"].agg(["mean", "std"]).fillna(0)
        ax.errorbar(s.index * 100, s["mean"], yerr=s["std"], marker="o", capsize=3,
                    label=f"{dist} (k={k_best}, PCA={p_best})")
    ax.set_xlabel("eta (% of Gold Set labeled)"); ax.set_ylabel("Macro F1 (val)")
    ax.set_title("Label propagation: F1 vs label scarcity"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    sub = df[np.isclose(df.eta, eta_ref)]
    for ax, col, ttl in ((axes[1], "k", "Sensitivity to k"), (axes[2], "n_pca", "Sensitivity to PCA dims")):
        sub.groupby(["distance", col])["macro_f1"].mean().unstack(0).plot(ax=ax, marker="o")
        ax.set_title(f"{ttl} (eta={eta_ref:.0%})"); ax.set_ylabel("Macro F1 (val)"); ax.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(Path(out_dir) / "wp3_summary.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    split = load_standard_split(args.features_path, args.labels_path, None)
    k_values = [int(v) for v in args.k_values.split(",")]
    pca_values = [int(v) for v in args.pca_components.split(",")]
    eta_values = [float(v) for v in args.eta_values.split(",")]
    distances = args.distances.split(",")
    seeds = [args.seed + i for i in range(args.n_seeds)]

    print(f"[WP3] Gold {len(split['X_train_L'])} | Pool {len(split['X_train_U'])} | "
          f"Val {len(split['X_val'])} | Test {len(split['X_test'])} (unused)")

    frames = []
    for dist in distances:
        for n_pca in pca_values:
            for k in k_values:
                print(f"[WP3] distance={dist} pca={n_pca} k={k}", flush=True)
                kw = dict(k=k, n_pca=n_pca, distance=dist, class_mass_norm=args.cmn)
                frames.append(run_eta_sweep(
                    split["X_train_L"], split["y_train_L"], split["X_train_U"],
                    split["X_val"], split["y_val"], eta_values, kw, out_dir,
                    seeds=seeds, use_unused_gold=args.use_unused_gold))
                pd.concat(frames).to_csv(out_dir / "wp3_results_raw.csv", index=False)

    df = pd.concat(frames, ignore_index=True)
    summary = (df.groupby(["distance", "n_pca", "k", "eta"], as_index=False)
                 .agg(macro_f1=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                      roc_auc=("roc_auc", "mean"), pr_auc=("pr_auc", "mean"),
                      unreachable=("unreachable", "mean"),
                      graph_components=("graph_components", "mean")))
    summary.to_csv(out_dir / "wp3_results_summary.csv", index=False)
    make_plots(df, out_dir, min(eta_values, key=lambda e: abs(e - 0.10)))

    print("\n[WP3] Top 5 configs by macro F1 (mean over eta and seeds):")
    print(df.groupby(["distance", "n_pca", "k"])["macro_f1"].mean()
            .sort_values(ascending=False).head(5).round(4))
    print(f"[WP3] Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()