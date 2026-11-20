"""
wp3_graph_propagation/eval_final.py
===================================
Compares graph label propagation against a gold-only kNN control, using the
SAME preprocessing (scaler / PCA / whitening), same k, same labeled subsets.
The only difference: the control ignores the unlabeled pool when predicting.

  python wp3_graph_propagation/eval_final.py --split val     # control comparison
  python wp3_graph_propagation/eval_final.py --split test    # final number: run ONCE
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wp3_graph_propagation.label_propagation import GraphLabelPropagator
from wp3_graph_propagation.run_wp3 import subsample_gold
from shared.data_utils import load_standard_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_path", default=str(ROOT / "outputs/wp1/features.npz"))
    p.add_argument("--labels_path", default=str(ROOT / "outputs/wp1/labels.npz"))
    p.add_argument("--out_dir", default=str(ROOT / "outputs/wp3"))
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--distance", default="euclidean")
    p.add_argument("--n_pca", type=int, default=5)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--eta_values", default="0.02,0.05,0.10,0.20,0.50")
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    s = load_standard_split(a.features_path, a.labels_path, None)
    X_eval, y_eval = (s["X_test"], s["y_test"]) if a.split == "test" else (s["X_val"], s["y_val"])
    etas = [float(v) for v in a.eta_values.split(",")]
    metric = "cosine" if a.distance == "cosine" else "euclidean"
    print(f"[WP3] split={a.split} config: {a.distance}, PCA={a.n_pca}, k={a.k}")

    rows = []
    for eta in etas:
        for i in range(a.n_seeds):
            seed = a.seed + i
            idx = subsample_gold(s["y_train_L"], eta, seed)
            X_L, y_L = s["X_train_L"][idx], s["y_train_L"][idx]

            m = GraphLabelPropagator(k=a.k, n_pca=a.n_pca, distance=a.distance)
            m.fit(X_L, y_L, s["X_train_U"])
            f1_graph = f1_score(y_eval, m.predict(X_eval), average="macro")

            # control: kNN on labeled points only, same embedding
            knn = KNeighborsClassifier(n_neighbors=min(a.k, len(X_L)),
                                       weights="distance", metric=metric)
            knn.fit(m._embed(X_L), y_L)
            f1_knn = f1_score(y_eval, knn.predict(m._embed(X_eval)), average="macro")

            rows.append(dict(eta=eta, seed=seed, n_labeled=len(X_L),
                             graph_f1=f1_graph, gold_only_knn_f1=f1_knn))

    df = pd.DataFrame(rows)
    df.to_csv(out / f"control_{a.split}_raw.csv", index=False)
    g = df.groupby("eta").agg(graph_f1=("graph_f1", "mean"), graph_std=("graph_f1", "std"),
                              knn_f1=("gold_only_knn_f1", "mean"),
                              knn_std=("gold_only_knn_f1", "std"))
    g["gain"] = g.graph_f1 - g.knn_f1
    g.to_csv(out / f"control_{a.split}.csv")
    print(g.round(4))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.errorbar(g.index * 100, g.graph_f1, yerr=g.graph_std.fillna(0), marker="o", capsize=3,
                label="Graph label propagation")
    ax.errorbar(g.index * 100, g.knn_f1, yerr=g.knn_std.fillna(0), marker="s", capsize=3,
                label="Gold-only kNN (control)")
    ax.set_xlabel("Gold fraction eta (%)"); ax.set_ylabel(f"Macro F1 ({a.split})")
    ax.set_title(f"{a.distance}, PCA={a.n_pca}, k={a.k}"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / f"control_{a.split}.png", dpi=150)


if __name__ == "__main__":
    main()
