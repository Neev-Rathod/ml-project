"""Builds eta_sweep.csv, graph_analysis.csv, f1_vs_eta.png from the sweep output."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/wp3")
s = pd.read_csv(out / "wp3_results_summary.csv")

s.to_csv(out / "eta_sweep.csv", index=False)
s[["distance", "n_pca", "k", "eta", "graph_components", "unreachable"]].to_csv(
    out / "graph_analysis.csv", index=False)

variants = {
    "Raw + Euclidean":   (s.distance == "euclidean") & (s.n_pca == 0),
    "PCA + Euclidean":   (s.distance == "euclidean") & (s.n_pca > 0),
    "PCA + Cosine":      (s.distance == "cosine") & (s.n_pca > 0),
    "PCA + Mahalanobis": (s.distance == "mahalanobis") & (s.n_pca > 0),
}
fig, ax = plt.subplots(figsize=(6.8, 4.6))
table = {}
for name, mask in variants.items():
    g = s[mask]
    p, k = g.groupby(["n_pca", "k"]).macro_f1.mean().idxmax()   # best config by mean F1
    t = g[(g.n_pca == p) & (g.k == k)].sort_values("eta")
    ax.errorbar(t.eta * 100, t.macro_f1, yerr=t.macro_f1_std.fillna(0), marker="o",
                capsize=3, label=f"{name} (PCA={p}, k={k})")
    table[name] = t.set_index("eta").macro_f1
ax.set_xlabel("Gold fraction eta (%)"); ax.set_ylabel("Macro F1 (validation)")
ax.set_title("Graph label propagation: F1 vs eta"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(out / "f1_vs_eta.png", dpi=150)

print("\nBest config per variant, macro F1 by eta:")
print(pd.DataFrame(table).round(3))
print("\nEffect of k at eta=0.02 (mean over distance, PCA):")
print(s[s.eta == 0.02].groupby("k").macro_f1.mean().round(3))
print("\nStd across seeds for the best PCA+Mahalanobis vs Raw+Euclidean is in eta_sweep.csv (macro_f1_std)")
