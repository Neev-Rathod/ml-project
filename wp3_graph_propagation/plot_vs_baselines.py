"""
Plot LP vs gold-only baselines.  Run from repo root:
    python wp3_graph_propagation/plot_vs_baselines.py --out_dir outputs/wp3
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# LP configs to show: (distance, n_pca, k)
LP_CONFIGS = [("euclidean", 5, 3), ("euclidean", 10, 5), ("cosine", 40, 3)]

p = argparse.ArgumentParser()
p.add_argument("--out_dir", default="outputs/wp3")
a = p.parse_args()
out = Path(a.out_dir)

lp = pd.read_csv(out / "wp3_results_summary.csv")
bl = pd.read_csv(out / "gold_baselines.csv")

fig, ax = plt.subplots(figsize=(10, 5))
for name, g in bl.groupby("model"):
    s = g.groupby("eta").macro_f1.agg(["mean", "std"]).fillna(0)
    ax.errorbar(s.index * 100, s["mean"], yerr=s["std"], ls="--", marker="s", capsize=3, label=f"{name} (gold only)")
for d, pc, k in LP_CONFIGS:
    g = lp[(lp.distance == d) & (lp.n_pca == pc) & (lp.k == k)].sort_values("eta")
    ax.errorbar(g.eta * 100, g.macro_f1, yerr=g.macro_f1_std.fillna(0), marker="o", capsize=3, lw=2,
                label=f"Label prop. ({d}, PCA={pc}, k={k})")
ax.set_xscale("log"); ax.set_xticks([2, 5, 10, 20, 50]); ax.set_xticklabels(["2", "5", "10", "20", "50"])
ax.set_xlabel("eta (% of Gold Set labeled)"); ax.set_ylabel("Macro F1 (validation)")
ax.set_title("Label propagation vs gold-only supervised baselines")
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.grid(alpha=.3); ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout(); fig.savefig(out / "wp3_vs_baselines.png", dpi=150, bbox_inches="tight")
print("saved", out / "wp3_vs_baselines.png")