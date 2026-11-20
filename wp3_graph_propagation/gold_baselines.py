"""
Gold-only supervised baselines on the SAME eta grid / seeds / subsamples as run_wp3.py.
    python wp3_graph_propagation/gold_baselines.py \
        --features_path outputs/wp1/features.npz --labels_path outputs/wp1/labels.npz \
        --out_dir outputs/wp3 --split val        # model selection / comparison
    ... --split test                             # held-out fold 10
Writes gold_baselines.csv (val) or gold_baselines_test.csv (test).
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from shared.data_utils import load_standard_split, SUPERCLASSES
from shared.metrics import evaluate_model
from wp3_graph_propagation.run_wp3 import subsample_gold

MODELS = {
    "LogReg (balanced)":   lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
    "LogReg (unweighted)": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "k-NN (k=5)":          lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "Gaussian NB":         lambda: make_pipeline(StandardScaler(), GaussianNB()),
    "SVM-RBF (balanced)":  lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=0)),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_path", required=True)
    p.add_argument("--labels_path", required=True)
    p.add_argument("--out_dir", default="outputs/wp3")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--eta_values", default="0.02,0.05,0.10,0.20,0.50")
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_svm", action="store_true")
    a = p.parse_args()

    s = load_standard_split(a.features_path, a.labels_path, None)
    XL, yL = s["X_train_L"], s["y_train_L"]
    Xe, ye = (s["X_test"], s["y_test"]) if a.split == "test" else (s["X_val"], s["y_val"])
    etas = [float(v) for v in a.eta_values.split(",")]
    print(f"[WP3] baselines on split={a.split}")
    rows = []
    for eta in etas:
        for seed in [a.seed + i for i in range(a.n_seeds)]:
            idx = subsample_gold(yL, eta, seed)          # identical subsample to run_wp3.py
            for name, mk in MODELS.items():
                if a.skip_svm and name.startswith("SVM"):
                    continue
                m = mk().fit(XL[idx], yL[idx])
                r = evaluate_model(m, Xe, ye)
                row = dict(model=name, split=a.split, eta=eta, seed=seed, n_labeled=len(idx),
                           macro_f1=r["macro_f1"], roc_auc=r.get("roc_auc"), pr_auc=r.get("pr_auc"))
                pcf = r.get("per_class_f1")
                if pcf is not None:
                    vals = list(pcf.values()) if isinstance(pcf, dict) else np.ravel(pcf)
                    for c, v in enumerate(vals):
                        row[f"f1_{SUPERCLASSES[c]}"] = float(v)
                rows.append(row)
                print(f"eta={eta} seed={seed} {name}: F1={r['macro_f1']:.4f}", flush=True)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / ("gold_baselines.csv" if a.split == "val" else "gold_baselines_test.csv"), index=False)
    print(df.groupby(["model", "eta"]).macro_f1.mean().unstack().round(4))
    f1cols = [c for c in df.columns if c.startswith("f1_")]
    if f1cols:
        print("\nPer-class F1 at eta=0.10 (mean over seeds):")
        print(df[df.eta == 0.10].groupby("model")[f1cols].mean().round(3))

if __name__ == "__main__":
    main()
