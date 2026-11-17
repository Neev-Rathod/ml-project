# WP3 – Graph Label Propagation
**Person:** Neev Rathod (2024A7PS0487G)

---

## What This WP Does

WP3 implements the first semi-supervised strategy from Proposal §4.3.

The core idea: **discard** the noisy labels in D_U entirely and instead spread the clean
Gold Set labels across a nearest-neighbor graph built from the feature geometry.

---

## Files in This Folder

| File | Purpose |
|------|---------|
| `run_wp3.py` | **Entry point** |
| `label_propagation.py` | kNN graph construction + harmonic label spreading |
| `README.md` | This file |

---

## Prerequisites

WP1 must have run to produce `outputs/wp1/features.npz` and `outputs/wp1/labels.npz`.

```bash
cd wp1_data_pipeline
python run_wp1.py --data_path D:\ml-project\dataset --out_dir ..\outputs\wp1
```

---

## How to Run

```bash
cd wp3_graph_propagation
python run_wp3.py \
    --features_path ..\outputs\wp1\features.npz \
    --labels_path   ..\outputs\wp1\labels.npz   \
    --out_dir       ..\outputs\wp3
```

---

## Implementation Guide

### Step 1 – `build_knn_graph()` in `label_propagation.py`

```python
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import lil_matrix

def build_knn_graph(X, k=10):
    N = len(X)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    # indices[:, 0] is the sample itself, skip it
    distances = distances[:, 1:]   # (N, k)
    indices   = indices[:, 1:]     # (N, k)

    # sigma_i = distance to the k-th neighbor (last column)
    sigma = distances[:, -1] + 1e-10   # (N,)

    W = lil_matrix((N, N), dtype=float)
    for i in range(N):
        for j_pos, j in enumerate(indices[i]):
            d = distances[i, j_pos]
            w = np.exp(-d**2 / (sigma[i] * sigma[j]))
            W[i, j] = w
            W[j, i] = w   # symmetry

    return W.tocsr()
```

### Step 2 – `harmonic_label_propagation()` in `label_propagation.py`

This solves Proposal Eq. 4.2: **F\*_U = (D_UU - W_UU)⁻¹ W_UL Y_L**

```python
def harmonic_label_propagation(W, labeled_indices, y_L, n_classes=5, reg=1e-6):
    N = W.shape[0]
    n_L = len(labeled_indices)
    all_idx = np.arange(N)
    unlabeled_mask = np.ones(N, dtype=bool)
    unlabeled_mask[labeled_indices] = False
    unlabeled_indices = all_idx[unlabeled_mask]

    # One-hot Y_L
    lb = LabelBinarizer()
    lb.fit(range(n_classes))
    Y_L = lb.transform(y_L).astype(float)   # (n_L, C)

    # Extract sub-matrices
    W_UU = W[unlabeled_indices, :][:, unlabeled_indices]
    W_UL = W[unlabeled_indices, :][:, labeled_indices]

    # Degree matrix for unlabeled block
    degree_UU = np.array(W[unlabeled_indices, :].sum(axis=1)).ravel()
    D_UU = diags(degree_UU)

    # Linear system: (D_UU - W_UU + reg*I) @ F_U = W_UL @ Y_L
    A = D_UU - W_UU + reg * eye(len(unlabeled_indices))
    B = W_UL @ Y_L   # (n_U, C)

    # Solve column by column
    F_U = np.zeros((len(unlabeled_indices), n_classes))
    for c in range(n_classes):
        F_U[:, c] = spsolve(A, B[:, c])

    # Assemble full F
    F = np.zeros((N, n_classes))
    F[labeled_indices] = Y_L
    F[unlabeled_indices] = np.clip(F_U, 0, None)

    # Row-normalise to get valid probabilities
    row_sums = F.sum(axis=1, keepdims=True) + 1e-10
    return F / row_sums
```

### Step 3 – `GraphLabelPropagator.fit()` in `label_propagation.py`

```python
def fit(self, X_L, y_L, X_U, y_U_noisy=None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    X_all = np.vstack([X_L, X_U])
    n_L   = len(X_L)

    # Scale
    scaler = StandardScaler()
    X_all  = scaler.fit_transform(X_all)

    # Optional PCA
    if self.n_pca > 0:
        pca   = PCA(n_components=self.n_pca)
        X_all = pca.fit_transform(X_all)

    # Optional distance transform
    if self.distance == "cosine":
        X_all = X_all / (np.linalg.norm(X_all, axis=1, keepdims=True) + 1e-10)
    elif self.distance == "mahalanobis":
        cov = np.cov(X_all.T) + 1e-3 * np.eye(X_all.shape[1])
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-10)
        inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        X_all = X_all @ inv_sqrt

    W = build_knn_graph(X_all, k=self.k)
    labeled_indices = np.arange(n_L)
    self._F = harmonic_label_propagation(W, labeled_indices, y_L)
    self._X_train = X_all
    self._scaler  = scaler
    self._pca     = pca if self.n_pca > 0 else None
    self._is_fitted = True
    return self
```

### Step 4 – `run_eta_sweep()` in `run_wp3.py`

```python
def run_eta_sweep(X_L_full, y_L_full, X_U, X_val, y_val,
                  eta_values, model_kwargs, out_dir, seed=42):
    rows = []
    for eta in eta_values:
        X_L, y_L = subsample_gold(X_L_full, y_L_full, eta, seed)
        model = GraphLabelPropagator(**model_kwargs)
        model.fit(X_L, y_L, X_U)
        res = evaluate_model(model, X_val, y_val)
        rows.append({"eta": eta, "macro_f1": res["macro_f1"],
                     "roc_auc": res["roc_auc"]})
        print(f"  η={eta:.2f} → F1={res['macro_f1']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "eta_sweep.csv", index=False)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(df["eta"], df["macro_f1"], marker="o")
    ax.set_xlabel("Gold fraction η"); ax.set_ylabel("Macro F1")
    ax.set_title(f"Label Propagation F1 vs η  ({model_kwargs})")
    fig.savefig(out_dir / "f1_vs_eta.png", dpi=150)
    plt.close(fig)
    return df
```

---

## Expected Output Files

```
outputs/wp3/
├── eta_sweep.csv          ← F1 for each eta × model variant
├── f1_vs_eta.png          ← main comparison plot
└── graph_analysis.csv     ← optional: graph connectivity stats
```

---

## Key Expected Findings (Proposal §5.3)

- F1 rises steeply as η increases from 2% → 20%, then levels off.
- PCA + Mahalanobis graph outperforms Raw + Euclidean, especially at small η.
- Larger k (more neighbors) helps when η is very small but can hurt precision at large k.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `MemoryError` building the full graph | Reduce k to 5 or use a random subsample of 5,000 training points |
| `spsolve` is very slow | Switch to `scipy.sparse.linalg.spsolve` with an ILU preconditioner, or use `sklearn.semi_supervised.LabelPropagation` as a quick check |
| All predictions collapse to one class | Check row normalisation in `harmonic_label_propagation` and ensure labeled rows have one-hot targets |
