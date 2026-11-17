# WP4 – Co-Training & Transductive SVMs
**Person:** Pratyush Somani (2024A7PS0514G)

---

## What This WP Does

WP4 implements the two margin/multi-view semi-supervised methods from Proposal §4.4–4.5.

- **Co-Training** – trains two classifiers on independent feature views; each one teaches the other using high-confidence unlabeled predictions.
- **Transductive SVM (TSVM)** – extends standard SVM to treat unlabeled labels as optimization variables, pushing the decision boundary into low-density regions.

Both methods **discard** the noisy machine-generated labels in D_U (like WP3).

---

## Files in This Folder

| File | Purpose |
|------|---------|
| `run_wp4.py` | **Entry point** |
| `cotraining.py` | Co-Training with two views (time-domain vs. frequency-domain) |
| `tsvm.py` | Transductive SVM with iterative label switching + C_U annealing |
| `README.md` | This file |

---

## Prerequisites

WP1 must have run:
```bash
cd wp1_data_pipeline
python run_wp1.py --data_path D:\ml-project\dataset --out_dir ..\outputs\wp1
```

---

## How to Run

```bash
cd wp4_cotraining_tsvm
python run_wp4.py \
    --features_path ..\outputs\wp1\features.npz \
    --labels_path   ..\outputs\wp1\labels.npz   \
    --out_dir       ..\outputs\wp4
```

---

## Implementation Guide

### Part A – Co-Training (`cotraining.py`)

**View definitions** (already in the file):
```python
VIEW1 = slice(0, 42)   # time-domain features
VIEW2 = slice(42, 84)  # frequency-domain features
```

**`CoTrainer.fit()` step by step:**

```python
def fit(self, X_L, y_L, X_U, y_U_noisy=None):
    # Split labeled data into views
    X_L1, X_L2 = X_L[:, VIEW1], X_L[:, VIEW2]
    X_U1, X_U2 = X_U[:, VIEW1].copy(), X_U[:, VIEW2].copy()

    # Initialize labeled pools
    L1_X, L1_y = X_L1.copy(), y_L.copy()
    L2_X, L2_y = X_L2.copy(), y_L.copy()

    # Unlabeled pool indices (we track which rows of X_U are still unused)
    remaining = np.arange(len(X_U))

    self._h1 = self._clf_fn()
    self._h2 = self._clf_fn()

    for _ in range(self.n_rounds):
        if len(remaining) == 0:
            break

        self._h1.fit(L1_X, L1_y)
        self._h2.fit(L2_X, L2_y)

        added = []

        # h1 → adds to L2
        proba1 = self._h1.predict_proba(X_U1[remaining])
        for c in range(5):
            conf = proba1[:, c]
            top_idx = np.argsort(conf)[-self.p:][::-1]
            top_idx = top_idx[conf[top_idx] > self.threshold]
            for local_i in top_idx:
                global_i = remaining[local_i]
                L2_X = np.vstack([L2_X, X_U2[global_i]])
                L2_y = np.append(L2_y, c)
                added.append(global_i)

        # h2 → adds to L1 (symmetric)
        proba2 = self._h2.predict_proba(X_U2[remaining])
        for c in range(5):
            conf = proba2[:, c]
            top_idx = np.argsort(conf)[-self.p:][::-1]
            top_idx = top_idx[conf[top_idx] > self.threshold]
            for local_i in top_idx:
                global_i = remaining[local_i]
                L1_X = np.vstack([L1_X, X_U1[global_i]])
                L1_y = np.append(L1_y, c)
                added.append(global_i)

        if len(added) == 0:
            break
        remaining = np.setdiff1d(remaining, added)

    # Final fit on full pools
    self._h1.fit(L1_X, L1_y)
    self._h2.fit(L2_X, L2_y)
    self._is_fitted = True
    return self
```

**`CoTrainer.predict_proba()`:**
```python
def predict_proba(self, X):
    p1 = self._h1.predict_proba(X[:, VIEW1])
    p2 = self._h2.predict_proba(X[:, VIEW2])
    return (p1 + p2) / 2
```

**Inter-view agreement rate** (record this in your sweep):
```python
pred1 = self._h1.predict(X_U[:, VIEW1])
pred2 = self._h2.predict(X_U[:, VIEW2])
agreement = (pred1 == pred2).mean()
```

---

### Part B – TSVM (`tsvm.py`)

The TSVM uses **C_U annealing**: start with a very small C_U and double it until C_U == C_U_max. This avoids getting stuck in poor local minima.

**`TSVMClassifier.fit()` step by step:**

```python
def fit(self, X_L, y_L, X_U, y_U_noisy=None):
    X_all = np.vstack([X_L, X_U])
    self._scaler.fit(X_all)
    X_L_s = self._scaler.transform(X_L)
    X_U_s = self._scaler.transform(X_U)

    # Step 1: Initial labels for X_U from standard SVM on D_L
    base_svm = SVC(C=self.C_L, kernel=self.kernel, probability=False)
    base_svm.fit(X_L_s, y_L)
    y_U_hat = base_svm.predict(X_U_s)

    C_U = self.C_U

    while C_U <= self.C_U_max:
        # Build training data: X_L (weight C_L) + X_U (weight C_U)
        X_train = np.vstack([X_L_s, X_U_s])
        y_train = np.hstack([y_L, y_U_hat])
        weights = np.hstack([
            np.full(len(X_L), self.C_L),
            np.full(len(X_U), C_U),
        ])

        # Fit SVM — sklearn's SVC accepts sample_weight
        clf = SVC(C=1.0, kernel=self.kernel, probability=False,
                  decision_function_shape="ovr")
        clf.fit(X_train, y_train, sample_weight=weights)

        # Switching: find beneficial label swaps in X_U
        changed = True
        for _ in range(self.max_iter):
            if not changed:
                break
            changed = False
            df = clf.decision_function(X_U_s)   # (n_U, C) scores
            # For each unlabeled point, check if the predicted class matches y_U_hat
            # If a swap of two unlabeled points (one above, one below margin)
            # improves the objective, perform the swap.
            # (Simplified: just re-assign any point where clf disagrees with y_U_hat
            #  and the confidence is high.)
            new_pred = clf.predict(X_U_s)
            diff_mask = new_pred != y_U_hat
            if diff_mask.any():
                y_U_hat[diff_mask] = new_pred[diff_mask]
                changed = True
                clf.fit(X_train, np.hstack([y_L, y_U_hat]),
                        sample_weight=weights)

        C_U = min(C_U * 2, self.C_U_max + 1e-9)

    self._clf = clf
    self._is_fitted = True
    return self
```

> **Note:** This is a simplified switching implementation. A full Joachims-style TSVM would
> search for pairs (i, j) with y_i ≠ y_j and both within the margin band and swap them.
> The simplified version above is sufficient for the project's analysis.

---

### Part C – Sweep functions in `run_wp4.py`

Both `run_eta_sweep_cotraining` and `run_eta_sweep_tsvm` follow the same pattern as WP3's `run_eta_sweep`:

```python
rows = []
for eta in eta_values:
    X_L, y_L = subsample_gold(X_L_full, y_L_full, eta, seed)
    model = CoTrainer(n_rounds=30, p=5, threshold=0.80)
    model.fit(X_L, y_L, X_U)
    res = evaluate_model(model, X_val, y_val)
    rows.append({"eta": eta, "macro_f1": res["macro_f1"]})
```

---

## Expected Output Files

```
outputs/wp4/
├── cotraining_eta_sweep.csv    ← macro F1 + agreement rate vs. eta
├── tsvm_eta_sweep.csv          ← macro F1 vs. eta for TSVM
├── f1_vs_eta_cotraining.png
├── f1_vs_eta_tsvm.png
└── view_agreement.png          ← h1/h2 agreement rate over co-training rounds
```

---

## Key Expected Findings (Proposal §5.3)

- Co-Training outperforms gold-only baseline at low η when views are sufficiently independent.
- As η grows, the gap narrows (more labeled data reduces the advantage of unlabeled exploitation).
- TSVM is more sensitive to class imbalance than Co-Training; support vector count grows when D_U contains many minority-class examples near the boundary.
- Inter-view agreement rate rises steadily across rounds (a sign the classifiers are converging).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Co-Training adds no samples | Lower `threshold` to 0.70 or increase `p` |
| TSVM is very slow on full dataset | Use `--max_records 3000` to test logic, then run full overnight |
| `predict_proba` not available on SVC | Make sure `probability=True` is set in the base SVC |
| Memory error | TSVM scales O(N²) with kernel SVM; consider linear kernel (`kernel='linear'`) for large datasets |
