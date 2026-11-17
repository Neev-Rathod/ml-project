# WP5 – Latent-Noise EM & Evaluation Suite
**Person:** Tanuj Yadav (2024AAPS0740G)

---

## What This WP Does

WP5 is the final integration point. It:

1. Implements the **Latent-Noise EM algorithm** (Proposal §4.6) – the method that *keeps* noisy labels and learns the annotator's confusion matrix.
2. Runs the **Discard vs. Correct comparison** (§5.3) – the main cross-WP experiment.
3. Produces the **Clinical Annotator Bias Audit** (§5.4) – the 5×5 confusion matrix heatmap showing which conditions PTB-XL's automated software confuses most.
4. Verifies **confusion matrix recovery** on synthetic data using Frobenius norm error.

---

## Files in This Folder

| File | Purpose |
|------|---------|
| `run_wp5.py` | **Entry point** |
| `latent_noise_em.py` | EM algorithm: E-step, M-step, convergence |
| `README.md` | This file |

---

## Prerequisites

WP1, WP2, WP3, and WP4 must be implemented (or at least WP1 for the PTB-XL experiments).

```bash
cd wp1_data_pipeline
python run_wp1.py --data_path D:\ml-project\dataset --out_dir ..\outputs\wp1
```

---

## How to Run

```bash
cd wp5_latent_noise_em
python run_wp5.py \
    --features_path ..\outputs\wp1\features.npz \
    --labels_path   ..\outputs\wp1\labels.npz   \
    --out_dir       ..\outputs\wp5
```

---

## Implementation Guide

### Part A – `LatentNoiseEM.fit()` in `latent_noise_em.py`

This is the core of WP5. The algorithm alternates between E-step and M-step.

#### Initialization

```python
C = self.n_classes
epsilon_init = 0.1

# Near-identity confusion matrix: strong diagonal, small off-diagonal
T = np.full((C, C), epsilon_init / (C - 1))
np.fill_diagonal(T, 1.0 - epsilon_init)
# T[c, :] should sum to 1
T = T / T.sum(axis=1, keepdims=True)
```

#### E-step (Eq. 4.4)

```python
# proba[j, c] = P(z=c | x_j; θ)  shape (n_U, C)
proba = clf.predict_proba(X_U_s)

# Compute responsibilities γ
# For each unlabeled j with observed label ỹ_j = y_U_noisy[j]:
gamma_U = np.zeros((n_U, C))
for j in range(n_U):
    k = y_U_noisy[j]        # observed noisy label
    for c in range(C):
        gamma_U[j, c] = T[c, k] * proba[j, c]
    total = gamma_U[j].sum()
    if total > 0:
        gamma_U[j] /= total
    else:
        gamma_U[j] = 1.0 / C  # uniform fallback
```

#### M-step (Eq. 4.5)

```python
# Update T
T_new = np.zeros((C, C))
for c in range(C):
    for k in range(C):
        mask_k = (y_U_noisy == k)
        T_new[c, k] = (self.alpha - 1) + gamma_U[mask_k, c].sum()
    T_new[c] /= T_new[c].sum()

# Update classifier with soft sample weights
# Hard assignment: take argmax of γ
y_U_hat    = np.argmax(gamma_U, axis=1)
w_U        = gamma_U.max(axis=1)        # confidence weight

y_gold_hot = np.eye(C)[y_L]             # (n_L, C)
gamma_L    = y_gold_hot                 # fixed for gold points

# Combine gold + unlabeled for weighted fit
X_fit = np.vstack([X_L_s, X_U_s])
y_fit = np.hstack([y_L, y_U_hat])
w_fit = np.hstack([np.ones(n_L), w_U])

clf = LogisticRegression(C=self.C_lr, max_iter=500, multi_class="multinomial")
clf.fit(X_fit, y_fit, sample_weight=w_fit)
```

#### Convergence check

```python
delta = np.linalg.norm(T_new - T, "fro")
T = T_new
if delta < self.tol:
    print(f"[EM] Converged at iteration {it+1}, ΔT={delta:.6f}")
    break
```

---

### Part B – `run_discard_vs_correct()` in `run_wp5.py`

This is the central cross-WP comparison. For each (η, ε) in the grid:

```python
# Import the other WPs' models
from wp3_graph_propagation.label_propagation import GraphLabelPropagator
from wp4_cotraining_tsvm.cotraining import CoTrainer

methods = {
    "EM":        LatentNoiseEM(n_classes=5),
    "GraphLP":   GraphLabelPropagator(k=10, n_pca=20),
    "CoTrain":   CoTrainer(n_rounds=30, p=5),
}

for name, model in methods.items():
    model.fit(X_L, y_L, X_U, y_U_noisy if name == "EM" else None)
    res = evaluate_model(model, X_val, y_val)
    # record (eta, epsilon, name, macro_f1)
```

Plot as a heatmap: rows = η, columns = ε, cell color = EM_F1 - GraphLP_F1.
Positive cells (blue) = EM wins; negative cells (red) = discard methods win.

---

### Part C – `run_clinical_audit()` in `run_wp5.py`

```python
import seaborn as sns, matplotlib.pyplot as plt

model = LatentNoiseEM(n_classes=5, max_iter=50)
model.fit(X_L, y_L, X_U, y_U_real_noisy)

T_hat = model.get_confusion_matrix()   # shape (5, 5)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    T_hat,
    annot=True, fmt=".2f",
    xticklabels=SUPERCLASSES,
    yticklabels=SUPERCLASSES,
    cmap="Blues", ax=ax,
    vmin=0, vmax=1,
)
ax.set_xlabel("Assigned label ỹ")
ax.set_ylabel("True label z")
ax.set_title("Learned Annotator Confusion Matrix T̂\n(PTB-XL automated reports)")
fig.tight_layout()
fig.savefig(out_dir / "T_hat_heatmap.png", dpi=150)

pd.DataFrame(T_hat, index=SUPERCLASSES, columns=SUPERCLASSES)\
  .to_csv(out_dir / "T_hat.csv")
```

---

### Part D – `run_frobenius_recovery()` in `run_wp5.py`

```python
from shared.metrics import frobenius_error
from wp2_synthetic_baselines.synthetic_generator import (
    generate_manifold, inject_asymmetric_noise
)

# Define a known T_true
T_true = np.eye(5) * 0.70 + np.ones((5,5)) * 0.075 - np.eye(5) * 0.075

n_U_values = [100, 300, 500, 1000, 2000, 5000]
errors = []
for n_U in n_U_values:
    X, y = generate_manifold(n_samples=n_U + 500, seed=seed)
    X_U_syn, y_U_clean = X[500:500+n_U], y[500:500+n_U]
    y_U_noisy = inject_asymmetric_noise(y_U_clean, T_true, seed=seed)
    model = LatentNoiseEM(n_classes=5, max_iter=50)
    model.fit(X[:500], y[:500], X_U_syn, y_U_noisy)
    errors.append(frobenius_error(model.T_hat_, T_true))
    print(f"n_U={n_U}: ||T̂ - T_true||_F = {errors[-1]:.4f}")

# Plot
plt.plot(n_U_values, errors, marker="o")
plt.xlabel("Unlabeled pool size n_U")
plt.ylabel("Frobenius error")
plt.title("T̂ Recovery Error vs. n_U")
plt.savefig(out_dir / "frobenius_recovery.png", dpi=150)
```

---

## Expected Output Files

```
outputs/wp5/
├── discard_vs_correct.csv       ← F1 for all (η, ε, method) combinations
├── discard_vs_correct_heatmap.png
├── T_hat_heatmap.png            ← clinical annotator confusion matrix
├── T_hat.csv
└── frobenius_recovery.png       ← T̂ recovery error vs. n_U
```

---

## Key Expected Findings (Proposal §5.3–5.4)

- At **low ε** (clean noisy labels): EM ≈ GraphLP/CoTrain (not much noise to correct).
- At **high ε and high η**: EM wins because it uses the extra signal in noisy labels.
- At **high ε and very low η**: GraphLP/CoTrain may win because EM has too few anchor points.
- Clinical audit: T̂ should show the automated software most often confuses **MI ↔ STTC** and **NORM ↔ HYP** (consistent with known ECG interpretation difficulty).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| EM does not converge | Increase `max_iter`; check that T rows sum to 1 at each step |
| T̂ collapses to identity | Reduce `alpha` to 1.01 (less Dirichlet smoothing) |
| Logistic regression with `sample_weight` converges poorly | Increase `max_iter` to 2000; normalise weights to sum to 1 |
| `frobenius_error` is flat | Check that `inject_asymmetric_noise` is actually introducing off-diagonal noise |
