# WP3 – Graph Label Propagation
**Person:** Pratyush Somani (2024A7PS0514G) · **Course:** BITS F464 Machine Learning 
## 1. Summary

WP3 implements Proposal §4.3: the noisy labels in the Untrusted Pool D_U are **discarded**, and clean Gold Set labels are
spread over a kNN graph built from feature geometry.

Main results (validation macro F1, 3 seeds; details in §6–§8):

- Graph label propagation (LP) reaches **0.25 → 0.34 macro F1** as the labeled fraction η goes from 2% to 50% of the Gold Set
  (Euclidean, PCA = 5, k = 3); the best variant per η reaches 0.25 → 0.37.
- **LP is well below simple supervised baselines trained on the Gold Set alone**, on both the validation and test folds (§8b). Balanced LogReg and RBF-SVM beat it by about 0.08–0.12 macro F1 at every η (validation, against the best LP variant per η), and an **unweighted** LogReg still beats it by 0.04–0.08, so class weighting does not explain the gap. LP only beats raw-feature 5-NN, and Gaussian NB at η = 50%.
- A gold-only kNN on the **same PCA embedding** as LP is better than LP on the test fold at every η ≤ 20% and on validation at η = 2%, 5% and 20%; the two are roughly tied at η = 50% (§8).
- Small k and a low-dimensional PCA embedding work best. Mahalanobis is tied with Euclidean only at PCA = 5 and is clearly worse otherwise. Class-mass normalisation made no meaningful difference. Adding the unused Gold records as extra unlabeled points hurt (§7.5).
- The graph is essentially connected, so connectivity is not the bottleneck.
- Test-fold (fold 10) macro F1 for the LP configuration above: **0.258 (η = 2%) … 0.352 (η = 50%)** (§8).

## 2. Files

| File | Purpose |
|------|---------|
| `label_propagation.py` | `build_knn_graph`, `harmonic_label_propagation`, `GraphLabelPropagator` |
| `run_wp3.py` | Sweep over distance × PCA × k × η × seeds, evaluated on the **validation fold only** |
| `make_report_files.py` | Builds `eta_sweep.csv`, `graph_analysis.csv`, `f1_vs_eta.png` from the sweep output |
| `eval_final.py` | Gold-only kNN control on the LP embedding (`--split val`) and final held-out evaluation (`--split test`) |
| `gold_baselines.py` | Gold-only supervised baselines (balanced and unweighted LogReg, SVM-RBF, Gaussian NB, 5-NN) on the same η grid, seeds and subsamples; `--split val` or `--split test` |
| `plot_vs_baselines.py` | Figure comparing LP with the gold-only baselines |

Outputs are in `outputs/wp3/`:

| File | Content |
|------|---------|
| `wp3_results_raw.csv` | One row per fit: config, η, seed, macro F1, ROC-AUC, PR-AUC, per-class F1, graph components, unreachable nodes, fit time |
| `wp3_results_summary.csv` | Mean/std over seeds per (distance, PCA, k, η) |
| `wp3_summary.png` | F1 vs η, sensitivity to k, sensitivity to PCA dims |
| `eta_sweep.csv`, `f1_vs_eta.png`, `graph_analysis.csv` | Files requested by the original WP3 guide (derived from the CSVs above) |
| `control_val*.{csv,png}`, `control_test*.{csv,png}` | LP vs gold-only kNN control, validation and test folds |
| `gold_baselines.csv`, `gold_baselines_test.csv` | Baseline results per model, η and seed with per-class F1 (validation / test fold) |
| `wp3_vs_baselines.png` | LP vs gold-only baselines |
| `run.log` | Console log of the main sweep |
| `ablations/nocmn/` | CMN-off sweep (§7.3) |
| `ablations/unusedgold/` | Unused-Gold ablation, 3 seeds (§7.5) |

## 3. Environment setup (WSL Ubuntu, Python 3.12)

The project was first cloned under `/mnt/c/...`, where `python3 -m venv` hung (very slow file I/O on the Windows
drive). It was moved to the Linux filesystem:

```bash
sudo apt update && sudo apt install python3.12-venv -y
cd ~
git clone https://github.com/Neev-Rathod/semi-supervised-ecg.git
cd semi-supervised-ecg
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

(`pip` alone failed with "externally-managed-environment" until the venv was recreated and `python -m pip` was used.)

Input data: `outputs/wp1/features.npz` (X: 21,837 × 84) and `outputs/wp1/labels.npz` (y, fold, gold_mask, ecg_ids), produced by WP1.
After `load_standard_split`: Gold Set D_L = 9,010, Untrusted Pool D_U = 3,968, Validation (fold 9) = 1,642, Test (fold 10) = 1,652.
Classes: 0 NORM, 1 MI, 2 STTC, 3 CD, 4 HYP.

## 4. Method as implemented

1. **Preprocessing** – `StandardScaler` fit on features of L ∪ U (no labels used), features clipped to ±5, then PCA (`n_pca` components, 0 = none) fit on L ∪ U.
2. **Distance** –
   - `euclidean`: plain Euclidean on the PCA features.
   - `cosine`: kNN search with cosine distance.
   - `mahalanobis`: Eq. 4.1, using Σ̂_L + λI (covariance of the labeled points only, λ = 0.01 × mean variance), implemented by Cholesky whitening, after which Euclidean distance equals Mahalanobis distance.
3. **Graph** – kNN graph (`k` neighbours per node) with locally scaled weights W_ij = exp(−d²/(σ_i σ_j)), σ_i = distance to the 7th nearest neighbour (`local_scale_k = 7`), symmetrised with `max(W, Wᵀ)`.
4. **Harmonic propagation** (Eq. 4.2) – F_U = (D_UU − W_UU + 1e-6·I)⁻¹ W_UL Y_L, solved with a sparse LU factorisation. Negative values are clipped, rows renormalised. Unlabeled nodes with no path to any labeled node get the smoothed class prior.
5. **Prediction for new points** (validation/test records are not graph nodes) – kernel-weighted average of the F rows of the k nearest graph nodes, using the same local scaling. (The original stub suggested copying the single nearest node's row; this is smoother.)
6. **Optional** – `--cmn`: class-mass normalisation (Zhu et al., 2003). `--use_unused_gold`: adds the Gold records not sampled at a given η to the unlabeled pool with labels hidden.
7. **η subsampling** – stratified per class (fraction η of each class, ≥ 1 per class), so η = 2% gives ≈ 180 labeled records. The noisy labels `y_U_noisy` are never passed to the model.
8. **Diagnostics recorded per fit** – number of graph connected components and number of unlabeled nodes unreachable from any labeled node.

Deviations from the original WP3 guide: vectorised graph construction instead of a Python double loop; Σ_L-based Mahalanobis (Eq. 4.1) instead of the covariance of all data; stratified η subsampling; out-of-sample kernel-weighted prediction; features clipped; PCA and scaler fit on L ∪ U features.

**Gold-only baselines (`gold_baselines.py`)** – each is a pipeline of `StandardScaler` plus one of: LogisticRegression (`class_weight="balanced"`, max_iter 3000), the same LogisticRegression without class weights, SVC (RBF, `class_weight="balanced"`, `probability=True`), GaussianNB, KNeighborsClassifier (k = 5). All use all 84 features, default hyperparameters (no tuning), and the identical stratified subsamples as the LP runs (same seeds 42–44). They never see D_U.

## 5. Chronological log of every run

All commands are run from the repo root with the venv active. Seeds are `42, 43, 44` for 3-seed runs and `42` for 1-seed runs. Arguments listed are those beyond `--features_path outputs/wp1/features.npz --labels_path outputs/wp1/labels.npz`.

| # | Purpose | Output | Arguments | Fits | Status |
|---|---------|--------|-----------|------|--------|
| 1 | Smoke test | `outputs/wp3_smoke` | `--k_values 10 --pca_components 20 --distances euclidean --eta_values 0.1 --n_seeds 1` | 1 | Pipeline works. Macro F1 = 0.2636. Not kept. |
| 2 | **Main sweep** | `outputs/wp3` | `--k_values 3,5,10,20 --pca_components 0,5,10,20,40 --eta_values 0.02,0.05,0.10,0.20,0.50 --distances euclidean,cosine,mahalanobis --n_seeds 3 --cmn` | 3×5×4×5×3 = 900 | Kept. Log in `run.log`. |
| 3 | CMN ablation (CMN **off**) | `outputs/wp3/ablations/nocmn` | `--k_values 5,10 --pca_components 10,20 --distances euclidean --n_seeds 3` (default η grid) | 4×5×3 = 60 | Kept. §7.3. |
| 4 | Exploratory: unused Gold as extra unlabeled data | `outputs/wp3_unusedgold` | `--k_values 5 --pca_components 10 --distances euclidean --eta_values 0.05,0.10,0.20 --n_seeds 1 --cmn --use_unused_gold` | 3 | Mean 0.2707 (seed 42) vs 0.3077 for the matched seed-42 runs of run 2 (0.287 / 0.310 / 0.326); this run: 0.239 / 0.258 / 0.315, i.e. worse at all three η in this seed. Superseded by run 10; folder not kept. |
| 5 | Report files | `outputs/wp3` | `python wp3_graph_propagation/make_report_files.py outputs/wp3` | – | Created `eta_sweep.csv`, `graph_analysis.csv`, `f1_vs_eta.png`. |
| 6 | Control on validation | `outputs/wp3` | `python wp3_graph_propagation/eval_final.py --split val` (Euclidean, PCA 5, k 3, 3 seeds, no CMN) | 15 | Kept. §8. |
| 7 | Final evaluation on test fold 10 | `outputs/wp3` | `python wp3_graph_propagation/eval_final.py --split test` (config frozen from validation) | 15 | Kept. §8. |
| 8 | Gold-only supervised baselines (validation) | `outputs/wp3` | `python wp3_graph_propagation/gold_baselines.py --split val --out_dir outputs/wp3` (5 models incl. unweighted LogReg, η grid 0.02–0.50, 3 seeds) | 5×5×3 = 75 | Kept. `gold_baselines.csv` incl. per-class F1. First run had 4 models (no unweighted LogReg); re-run with the extended script reproduced the original four exactly. §8b. |
| 9 | Baseline comparison figure | `outputs/wp3` | `python wp3_graph_propagation/plot_vs_baselines.py --out_dir outputs/wp3` | – | Kept. `wp3_vs_baselines.png` (legend moved, tick labels fixed, PCA 5 / k 3 LP curve and unweighted LogReg added). |
| 10 | Unused-Gold ablation, 3 seeds | `outputs/wp3/ablations/unusedgold` | `--k_values 5 --pca_components 10 --distances euclidean --eta_values 0.05,0.10,0.20 --n_seeds 3 --cmn --use_unused_gold` | 9 | Kept. Mean 0.2667 vs 0.3003 for the same config without unused Gold. §7.5. |
| 11 | Gold-only baselines on the test fold | `outputs/wp3` | `python wp3_graph_propagation/gold_baselines.py --split test --out_dir outputs/wp3` | 75 | Kept. `gold_baselines_test.csv`. §8b. |

Run 2 top configurations by mean macro F1 (validation, mean over η and seeds):

| Distance | PCA | k | Macro F1 |
|---|---|---|---|
| euclidean | 5 | 3 | 0.3054 |
| mahalanobis | 5 | 3 | 0.3042 |
| euclidean | 10 | 3 | 0.3042 |
| euclidean | 5 | 5 | 0.3036 |
| cosine | 40 | 3 | 0.3012 |

These differ by ≈ 0.004 and are within seed noise (see `macro_f1_std`), so Euclidean / PCA 5 / k 3 for the follow-up runs is a convenient pick, not a proven winner.

Run 3 top configurations (no CMN, mean over η and seeds): PCA 10 / k 5 = 0.2943, PCA 20 / k 5 = 0.2848, PCA 10 / k 10 = 0.2731, PCA 20 / k 10 = 0.2634.

## 6. Sweep results (run 2, validation macro F1, mean of 3 seeds)

Best (PCA, k) chosen per variant by mean F1 (exact choices in the legend of `f1_vs_eta.png`):

| η | Raw + Euclidean | PCA + Euclidean | PCA + Cosine | PCA + Mahalanobis |
|---|---|---|---|---|
| 2%  | 0.216 | 0.246 | 0.223 | 0.254 |
| 5%  | 0.293 | 0.310 | 0.276 | 0.301 |
| 10% | 0.283 | 0.312 | 0.297 | 0.302 |
| 20% | 0.316 | 0.318 | 0.338 | 0.329 |
| 50% | 0.361 | 0.341 | 0.373 | 0.334 |

Mean macro F1 by distance and PCA dimension (averaged over k, η, seeds):

| Distance | PCA 0 | PCA 5 | PCA 10 | PCA 20 | PCA 40 |
|---|---|---|---|---|---|
| cosine | 0.273 | 0.272 | 0.273 | 0.272 | 0.273 |
| euclidean | 0.266 | 0.294 | 0.281 | 0.271 | 0.267 |
| mahalanobis | 0.211 | 0.290 | 0.259 | 0.238 | 0.224 |

By distance and k (averaged over PCA, η, seeds):

| Distance | k = 3 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|
| cosine | 0.295 | 0.283 | 0.266 | 0.246 |
| euclidean | 0.299 | 0.289 | 0.270 | 0.245 |
| mahalanobis | 0.273 | 0.258 | 0.234 | 0.213 |

Effect of k at η = 2% (mean over distances and PCA settings): k = 3 → 0.219, k = 5 → 0.207, k = 10 → 0.188, k = 20 → 0.174.

Per-class F1 and graph diagnostics for Euclidean, PCA = 5, k = 3 (run 2, with CMN, mean of 3 seeds):

| η | Macro F1 | Unreachable nodes | Graph components | F1 NORM | F1 MI | F1 STTC | F1 CD | F1 HYP |
|---|---|---|---|---|---|---|---|---|
| 2%  | 0.246 | 2.7 | 1.3 | 0.716 | 0.095 | 0.068 | 0.331 | 0.018 |
| 5%  | 0.310 | 4.3 | 1.7 | 0.716 | 0.158 | 0.085 | 0.378 | 0.212 |
| 10% | 0.312 | 5.3 | 2.0 | 0.720 | 0.181 | 0.106 | 0.375 | 0.179 |
| 20% | 0.318 | 2.7 | 1.3 | 0.710 | 0.200 | 0.126 | 0.351 | 0.206 |
| 50% | 0.341 | 0.0 | 1.7 | 0.716 | 0.238 | 0.147 | 0.392 | 0.212 |

## 7. Findings

### 7.1 Label scarcity and class structure
Macro F1 rises with η (largest jump 2% → 5%) and is still rising at 50%. NORM is stable at ≈ 0.72 F1; MI, STTC and HYP are weak
(HYP ≈ 0.02 at η = 2%, where only ≈ 6 HYP records are labeled; STTC ≤ 0.15 even at η = 50%). Overall F1 of 0.25–0.34 is consistent with the
heavy class overlap seen in WP1's k-Means analysis (purity ≈ 0.58).

### 7.2 Distance metric, PCA and k
- PCA-based variants lead at small η (≤ 10%). At η = 20% cosine is highest (0.338) and raw and PCA Euclidean tie (0.316 vs 0.318); at η = 50% cosine (0.373) and raw Euclidean (0.361) lead PCA Euclidean (0.341). Several of these gaps are within seed noise; only the small-η gaps (≈ 3–4 pp) are likely real.
- **Mahalanobis is tied with Euclidean only at low PCA dimension** (0.290 vs 0.294 at PCA = 5). It is clearly worse on raw features (0.211 vs 0.266), at PCA 40 (0.224 vs 0.267), and at every k (e.g. 0.273 vs 0.299 at k = 3; 0.213 vs 0.245 at k = 20). A possible cause, not tested, is that Σ̂_L estimated from ≈ 180–4,500 labeled points is noisy in higher dimensions. WP1 recommended Mahalanobis for graph construction; WP1's advantage was measured as ARI of k-Means clustering, not as classification.
- Cosine is insensitive to PCA dimension (0.272–0.273 for all settings).
- **Small k is best** (§6). This contradicts the original guide's expectation that a larger k helps at very small η.
- The best configurations sit at the edge of the grid (k = 3, PCA = 5); smaller values were not explored.

### 7.3 Class-mass normalisation
Paired comparison of run 2 (CMN on) with run 3 (CMN off) over 60 matched runs: mean ΔF1 (CMN − no CMN) = **+0.0014**.
By η: 2% +0.0022, 5% +0.0088, 10% +0.0017, 20% −0.0021, 50% −0.0033. Mixed signs and tiny size: CMN makes no meaningful difference.

### 7.4 Graph connectivity
With k = 3 the graph is nearly connected: 1–2 components and only 0–5 unreachable unlabeled nodes out of ≈ 4,000 (§6). Connectivity is therefore not what limits performance; class overlap in the feature space is.

### 7.5 Does extra unlabeled data help?
- **Unused Gold as additional unlabeled points** (run 10, Euclidean / PCA 10 / k 5, η ∈ {5%, 10%, 20%}, 3 seeds): mean F1 **0.2667 with unused Gold vs 0.3003 without** (matched configuration from run 2). In the single-seed run 4 the difference was negative at all three η (seed 42: 0.239 vs 0.287, 0.258 vs 0.310, 0.315 vs 0.326). Adding ≈ 7,200–8,600 unlabeled points (the unused Gold records, depending on η) made LP worse. A possible explanation, untested: the extra points are ≈ 62% NORM, so propagated labels become even more NORM-heavy.
- **Gold-only kNN control** on the same embedding (§8): no consistent benefit from the pool.
- Other possible explanations (**untested hypotheses**): the pool's label distribution differs from the Gold Set (WP1: MI is 28% of D_U vs 10% of D_L); with strongly overlapping classes propagated labels mostly spread the majority class; the cluster assumption is weak for these hand-crafted features.

### 7.6 Comparison with simple supervised baselines
See §8b. Balanced LogReg, unweighted LogReg and RBF-SVM trained on the labeled Gold records alone beat LP at every η on both the validation and test folds; Gaussian NB beats LP at η ≤ 20%. The gap persists without class weighting and when LP uses all 84 features (no PCA).

## 8. Control on the LP embedding and final evaluation (runs 6–7)

Setting: Euclidean, PCA = 5, k = 3, no CMN, 3 seeds (42–44), same stratified labeled subsets for both methods.
The control is a distance-weighted kNN trained on the labeled points only, using the same scaler/PCA embedding and the same k, so it differs from LP in not using the unlabeled pool (and in its prediction rule).
Values are mean (std) macro F1; gain = LP − control.

**Validation (fold 9)**

| η | Graph LP | Gold-only kNN | Gain |
|---|---|---|---|
| 2%  | 0.2473 (0.0063) | 0.2830 (0.0174) | −0.0357 |
| 5%  | 0.2957 (0.0103) | 0.3183 (0.0151) | −0.0226 |
| 10% | 0.3127 (0.0084) | 0.3007 (0.0039) | +0.0120 |
| 20% | 0.3179 (0.0117) | 0.3255 (0.0039) | −0.0076 |
| 50% | 0.3389 (0.0170) | 0.3286 (0.0135) | +0.0104 |

**Test (fold 10, human-validated records)**

| η | Graph LP | Gold-only kNN | Gain |
|---|---|---|---|
| 2%  | 0.2584 (0.0142) | 0.3062 (0.0115) | −0.0478 |
| 5%  | 0.3055 (0.0244) | 0.3187 (0.0027) | −0.0131 |
| 10% | 0.3090 (0.0078) | 0.3276 (0.0100) | −0.0186 |
| 20% | 0.3149 (0.0057) | 0.3249 (0.0122) | −0.0100 |
| 50% | 0.3517 (0.0157) | 0.3483 (0.0096) | +0.0033 |

The configuration was selected on validation data only. The test fold was touched only by `eval_final.py --split test` and `gold_baselines.py --split test` (run 11); neither was used for any selection.
(Validation values for LP differ slightly from §6 because the sweep used `--cmn` and run 6 does not.)
Only the η = 2% gaps clearly exceed the seed std; the sign of the gain at η = 10% flips between validation and test.

### 8b. Gold-only supervised baselines (runs 8 and 11; mean macro F1 over seeds 42–44)

Same stratified labeled subsets as the LP runs; baselines use all 84 standardized features, default hyperparameters (`gold_baselines.py`).
LP column = Euclidean, PCA 5, k 3 (validation: run 2 with CMN; test: run 7, no CMN).

**Validation (fold 9)**

| η | LogReg (bal.) | LogReg (unweighted) | SVM-RBF (bal.) | Gaussian NB | 5-NN | Graph LP |
|---|---|---|---|---|---|---|
| 2%  | 0.352 | 0.333 | 0.344 | 0.326 | 0.213 | 0.246 |
| 5%  | 0.388 | 0.391 | 0.400 | 0.358 | 0.253 | 0.310 |
| 10% | 0.411 | 0.385 | 0.429 | 0.379 | 0.264 | 0.312 |
| 20% | 0.458 | 0.422 | 0.459 | 0.374 | 0.274 | 0.318 |
| 50% | 0.480 | 0.417 | 0.489 | 0.324 | 0.314 | 0.341 |

**Test (fold 10)**

| η | LogReg (bal.) | LogReg (unweighted) | SVM-RBF (bal.) | Gaussian NB | 5-NN | Graph LP |
|---|---|---|---|---|---|---|
| 2%  | 0.335 | 0.341 | 0.352 | 0.340 | 0.219 | 0.258 |
| 5%  | 0.388 | 0.393 | 0.405 | 0.366 | 0.265 | 0.306 |
| 10% | 0.407 | 0.408 | 0.416 | 0.384 | 0.276 | 0.309 |
| 20% | 0.434 | 0.428 | 0.441 | 0.386 | 0.285 | 0.315 |
| 50% | 0.457 | 0.429 | 0.471 | 0.325 | 0.311 | 0.352 |

- **Size of the gap.** On validation, balanced LogReg/SVM beat the best LP variant per η (§6) by about 0.08–0.12; on test they beat the Euclidean/PCA 5/k 3 LP configuration by about 0.08–0.13 (SVM 0.094–0.126, balanced LogReg 0.077–0.119). The LP numbers come from the best of a large grid selected on validation while the baselines use untuned defaults, so the gap is not inflated by tuning.
- **Class weighting does not explain the gap.** Unweighted LogReg is still 0.04–0.08 above the best LP variant per η on validation (0.08–0.11 above the Euclidean/PCA 5/k 3 configuration on test). Balanced and unweighted LogReg are close at η ≤ 10%; balanced pulls ahead at η = 20–50% (validation 0.458 vs 0.422 and 0.480 vs 0.417; test 0.434 vs 0.428 and 0.457 vs 0.429).
- **Feature dimension does not explain the gap either.** LP with no PCA (all 84 features, §6: best k per η) scores 0.216 / 0.293 / 0.283 / 0.316 / 0.361 on validation, still 0.06–0.12 below unweighted LogReg. (LP clips standardised features at ±5; the baselines do not.)
- LP beats raw-feature 5-NN at every η and Gaussian NB at η = 50%. The kNN control in §8 (kNN on the PCA embedding) beats LP at small η, which suggests that the embedding matters more than the propagation step (the control also differs from raw 5-NN in k and distance weighting).
- Gaussian NB improves up to η ≈ 10–20% and then falls (validation 0.379 → 0.324 at 50%; test 0.386 → 0.325), while LogReg and SVM keep improving. A possible reading, not tested here, is the generative-vs-discriminative asymptotic-error trade-off (Ng & Jordan).

**Per-class F1 at η = 10% (validation, mean of 3 seeds; LP from run 2 with CMN)**

| Model | NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|---|
| LogReg (balanced) | 0.664 | 0.351 | 0.358 | 0.409 | 0.271 |
| LogReg (unweighted) | 0.775 | 0.260 | 0.262 | 0.427 | 0.199 |
| SVM-RBF (balanced) | 0.683 | 0.361 | 0.318 | 0.429 | 0.356 |
| Gaussian NB | 0.723 | 0.257 | 0.190 | 0.404 | 0.321 |
| 5-NN | 0.740 | 0.096 | 0.073 | 0.389 | 0.020 |
| Graph LP (Euc, PCA 5, k 3) | 0.720 | 0.181 | 0.106 | 0.375 | 0.179 |

Unweighted LogReg beats LP on all five classes, including NORM (0.775 vs 0.720), so the gap is not a minority-versus-majority trade-off; it is largest on STTC (0.106 vs 0.262) and MI (0.181 vs 0.260). Class weighting moves F1 from NORM to MI and STTC (LogReg NORM 0.775 → 0.664). 5-NN is near zero on MI, STTC and HYP; LP recovers part of this. Test-fold per-class values for the baselines are in `gold_baselines_test.csv` and show the same pattern; per-class F1 for LP was recorded only on validation. HYP numbers are noisy (≈ 30 labeled HYP records at η = 10%).

Figure: `wp3_vs_baselines.png` (validation).

## 9. Limitations

- Only 3 seeds; several between-metric and between-method differences are within noise.
- Best LP configurations are at the edge of the grid (k = 3, PCA = 5); smaller k / PCA not tested.
- Model selection used a single validation fold; η = 100% of the Gold Set was not run.
- The kNN control uses a different prediction rule from LP, so that gap is not purely "unlabeled data used vs not".
- Baseline comparison: the controls for class weighting (unweighted LogReg) and feature dimension (LP without PCA) show the gap persists, but they are not exact (LP clips features at ±5, baselines do not; baselines use default hyperparameters; no class-prior-corrected LP was run, so LP has no counterpart to the balanced baselines' minority-class boost).
- Per-class F1 for LP was recorded only on the validation fold (run 2, CMN on); baselines were not tuned.
- All results use WP1's hand-crafted 84-d features; classes overlap strongly in this space.

## 10. Reproducing everything

```bash
mkdir -p outputs/wp3
python wp3_graph_propagation/run_wp3.py \
  --features_path outputs/wp1/features.npz --labels_path outputs/wp1/labels.npz \
  --out_dir outputs/wp3 \
  --k_values 3,5,10,20 --pca_components 0,5,10,20,40 \
  --eta_values 0.02,0.05,0.10,0.20,0.50 \
  --distances euclidean,cosine,mahalanobis --n_seeds 3 --cmn 2>&1 | tee outputs/wp3/run.log

python wp3_graph_propagation/run_wp3.py \
  --features_path outputs/wp1/features.npz --labels_path outputs/wp1/labels.npz \
  --out_dir outputs/wp3/ablations/nocmn \
  --k_values 5,10 --pca_components 10,20 --distances euclidean --n_seeds 3

python wp3_graph_propagation/run_wp3.py \
  --features_path outputs/wp1/features.npz --labels_path outputs/wp1/labels.npz \
  --out_dir outputs/wp3/ablations/unusedgold \
  --k_values 5 --pca_components 10 --distances euclidean \
  --eta_values 0.05,0.10,0.20 --n_seeds 3 --cmn --use_unused_gold

python wp3_graph_propagation/make_report_files.py outputs/wp3
python wp3_graph_propagation/eval_final.py --split val
python wp3_graph_propagation/eval_final.py --split test

python wp3_graph_propagation/gold_baselines.py \
  --features_path outputs/wp1/features.npz --labels_path outputs/wp1/labels.npz --out_dir outputs/wp3 --split val
python wp3_graph_propagation/gold_baselines.py \
  --features_path outputs/wp1/features.npz --labels_path outputs/wp1/labels.npz --out_dir outputs/wp3 --split test
python wp3_graph_propagation/plot_vs_baselines.py --out_dir outputs/wp3
```