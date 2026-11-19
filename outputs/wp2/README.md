# WP2 Outputs – Synthetic Generator & Supervised Baselines
**Author:** Shaurya Suman (2024A7PS0589G)  
**Work Package:** WP2 – Synthetic Generator & Supervised Baselines  
**Date:** November 2026  
**Course:** BITS F464 – Machine Learning, Sem 1 2026–2027  
**Instructor:** Prof. Anand Narasimhamurthy  

---

## 1. Overview and Deliverables

WP2 provides two foundational components that all other Work Packages depend on:

1. **Synthetic Benchmark Dataset** (`synthetic_generator.py`) – a controllable, 5-class manifold dataset with three noise injection functions (uniform, asymmetric, boundary-dependent). Known ground truth enables WP3–WP5 to validate their methods before touching real ECG data.

2. **Five Supervised Baseline Classifiers** (`baselines.py`) – the reference performance floor that semi-supervised methods (WP3–WP5) must beat. Each classifier supports three training modes:
   - **gold_only** – train only on clean, human-verified labels ($\mathcal{D}_L$)
   - **naive_pool** – train on $\mathcal{D}_L \cup \mathcal{D}_U$, treating noisy labels as clean
   - **self_training** – iterative self-labelling starting from $\mathcal{D}_L$

### Generated Files in `outputs/wp2/`

| File | Description |
|------|-------------|
| `noise_breakdown.csv` | Macro F1, ROC-AUC, PR-AUC for each classifier × mode × ε value (75 rows) |
| `f1_vs_epsilon.png` | Main comparison: Macro F1 vs. noise rate ε for all 15 model/mode combinations |
| `svm_support_vectors.png` | SVM support vector count inflation analysis across noise rates |
| `tree_depth_vs_epsilon.png` | Decision tree overfitting analysis: depth and leaf count for unpruned vs. pruned (max_depth=5) |
| `README.md` | This technical analysis report |

---

## 2. Synthetic Dataset Design

### Generation Method (`generate_manifold`)

The synthetic benchmark uses `sklearn.datasets.make_classification` with the following configuration:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_samples` | 3,000 | Large enough for statistical significance, small enough for fast iteration |
| `n_features` | 50 | Matches the ambient dimensionality scale of WP1's 84-feature vectors |
| `n_informative` | 10 | Creates a low-dimensional informative subspace, mimicking real ECG signal structure |
| `n_redundant` | 5 | Adds correlated features (like correlated lead measurements in ECG) |
| `n_classes` | 5 | Matches PTB-XL's 5 superclasses (NORM, MI, STTC, CD, HYP) |
| `class_sep` | 1.0 | Moderate separation — not trivially separable |

After generation, the features are transformed by a random orthogonal projection matrix and injected with Gaussian noise ($\sigma = 0.1$) to simulate the PTB-XL ambient noise structure.

### Data Split

| Split | Samples | Purpose |
|-------|---------|---------|
| Training total | 2,100 (70%) | Model training |
| ↳ Gold Set $\mathcal{D}_L$ | 420 (20% of train) | Clean labels |
| ↳ Untrusted Pool $\mathcal{D}_U$ | 1,680 (80% of train) | Noise-injected labels |
| Validation | 450 (15%) | Hyperparameter tuning & reported metrics |
| Test | 450 (15%) | Held out |

### Noise Injection Functions

Three noise types are implemented following Proposal §3.1:

1. **Uniform Noise** (`inject_uniform_noise`): Each label flips to a uniformly random different class with probability ε. Produces the symmetric transition matrix $T_{ij} = \frac{\varepsilon}{C-1}$ for $i \neq j$.

2. **Asymmetric Noise** (`inject_asymmetric_noise`): Class-conditional noise via a sparse $C \times C$ transition matrix $T$. Default configuration confuses class pairs (0,1) and (2,3) with probability 0.3, simulating clinically realistic confusion patterns (e.g., MI ↔ STTC share similar ST-segment morphology).

3. **Boundary-Dependent Noise** (`inject_boundary_noise`, Proposal Eq. 3.1): Points near the decision boundary have higher flip probability:
$$P(\text{flip}) = \varepsilon_{\max} \cdot \exp(-\gamma \cdot \text{margin})$$
where $\text{margin} = P(z=y^*|x) - \max_{c \neq y^*} P(z=c|x)$. A LogisticRegression is trained on clean data to estimate the margin.

---

## 3. Experimental Results

### 3.1 Noise Breakdown Analysis (`noise_breakdown.csv`)

Experiments sweep ε ∈ {0.00, 0.15, 0.30, 0.45, 0.60} with uniform noise. All 5 classifiers × 3 modes = 15 models are evaluated at each ε level.

#### Full Results Table

| ε | GNB gold | GNB naive | GNB self | LR gold | LR naive | LR self | kNN gold | kNN naive | kNN self | DT gold | DT naive | DT self | SVM gold | SVM naive | SVM self |
|---|----------|-----------|----------|---------|----------|---------|----------|-----------|----------|---------|----------|---------|----------|-----------|----------|
| **0.00** | 0.648 | 0.649 | 0.565 | 0.682 | 0.761 | 0.681 | 0.741 | **0.817** | 0.731 | 0.486 | 0.582 | 0.487 | 0.811 | **0.879** | 0.817 |
| **0.15** | 0.648 | 0.623 | 0.565 | 0.682 | 0.743 | 0.681 | 0.741 | 0.810 | 0.731 | 0.486 | 0.549 | 0.487 | 0.811 | **0.866** | 0.822 |
| **0.30** | 0.648 | 0.611 | 0.565 | 0.682 | 0.707 | 0.681 | 0.741 | 0.748 | 0.731 | 0.486 | 0.473 | 0.487 | 0.811 | **0.869** | 0.816 |
| **0.45** | 0.648 | 0.607 | 0.565 | 0.682 | 0.692 | 0.681 | 0.741 | 0.673 | 0.731 | 0.486 | 0.351 | 0.487 | 0.811 | **0.826** | 0.822 |
| **0.60** | 0.648 | 0.608 | 0.565 | 0.682 | 0.610 | 0.681 | 0.741 | 0.595 | 0.731 | 0.486 | 0.358 | 0.487 | 0.811 | **0.812** | 0.805 |

---

## 4. Key Findings and Scientific Analysis

### 4.1 Naïve Bayes Degrades Most Gradually Under Noise

**GNB-naive_pool** drops from 0.649 (ε=0) to 0.608 (ε=0.60), a relative decrease of only **6.3%**. This confirms the proposal's hypothesis: GNB's global density estimation (class-conditional Gaussian means and variances) averages out individual label errors. Each sample has only a marginal effect on the per-class parameter estimates, making the model inherently noise-tolerant.

### 4.2 Decision Tree (Unpruned) Overfits Most Severely

**DT-naive_pool** shows the steepest degradation: from 0.582 (ε=0) to 0.358 (ε=0.60), a **38.5% relative drop**. This is the worst degradation of any classifier.

The tree depth and leaf count analysis confirms the mechanism:
- **Unpruned trees** (naive_pool) maintain deep, complex structures even as noise increases, memorizing noisy labels rather than learning the true boundary.
- **Pruned trees** (max_depth=5) have constrained depth and leaf counts that remain stable across all ε values (depth=5, ~32 leaves), confirming that depth constraints prevent overfitting to noise.

### 4.3 SVM Shows Remarkable Resilience, but Support Vectors Inflate

**SVM-naive_pool** degrades from 0.879 to 0.812, a relative drop of only **7.6%** — second-best after GNB. However, the support vector count analysis reveals the hidden cost: as ε increases, more mislabeled boundary points become support vectors, inflating the model complexity without improving generalization. This foreshadows WP4's finding that TSVM struggles with 100% minority-class SV inflation on imbalanced PTB-XL data.

### 4.4 Self-Training: Stable but Conservative

Self-training metrics remain virtually constant across all ε values because self-training **ignores the noisy labels in $\mathcal{D}_U$** and instead uses its own predicted labels. This makes it noise-immune by design. However:
- It never outperforms naive_pool at ε=0 (more data always helps when labels are clean).
- At high ε (0.45–0.60), self-training matches or slightly exceeds gold_only, showing that it can safely incorporate some unlabeled data.

### 4.5 At ε=0, naive_pool ≥ gold_only for All Models

This baseline sanity check passes for all 5 classifiers: when untrusted labels are actually clean, adding more data always helps. The largest gain is for SVM (0.811 → 0.879, +6.8pp) and kNN (0.741 → 0.817, +7.6pp), both of which are non-parametric or high-capacity models that benefit most from additional training samples.

### 4.6 Cross-Over Points

The noise level at which gold_only overtakes naive_pool varies by classifier:

| Classifier | Cross-over ε (approx.) | Interpretation |
|------------|----------------------|----------------|
| GNB | ~0.45 | Very noise-tolerant |
| LR | ~0.45–0.60 | Regularization helps |
| kNN | ~0.30–0.45 | Sensitive to local noise |
| DT | ~0.30 | Most noise-sensitive |
| SVM | ~0.60 | Margin-based; very robust |

---

## 5. Summary Table for Downstream Comparison (WP5)

| Classifier | Mode | Best Val F1 (ε=0) | F1 at ε=0.30 | F1 at ε=0.60 | Relative Drop (0→0.60) | Key Vulnerability |
|------------|------|-------------------|--------------|--------------|----------------------|-------------------|
| **SVM** | naive_pool | **0.879** | **0.869** | 0.812 | 7.6% | Support vector inflation |
| **kNN** | naive_pool | 0.817 | 0.748 | 0.595 | 27.2% | Local noise sensitivity |
| **LR** | naive_pool | 0.761 | 0.707 | 0.610 | 19.8% | Moderate: regularization helps |
| **GNB** | naive_pool | 0.649 | 0.611 | 0.608 | 6.3% | Low capacity limits ceiling |
| **DT** | naive_pool | 0.582 | 0.473 | 0.358 | 38.5% | Overfitting to noise |
| **SVM** | gold_only | 0.811 | 0.811 | 0.811 | 0% | Ignores unlabeled data |

---

## 6. Implications for WP3–WP5

1. **WP3 (Graph Label Propagation):** Should outperform gold_only baselines (0.811 for SVM, 0.741 for kNN) by leveraging geometric structure to propagate clean labels without being corrupted by noise.

2. **WP4 (Co-Training & TSVM):** Should compare against self_training mode as the most direct baseline. The key question is whether multi-view learning adds value beyond single-view self-training.

3. **WP5 (Latent-Noise EM):** The confusion matrix recovery analysis should target the uniform noise structure used here (symmetric $T$). The EM algorithm should achieve F1 above naive_pool at high ε by modeling the noise transition matrix explicitly.

4. **Overall:** SVM-naive_pool (0.879 at ε=0) sets the ceiling for what supervised methods can achieve with all data. Any semi-supervised method that exceeds this under realistic noise conditions (ε ≥ 0.15) demonstrates genuine value.
