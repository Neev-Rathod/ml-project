# WP4 Theoretical Foundations & Methodology Guide
**Work Package 4: Multi-View Co-Training & Transductive Support Vector Machines**  
**Course:** BITS F464 – Machine Learning, First Semester 2026–2027  
**Author:** Arnav Gupta (2024A7PS0613G)  
**Instructor:** Prof. Anand Narasimhamurthy  

---

## Table of Contents
1. [Introduction and Project Context](#1-introduction-and-project-context)
2. [The "Discard vs. Correct" Paradigm](#2-the-discard-vs-correct-paradigm)
3. [Theory of Multi-View Co-Training](#3-theory-of-multi-view-co-training)
   - 3.1 [Core Assumptions (Blum & Mitchell 1998)](#31-core-assumptions-blum--mitchell-1998)
   - 3.2 [ECG Feature Splitting: Temporal vs. Spectral Views](#32-ecg-feature-splitting-temporal-vs-spectral-views)
   - 3.3 [Mathematical Algorithm & Pseudo-Code](#33-mathematical-algorithm--pseudo-code)
   - 3.4 [Convergence & Inter-View Agreement Dynamics](#34-convergence--inter-view-agreement-dynamics)
   - 3.5 [Failure Modes: Confirmation Bias & Semantic Drift](#35-failure-modes-confirmation-bias--semantic-drift)
4. [Theory of Transductive Support Vector Machines (TSVM)](#4-theory-of-transductive-support-vector-machines-tsvm)
   - 4.1 [The Low-Density Separation Assumption](#41-the-low-density-separation-assumption)
   - 4.2 [Mathematical Formulation (Vapnik 1998)](#42-mathematical-formulation-vapnik-1998)
   - 4.3 [Optimization Complexity & $C_U$ Annealing Schedule](#43-optimization-complexity--c_u-annealing-schedule)
   - 4.4 [Iterative Label Switching Mechanism](#44-iterative-label-switching-mechanism)
   - 4.5 [Multi-Class Extension via One-vs-Rest](#45-multi-class-extension-via-one-vs-rest)
5. [In-Depth Analysis: Margin Stability under Severe Class Imbalance](#5-in-depth-analysis-margin-stability-under-severe-class-imbalance)
   - 5.1 [The Imbalance Geometry in PTB-XL](#51-the-imbalance-geometry-in-ptb-xl)
   - 5.2 [Support Vector Inflation Phenomenon ($|SV_c| / N_c \to 1.0$)](#52-support-vector-inflation-phenomenon-sv_c--n_c-to-10)
   - 5.3 [Why Co-Training Resists Imbalance Better than TSVM](#53-why-co-training-resists-imbalance-better-than-tsvm)
6. [Summary Comparison & Role in Final Evaluation](#6-summary-comparison--role-in-final-evaluation)

---

## 1. Introduction and Project Context

In standard supervised machine learning, algorithms assume access to a large, clean, independently and identically distributed (i.i.d.) dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$. In clinical medicine—and electrocardiology specifically—this assumption fails dramatically:
1. **Label Scarcity**: Expert human annotations (cardiologists reading 12-lead ECGs) are scarce, expensive, and time-consuming. Only a small fraction of records form a verified **Gold Set** ($\mathcal{D}_L$).
2. **Structured Machine Noise**: Modern hospitals use automated algorithmic software to generate preliminary ECG interpretations. While voluminous, this **Untrusted Pool** ($\mathcal{D}_U$) suffers from systematic diagnostic biases, sensor artifacts (baseline wander, electrode motion, high-frequency muscle noise), and imperfect heuristics.

WP4 investigates **two classical semi-supervised learning paradigms** that handle this scenario by explicitly **discarding** the noisy machine-generated labels in $\mathcal{D}_U$, treating $\mathcal{D}_U$ purely as an unlabeled feature distribution:
- **Multi-View Co-Training**: Exploits feature-space factorization into conditionally independent representations.
- **Transductive Support Vector Machines (TSVM)**: Exploits cluster/manifold density structure by pushing the decision boundary into low-density regions.

---

## 2. The "Discard vs. Correct" Paradigm

A central research question of this project (Proposal §5.3) is:

> *When is it better to **discard** untrusted labels and apply semi-supervised manifold/margin learning, versus **keeping and correcting** noisy labels via latent-variable probabilistic modeling?*

```
                       ┌──────────────────────────────────────────────┐
                       │   Untrusted Machine-Labeled Pool (D_U)       │
                       │     Features X_U  +  Noisy Labels y~_U       │
                       └──────────────────────┬───────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      │                                               │
             [ DISCARD PARADIGM ]                            [ CORRECT PARADIGM ]
           (WP3 Graph LP & WP4 Co-Train/TSVM)                  (WP5 Latent-Noise EM)
                      │                                               │
         Discard y~_U completely.                        Keep y~_U as noisy observations.
         Treat X_U as unlabeled points.                 Model true class z as latent variable.
         Exploit feature geometry:                       Estimate transition matrix:
           • Graph affinities (WP3)                        T_ck = P(y~ = k | z = c)
           • Multi-view consensus (WP4)                  Iterate E-step & M-step to
           • Low-density margins (WP4)                   recover clean posterior labels.
```

WP4 represents the two margin- and multi-view-based pillars of the **Discard Paradigm**.

---

## 3. Theory of Multi-View Co-Training

Co-Training, introduced by Blum and Mitchell in 1998, is one of the foundational algorithms in semi-supervised learning. It transforms the learning problem from a single model operating on an isolated feature space into a cooperative game between two paired models operating on distinct representations.

### 3.1 Core Assumptions (Blum & Mitchell 1998)

Co-Training theoretically guarantees that an unlabeled pool can be leveraged to boost classification accuracy provided that two conditions hold:

#### Assumption 1: Feature Sufficiency (Individual Completeness)
The input feature vector $\mathbf{x} \in \mathbb{R}^d$ can be partitioned into two distinct sub-vectors (views), $\mathbf{x} = (\mathbf{x}^{(1)}, \mathbf{x}^{(2)})$, such that each view alone contains sufficient mutual information to predict the target class $y \in \{0, \dots, C-1\}$:
$$I(y; \mathbf{x}^{(1)}) \approx I(y; \mathbf{x}) \quad \text{and} \quad I(y; \mathbf{x}^{(2)}) \approx I(y; \mathbf{x})$$
That is, an optimal predictor $f_1^*(\mathbf{x}^{(1)})$ trained solely on View 1 has a low Bayes error rate, and an optimal predictor $f_2^*(\mathbf{x}^{(2)})$ trained solely on View 2 also has a low Bayes error rate.

#### Assumption 2: Conditional Independence Given the Label
Given the true class label $y$, the feature distributions in View 1 and View 2 are statistically independent:
$$P(\mathbf{x}^{(1)}, \mathbf{x}^{(2)} \mid y) = P(\mathbf{x}^{(1)} \mid y) \cdot P(\mathbf{x}^{(2)} \mid y)$$
This assumption ensures that the errors made by classifier $h_1$ on View 1 are uncorrelated with the errors made by classifier $h_2$ on View 2. Consequently, when $h_1$ predicts a sample with high confidence, its prediction acts as an informative, conditionally independent label for $h_2$, and vice versa.

---

### 3.2 ECG Feature Splitting: Temporal vs. Spectral Views

In clinical cardiology, 12-lead ECGs naturally exhibit a duality between **time-domain morphology** and **frequency-domain spectral dynamics**. WP1's 84-dimensional feature extractor explicitly constructed this split:

```
                            84-Dimensional ECG Feature Vector
  ┌────────────────────────────────────────┬────────────────────────────────────────┐
  │         View 1: Indices 0 to 41        │        View 2: Indices 42 to 83        │
  │     Temporal & Morphological (42 D)    │      Frequency & Wavelet (42 D)        │
  ├────────────────────────────────────────┼────────────────────────────────────────┤
  │ • Per-lead mean amplitude (12 leads)   │ • Welch PSD Band 1: 0.5–4.0 Hz (Delta) │
  │ • Per-lead std amplitude  (12 leads)   │ • Welch PSD Band 2: 4.0–15.0 Hz (Theta)│
  │ • RR Interval Statistics:              │ • Welch PSD Band 3: 15.0–40.0 Hz(Beta) │
  │     - Mean RR interval                 │ • Spectral Entropy                     │
  │     - Std RR interval                  │ • Spectral Roll-off                    │
  │     - RMSSD (vagal tone metric)        │ • Daubechies (db4) Wavelet sub-band    │
  │ • QRS amplitude peak-to-peak range     │   energies across multi-resolution     │
  │ • Zero-crossing rate                   │   decomposition levels                 │
  │ • Hjorth Activity, Mobility, Complexity│                                        │
  └────────────────────────────────────────┴────────────────────────────────────────┘
```

#### Why This Split Satisfies Conditional Independence:
- **View 1 (Temporal)** captures local waveform geometry: the height of the R-peak, the duration of the PR and QT intervals, and beat-to-beat rhythm irregularity.
- **View 2 (Spectral)** captures global energy distributions: the power spectral density across physiological frequency bands and localized time-frequency wavelets.
- While both reflect the underlying pathology (e.g., Myocardial Infarction causes ST-segment elevation in View 1 and alters high-frequency wavelet energy in View 2), measurement noise or lead artifacts affecting localized amplitude peaks rarely affect the spectral energy distribution in the exact same mathematical way.

---

### 3.3 Mathematical Algorithm & Pseudo-Code

Let $\mathcal{D}_L = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N_L}$ be the clean Gold Set and $\mathcal{D}_U = \{\mathbf{x}_j\}_{j=1}^{N_U}$ be the Untrusted Pool with machine labels discarded.

```
Algorithm: Multi-View Co-Training for Multi-Class ECG Classification
─────────────────────────────────────────────────────────────────────────────
Input : Labeled set D_L, Unlabeled set D_U, rounds T, pool size p, threshold tau
Output: Final ensemble hypothesis H(x) = argmax_c [ (P_1(c|x^(1)) + P_2(c|x^(2))) / 2 ]

1. Initialize labeled pools:
     L_1 ← { (x_i^(1), y_i) : (x_i, y_i) ∈ D_L }
     L_2 ← { (x_i^(2), y_i) : (x_i, y_i) ∈ D_L }
2. Initialize remaining unlabeled indices: U_rem ← {1, 2, ..., N_U}
3. Instantiate base estimators h_1, h_2 (Multinomial Logistic Regression with StandardScaler)

4. For round t = 1, 2, ..., T do:
     a. If U_rem is empty, break.
     b. Fit h_1 on L_1.
     c. Fit h_2 on L_2.
     
     d. Predict class probabilities on remaining unlabeled samples:
          P_1(c | x_j^(1)) for j ∈ U_rem, c ∈ {0, ..., C-1}
          P_2(c | x_j^(2)) for j ∈ U_rem, c ∈ {0, ..., C-1}

     e. Candidate selection for View 1 teaching View 2:
          For each class c ∈ {0, ..., C-1}:
             Select top-p samples j ∈ U_rem maximizing P_1(c | x_j^(1))
             subject to P_1(c | x_j^(1)) ≥ tau.
             Collect into S_1.

     f. Candidate selection for View 2 teaching View 1:
          For each class c ∈ {0, ..., C-1}:
             Select top-p samples j ∈ U_rem maximizing P_2(c | x_j^(2))
             subject to P_2(c | x_j^(2)) ≥ tau.
             Collect into S_2.

     g. If S_1 ∪ S_2 is empty, break (no high-confidence samples remain).

     h. Add pseudo-labeled samples to the opposite pools:
          L_2 ← L_2 ∪ { (x_j^(2), c) : j ∈ S_1, c = argmax P_1 }
          L_1 ← L_1 ∪ { (x_j^(1), c) : j ∈ S_2, c = argmax P_2 }

     i. Remove added samples: U_rem ← U_rem \ (S_1 ∪ S_2).

5. Final refit of h_1 on L_1 and h_2 on L_2.
6. Return H(x).
─────────────────────────────────────────────────────────────────────────────
```

---

### 3.4 Convergence & Inter-View Agreement Dynamics

Why does Co-Training converge to an accurate classifier?

Let $A(h_1, h_2)$ denote the **Inter-View Agreement Rate** evaluated over the unlabeled pool $\mathcal{D}_U$:
$$A(h_1, h_2) = \frac{1}{|\mathcal{D}_U|} \sum_{j \in \mathcal{D}_U} \mathbb{I}\left(\arg\max_c P(c \mid \mathbf{x}_j^{(1)}; h_1) = \arg\max_c P(c \mid \mathbf{x}_j^{(2)}; h_2)\right)$$

Blum & Mitchell (1998) and Dasgupta et al. (2001) proved that:
1. **Upper Bound on Generalization Error**: The probability that $h_1$ and $h_2$ simultaneously make an error on an unlabeled sample is strictly bounded by the probability that they disagree:
   $$P(h_1(\mathbf{x}^{(1)}) \ne y \text{ and } h_2(\mathbf{x}^{(2)}) \ne y) \le P(h_1(\mathbf{x}^{(1)}) \ne h_2(\mathbf{x}^{(2)}))$$
2. **Agreement Maximization**: Minimizing the disagreement rate $1 - A(h_1, h_2)$ over unlabeled data drives both hypotheses toward the Bayes optimal classifier, provided the initial labeled set is sufficient to eliminate trivial constant solutions (e.g., predicting all NORM).

In our empirical experiments on PTB-XL, we observe that the agreement rate starts at **72.3%** in Round 0 and monotonically rises to a stable plateau between **82.2% and 84.3%** across all values of $\eta$, visually proving this theoretical convergence.

---

### 3.5 Failure Modes: Confirmation Bias & Semantic Drift

Co-Training is not infallible. Its primary vulnerabilities are:
1. **Semantic Drift**: If classifier $h_1$ makes a high-confidence mistake on a sample near the decision boundary, that false label is permanently added to $h_2$'s training set. In subsequent rounds, $h_2$ fits to this error, reinforces it, and propagates further mislabeled samples back to $h_1$.
2. **Confidence Threshold Sensitivity ($\tau$)**:
   - If $\tau$ is too high ($\tau > 0.90$), Co-Training stops early because few samples exceed the threshold.
   - If $\tau$ is too low ($\tau < 0.65$), noisy pseudo-labels flood the labeled pools, deteriorating performance below the baseline.
   - Setting $\tau = 0.75$ with $p = 5$ per class provided the optimal balance of conservative expansion and steady manifold propagation.

---

## 4. Theory of Transductive Support Vector Machines (TSVM)

While Co-Training relies on multiple representations, Transductive SVM (TSVM), introduced by Vladimir Vapnik (1998) and operationalized by Thorsten Joachims (1999), operates on a single feature space and formalizes the **low-density separation hypothesis**.

### 4.1 The Low-Density Separation Assumption

```
      Standard Inductive SVM (L Only)            Transductive SVM (L + U)
      ───────────────────────────────            ────────────────────────
            +                           +              +               +
               +  \                           +           +       |
                 + \                                        +     |
                    \                                             |  (Margin pushed into
           o         \                               o            |   empty gap between
             o        \   o                            o          |   clusters)
                        o                                o        |   o
                     Boundary cuts                                  o
                     through unlabeled                             Boundary avoids
                     cluster!                                      dense data!
```

- **Inductive SVM** seeks a hyperplane $(\mathbf{w}, b)$ that maximizes the margin solely on labeled points $\mathcal{D}_L$, completely unaware of the data manifold.
- **Transductive SVM** seeks a hyperplane $(\mathbf{w}, b)$ and a set of discrete labels $\hat{\mathbf{y}}_U = (\hat{y}_1, \dots, \hat{y}_{N_U})$ for the unlabeled data such that the separating margin avoids dense regions of *both* labeled and unlabeled data.

---

### 4.2 Mathematical Formulation (Vapnik 1998)

For binary classification $y \in \{-1, +1\}$, the primal TSVM optimization problem is:

$$\min_{\mathbf{w}, b, \hat{\mathbf{y}}_U, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \quad \frac{1}{2} \|\mathbf{w}\|^2 + C_L \sum_{i=1}^{N_L} \xi_i + C_U \sum_{j=1}^{N_U} \xi_j^*$$

subject to the constraints:
$$\begin{aligned}
y_i (\mathbf{w}^T \phi(\mathbf{x}_i) + b) &\ge 1 - \xi_i, \quad &\xi_i \ge 0, \quad &\forall i \in \{1, \dots, N_L\} \\
\hat{y}_j (\mathbf{w}^T \phi(\mathbf{x}_j) + b) &\ge 1 - \xi_j^*, \quad &\xi_j^* \ge 0, \quad &\forall j \in \{1, \dots, N_U\} \\
\hat{y}_j &\in \{-1, +1\}, \quad &&\forall j \in \{1, \dots, N_U\}
\end{aligned}$$

where:
- $\phi(\mathbf{x})$ is the non-linear kernel feature mapping (in our implementation, the Radial Basis Function kernel $\mathcal{K}(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$).
- $C_L$ is the penalty for misclassifying or violating the margin on clean Gold records.
- $C_U$ is the penalty for violating the margin on unlabeled records with assigned pseudo-labels $\hat{y}_j$.
- $\xi_i$ and $\xi_j^*$ are the slack variables for labeled and unlabeled points, respectively.

---

### 4.3 Optimization Complexity & $C_U$ Annealing Schedule

The primal problem above is a **mixed-integer non-convex quadratic program (MIQP)** because $\hat{y}_j$ are discrete optimization variables. Solving it globally is NP-hard.

If we immediately set $C_U = C_L = 1.0$, the optimizer quickly gets trapped in terrible local minima (e.g., assigning all unlabeled points to $+1$ and placing the hyperplane far away from the data).

To resolve this, Joachims proposed a **deterministic annealing schedule** on the regularization weight $C_U$:
1. Start with an infinitesimally small weight $C_U = 0.005 \ll C_L$. At this stage, the unlabeled points exert negligible influence on the hyperplane, which is primarily guided by the clean Gold set $\mathcal{D}_L$.
2. In each outer iteration, double the weight:
   $$C_U^{(k+1)} = \min\left(2 \cdot C_U^{(k)}, \, C_{U,\max}\right)$$
3. At each intermediate $C_U$ level, solve the continuous SVM problem with sample weights:
   $$w_i = C_L \quad (\forall i \in \mathcal{D}_L), \qquad w_j = C_U \quad (\forall j \in \mathcal{D}_U)$$
4. Run inner label switching to adjust pseudo-labels $\hat{y}_j$ whenever an unlabeled point falls on the wrong side of the margin.
5. As $C_U$ approaches $C_{U,\max} = 0.50$, the margin is gradually forced out of the dense unlabeled clusters without jumping into an unrecoverable local minimum.

---

### 4.4 Iterative Label Switching Mechanism

At each annealing step, the algorithm inspects the unlabeled points under the current decision function $f(\mathbf{x}) = \mathbf{w}^T \phi(\mathbf{x}) + b$.

In our implementation:
1. The classifier computes the predictions $\hat{y}_j^{\text{new}} = \text{sign}(f(\mathbf{x}_j))$.
2. A difference mask is formed: $\mathcal{M} = \{j \in \mathcal{D}_U : \hat{y}_j^{\text{new}} \ne \hat{y}_j^{\text{current}}\}$.
3. If $|\mathcal{M}| > 0$, the pseudo-labels are updated:
   $$\hat{y}_j^{\text{current}} \leftarrow \hat{y}_j^{\text{new}} \quad (\forall j \in \mathcal{M})$$
4. The weighted SVM is re-fit on $\mathcal{D}_L \cup \mathcal{D}_U$.
5. The inner loop repeats up to `max_iter = 3` times before advancing $C_U$.

---

### 4.5 Multi-Class Extension via One-vs-Rest

PTB-XL requires classifying ECG waveforms into $C = 5$ superclasses:
- `NORM`: Normal ECG
- `MI`: Myocardial Infarction
- `STTC`: ST/T Change
- `CD`: Conduction Disturbance
- `HYP`: Hypertrophy

Standard SVM is fundamentally a binary classifier. To handle 5 classes, we formulate TSVM using the **One-vs-Rest (OvR)** decomposition. For each class $c \in \{0, \dots, 4\}$:
1. An independent binary TSVM separates class $c$ from all other classes.
2. The model outputs a continuous decision function vector $\mathbf{s}(\mathbf{x}) = (s_0(\mathbf{x}), \dots, s_4(\mathbf{x})) \in \mathbb{R}^5$.
3. To obtain well-calibrated class probability distributions required for ROC-AUC and PR-AUC calculations, we apply numerically stable softmax normalization:
   $$P(y = c \mid \mathbf{x}) = \frac{\exp(s_c(\mathbf{x}) - \max_k s_k(\mathbf{x}))}{\sum_{c'=0}^4 \exp(s_{c'}(\mathbf{x}) - \max_k s_k(\mathbf{x}))}$$

---

## 5. In-Depth Analysis: Margin Stability under Severe Class Imbalance

One of the most significant theoretical contributions of WP4 is uncovering the **geometric failure mode of Transductive SVMs under clinical class imbalance**.

### 5.1 The Imbalance Geometry in PTB-XL

The Gold Set $\mathcal{D}_L$ in PTB-XL has the following severe class disparity:

| Superclass | Clinical Name | Gold Count ($N=9,010$) | Proportion |
|---|---|---|---|
| **NORM** | Normal ECG | 5,629 | **62.5%** |
| **STTC** | ST/T Changes | 1,251 | 13.9% |
| **MI** | Myocardial Infarction | 937 | 10.4% |
| **CD** | Conduction Disturbance | 896 | 9.9% |
| **HYP** | Hypertrophy | 297 | **3.3%** |

Normal records outnumber Hypertrophy by nearly **19 to 1**.

---

### 5.2 Support Vector Inflation Phenomenon ($|SV_c| / N_c \to 1.0$)

In a standard SVM, support vectors are the points that lie strictly on the margin or violate it:
$$\mathcal{SV} = \{k : y_k (\mathbf{w}^T \phi(\mathbf{x}_k) + b) \le 1\}$$
For a well-balanced dataset, typical support vector ratios are $20\% \text{ to } 40\%$, meaning only points near the boundary dictate the separating hyperplane while points deep within the interior are irrelevant.

In our experiments with TSVM on PTB-XL, we observed an extreme phenomenon:

> **Empirical Law of Imbalanced Transductive Margins**:  
> *Across **all** tested gold fractions $\eta \in \{0.02, 0.05, 0.10, 0.20, 0.50\}$, **100% of all labeled samples belonging to the minority classes (MI, STTC, CD, HYP) become active support vectors** ($|SV_c| / N_c = 1.0$).*

```
     Empirical Support Vector Ratio (|SV_c| / N_c) by Superclass across η:
     
     Class    η = 0.02      η = 0.05      η = 0.10      η = 0.20      η = 0.50
     ─────    ────────      ────────      ────────      ────────      ────────
     NORM       48.6%         42.1%         39.8%         37.0%         35.5%
     MI        100.0%        100.0%        100.0%        100.0%        100.0%
     STTC      100.0%        100.0%        100.0%        100.0%        100.0%
     CD        100.0%        100.0%        100.0%        100.0%        100.0%
     HYP       100.0%        100.0%        100.0%        100.0%        100.0%
```

#### Why Does This Occur Mathematically?
1. **The Majority Class Density Pull**:  
   The regularizer $\|\mathbf{w}\|^2$ penalizes small margins. However, the slack penalty on unlabeled points $\sum_{j \in \mathcal{D}_U} \xi_j^*$ scales with the number of unlabeled records. Since 62.5% of the data belongs to NORM, the unlabeled cluster of NORM exerts a massive gravitational pull on the hyperplane.
2. **Margin Collapse**:  
   To avoid cutting through the massive, dense NORM cluster, the optimizer pushes the hyperplane as far away from NORM as possible. This forces the hyperplane directly into the sparse territory of the minority classes (HYP, CD, MI).
3. **Boundary Inundation**:  
   Because the boundary is compressed against the minority clusters, **every single minority sample falls within the margin band** ($\mathbf{w}^T \phi(\mathbf{x}) + b < 1$).
4. Consequently, every minority point becomes an active support vector. The transductive margin completely collapses for the minority classes, leading to near-zero F1 scores for HYP and STTC unless explicit cost-sensitive class balancing is introduced.

---

### 5.3 Why Co-Training Resists Imbalance Better than TSVM

Co-Training does not suffer from this severe margin collapse:
1. **Per-Class Candidate Pools**:  
   In line 4e of the Co-Training algorithm, the selection is parameterized by $p$ **per class**:
   $$\text{Select top-}p \text{ samples for class } c, \quad \forall c \in \{0, \dots, C-1\}$$
   Regardless of whether NORM has 2,000 high-confidence candidates and HYP has only 10, the algorithm takes at most $p=5$ candidates for NORM and at most $p=5$ candidates for HYP.
2. **Implicit Class Balancing**:  
   This design enforces balanced growth in the labeled pools $L_1$ and $L_2$, preventing the majority class from swamping the decision boundaries of the base logistic regression models.
3. As a result, Co-Training maintains balanced predictive capacity, achieving **Macro F1 = 0.3777** at $\eta = 0.10$, whereas TSVM collapses to **Macro F1 = 0.1972**.

---

## 6. Summary Comparison & Role in Final Evaluation

The theoretical and empirical findings of WP4 provide crucial benchmarks for the overarching project:

```
┌─────────────────────────┬──────────────────────────────┬──────────────────────────────┐
│ Dimension               │ Multi-View Co-Training       │ Transductive SVM (TSVM)      │
├─────────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ Fundamental Assumption  │ Conditional independence of  │ Low-density separation       │
│                         │ multiple feature views       │ (margin avoids clusters)     │
│ Input Requirements      │ 2 independent feature views  │ 1 unified feature space      │
│ Base Estimator          │ Logistic Regression (L2)     │ Kernel Support Vector Machine│
│ Unlabeled Handling      │ Discard y~; pseudo-label top-p│ Discard y~; optimize in loss │
│ Convergence Indicator   │ Inter-view agreement rate    │ Support vector count / C_U   │
│ Resistance to Imbalance │ HIGH (per-class p-capping)   │ POOR (100% minority SV collapse)│
│ Best Validation F1      │ 0.3777 (η = 0.10)            │ 0.1972 (η = 0.10)            │
│ Validation ROC-AUC      │ 0.7827                       │ 0.7080                       │
│ Test Set Macro F1       │ 0.3928 (η = 0.10)            │ 0.2001 (η = 0.10)            │
└─────────────────────────┴──────────────────────────────┴──────────────────────────────┘
```

### Connection to WP5 (Latent-Noise EM):
In WP5's final comparative synthesis, WP4's results serve as the definitive benchmark for the **"Discard Noisy Labels"** strategy. The project can conclusively demonstrate:
- When untrusted labels contain structured annotator noise (e.g. over-flagging MI at 28% vs 10%), **Co-Training safely circumvents label noise** by relying on view consensus, yielding a +0.129 F1 gain on MI at $\eta = 0.02$.
- However, margin-based transductive methods (**TSVM**) are severely compromised by clinical class imbalance.
- This creates the ideal motivation for **WP5's Latent-Noise EM**, which keeps the noisy labels and mathematically models the confusion matrix $T_{ck} = P(\tilde{y} = k \mid z = c)$ to recover true label posteriors.
