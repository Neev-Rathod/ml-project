"""
wp2_synthetic_baselines/synthetic_generator.py
===============================================
Generates controlled synthetic datasets for benchmarking (Proposal §3.1).

Three types of label noise are supported:
  1. Uniform noise         – every class flips to any other with probability ε/(C-1)
  2. Asymmetric noise      – specific pairs confused via a sparse transition matrix T
  3. Boundary-dependent    – points near the decision boundary flip with higher probability

Reference: Proposal §3.1, Equation 3.1
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


CLASSES = 5
FEATURE_DIM = 50
INFORMATIVE_DIMS = 10


def generate_manifold(
    n_samples: int = 2000,
    n_classes: int = CLASSES,
    n_features: int = FEATURE_DIM,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a non-linear multi-class manifold in FEATURE_DIM dimensions.
    Uses sklearn's make_classification with a 3D informative subspace projected
    to FEATURE_DIM dimensions with added noise.

    Parameters
    ----------
    n_samples  : total number of samples
    n_classes  : number of classes (5 to match PTB-XL)
    n_features : ambient dimensionality (50)
    seed       : random seed

    Returns
    -------
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,) integer labels 0…n_classes-1
    """
    rng = np.random.default_rng(seed)

    # 1. Generate a base classification dataset with informative + redundant dims
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=INFORMATIVE_DIMS,
        n_redundant=5,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.0,
        flip_y=0.0,          # No label noise at generation time
        random_state=seed,
    )

    # 2. Apply a random projection to introduce non-linear mixing
    #    This makes the classes harder to separate with a linear classifier
    projection = rng.standard_normal((n_features, n_features))
    projection = np.linalg.qr(projection)[0]  # orthogonal projection
    X = X @ projection

    # 3. Add Gaussian noise to simulate the PTB-XL ambient noise structure
    noise_scale = 0.1
    X += rng.standard_normal(X.shape) * noise_scale

    return X, y


def inject_uniform_noise(
    y: np.ndarray,
    epsilon: float,
    n_classes: int = CLASSES,
    seed: int = 42,
) -> np.ndarray:
    """
    Uniform label noise: each label is flipped to a random different class
    with probability epsilon.

    Parameters
    ----------
    y       : clean integer labels (N,)
    epsilon : flip probability (0 ≤ ε ≤ 1)
    n_classes : C

    Returns
    -------
    y_noisy : ndarray (N,) with noisy labels
    """
    if epsilon <= 0.0:
        return y.copy()

    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    n = len(y)

    # For each sample, flip with probability epsilon to a uniform random
    # class drawn from {0, ..., C-1} \ {y[i]}
    flip_mask = rng.random(n) < epsilon

    for i in np.where(flip_mask)[0]:
        # Choose uniformly from all classes except the true class
        candidates = [c for c in range(n_classes) if c != y[i]]
        y_noisy[i] = rng.choice(candidates)

    return y_noisy


def inject_asymmetric_noise(
    y: np.ndarray,
    T: np.ndarray = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Asymmetric (class-conditional) noise via a C×C transition matrix T.
    T[c, k] = P(ỹ = k | z = c).  Rows must sum to 1.

    If T is None, builds a default transition matrix that confuses class
    pairs (0,1) and (2,3) with probability 0.3.

    Parameters
    ----------
    y : clean integer labels (N,)
    T : transition matrix (C, C) or None for default

    Returns
    -------
    y_noisy : ndarray (N,)
    """
    rng = np.random.default_rng(seed)

    if T is None:
        # Build default T: confuse class pairs (0,1) and (2,3) with prob 0.3
        C = CLASSES
        T = np.eye(C)
        # Class 0 ↔ Class 1: 30% confusion
        T[0, 0] = 0.7;  T[0, 1] = 0.3
        T[1, 1] = 0.7;  T[1, 0] = 0.3
        # Class 2 ↔ Class 3: 30% confusion
        T[2, 2] = 0.7;  T[2, 3] = 0.3
        T[3, 3] = 0.7;  T[3, 2] = 0.3
        # Class 4: no confusion (stays on diagonal)

    # Validate that rows sum to 1
    assert np.allclose(T.sum(axis=1), 1.0), \
        f"T rows must sum to 1, got: {T.sum(axis=1)}"

    y_noisy = np.empty_like(y)
    for i in range(len(y)):
        # Draw noisy label from Categorical(T[y[i], :])
        y_noisy[i] = rng.choice(len(T), p=T[y[i]])

    return y_noisy


def inject_boundary_noise(
    y: np.ndarray,
    X: np.ndarray,
    epsilon_max: float = 0.4,
    gamma: float = 5.0,
    n_classes: int = CLASSES,
    seed: int = 42,
) -> np.ndarray:
    """
    Boundary-dependent noise (Proposal Eq. 3.1).
    Records near the decision boundary have higher flip probability.

    P(flip) = epsilon_max * exp(-gamma * margin)
    where margin = P(z=y*|x) - max_{c≠y*} P(z=c|x)

    Parameters
    ----------
    y           : clean labels (N,)
    X           : feature matrix (N, d) – used to estimate boundary proximity
    epsilon_max : maximum flip probability for boundary points
    gamma       : sharpness of the boundary effect

    Returns
    -------
    y_noisy : ndarray (N,)
    """
    rng = np.random.default_rng(seed)

    # 1. Train a LogisticRegression on (X, y) to get P(z=c | x)
    lr = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    )
    lr.fit(X, y)
    proba = lr.predict_proba(X)  # shape (N, C)

    # Align probabilities with class indices 0..n_classes-1
    # (in case some classes are missing from y)
    classes = lr.classes_
    full_proba = np.zeros((len(X), n_classes))
    for col_idx, c in enumerate(classes):
        if 0 <= c < n_classes:
            full_proba[:, c] = proba[:, col_idx]

    y_noisy = y.copy()

    for i in range(len(y)):
        true_class = y[i]
        p_true = full_proba[i, true_class]

        # max probability among classes other than the true class
        other_probs = np.concatenate([
            full_proba[i, :true_class],
            full_proba[i, true_class + 1:]
        ])
        p_max_other = other_probs.max()

        # 2. Compute margin
        margin = p_true - p_max_other

        # 3. Compute flip probability using Eq. 3.1
        p_flip = epsilon_max * np.exp(-gamma * margin)
        p_flip = np.clip(p_flip, 0.0, 1.0)

        # 4. Sample noisy label
        if rng.random() < p_flip:
            candidates = [c for c in range(n_classes) if c != true_class]
            y_noisy[i] = rng.choice(candidates)

    return y_noisy
