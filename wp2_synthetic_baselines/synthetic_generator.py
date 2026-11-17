"""
wp2_synthetic_baselines/synthetic_generator.py
===============================================
Generates controlled synthetic datasets for benchmarking (Proposal §3.1).

Three types of label noise are supported:
  1. Uniform noise         – every class flips to any other with probability ε/(C-1)
  2. Asymmetric noise      – specific pairs confused via a sparse transition matrix T
  3. Boundary-dependent    – points near the decision boundary flip with higher probability

TODO for Arnav (WP2)
---------------------
Implement the three functions marked with TODO below.  The signatures and
docstrings are already set; fill in the bodies.

Reference: Proposal §3.1, Equation 3.1
"""

import numpy as np
from sklearn.datasets import make_classification


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
    # TODO: Implement.
    # Suggested approach:
    #   1. Use make_classification with n_informative=INFORMATIVE_DIMS,
    #      n_redundant=5, n_repeated=0, n_features=FEATURE_DIM.
    #   2. Add a small amount of Gaussian noise to the remaining dimensions
    #      to simulate the PTB-XL ambient noise structure.
    raise NotImplementedError("generate_manifold not yet implemented")


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
    # TODO: Implement.
    # For each sample i:
    #   with prob epsilon → sample uniformly from {0,...,C-1} \ {y[i]}
    #   otherwise         → keep y[i]
    raise NotImplementedError("inject_uniform_noise not yet implemented")


def inject_asymmetric_noise(
    y: np.ndarray,
    T: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Asymmetric (class-conditional) noise via a C×C transition matrix T.
    T[c, k] = P(ỹ = k | z = c).  Rows must sum to 1.

    Parameters
    ----------
    y : clean integer labels (N,)
    T : transition matrix (C, C)

    Returns
    -------
    y_noisy : ndarray (N,)
    """
    # TODO: Implement.
    # For each sample, draw ỹ from Categorical(T[y[i], :])
    raise NotImplementedError("inject_asymmetric_noise not yet implemented")


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
    # TODO: Implement.
    # Steps:
    #   1. Train a LogisticRegression on (X, y) to get P(z=c | x).
    #   2. For each sample, compute margin = P(z=y*|x) - max_{c≠y*} P(z=c|x).
    #   3. Compute flip probability using Equation 3.1 from the proposal.
    #   4. Sample noisy labels accordingly.
    raise NotImplementedError("inject_boundary_noise not yet implemented")
