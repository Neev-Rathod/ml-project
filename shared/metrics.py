"""
shared/metrics.py
=================
Evaluation metrics used uniformly across all Work Packages.
All functions return plain dicts so results are easy to serialise to CSV.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from shared.data_utils import SUPERCLASSES


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute all project metrics for a fitted model on a test set.

    Parameters
    ----------
    model   : any object with .predict() and .predict_proba() methods
    X_test  : ndarray (n, d)
    y_test  : ndarray (n,) integer labels 0…C-1

    Returns
    -------
    results : dict with keys
        macro_f1        – float
        per_class_f1    – dict {class_name: f1}
        roc_auc         – float  (macro one-vs-rest)
        pr_auc          – float  (macro one-vs-rest)
        confusion_matrix – ndarray (C, C)
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    C = len(SUPERCLASSES)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    per_class_f1_arr = f1_score(
        y_test, y_pred, average=None, labels=list(range(C)), zero_division=0
    )
    per_class_f1 = {SUPERCLASSES[c]: round(float(per_class_f1_arr[c]), 4)
                    for c in range(C)}

    # one-vs-rest AUCs — requires probability scores
    try:
        roc_auc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro"
        )
    except ValueError:
        roc_auc = float("nan")

    try:
        # PR-AUC: average over classes
        pr_aucs = []
        for c in range(C):
            binary_y = (y_test == c).astype(int)
            if binary_y.sum() > 0:
                pr_aucs.append(average_precision_score(binary_y, y_proba[:, c]))
        pr_auc = float(np.mean(pr_aucs)) if pr_aucs else float("nan")
    except ValueError:
        pr_auc = float("nan")

    cm = confusion_matrix(y_test, y_pred, labels=list(range(C)))

    return {
        "macro_f1":        round(float(macro_f1), 4),
        "per_class_f1":    per_class_f1,
        "roc_auc":         round(float(roc_auc), 4),
        "pr_auc":          round(float(pr_auc), 4),
        "confusion_matrix": cm,
    }


def frobenius_error(T_hat: np.ndarray, T_true: np.ndarray) -> float:
    """
    Frobenius norm error between estimated and true noise transition matrices.
    Used in the synthetic benchmark (WP5 §5.4).
    """
    return float(np.linalg.norm(T_hat - T_true, "fro"))


def print_results(results: dict, model_name: str = "") -> None:
    """Pretty-print an evaluate_model() results dict."""
    header = f"── {model_name} ──" if model_name else "── Results ──"
    print(header)
    print(f"  Macro F1  : {results['macro_f1']:.4f}")
    print(f"  ROC-AUC   : {results['roc_auc']:.4f}")
    print(f"  PR-AUC    : {results['pr_auc']:.4f}")
    print("  Per-class F1:")
    for cls, f1 in results["per_class_f1"].items():
        print(f"    {cls:6s}: {f1:.4f}")
