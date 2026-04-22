"""
src/inference/threshold.py — Per-class threshold optimization for Macro F1.

The default argmax of softmax probabilities is not optimal for Macro F1
because it treats all classes equally. Per-class multiplicative thresholds
let us boost under-predicted classes and suppress over-predicted ones.

Usage:
    from src.inference.threshold import (
        apply_thresholds,
        robust_threshold_optimization,
    )

    thresholds, f1 = robust_threshold_optimization(oof_probs, oof_labels)
    preds = apply_thresholds(test_probs, thresholds)
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score

from src.utils.config import NUM_CLASSES


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Apply per-class multiplicative thresholds to a probability matrix.

    How thresholds work:
        We multiply each class's probability column by a threshold value,
        then take argmax. A threshold > 1.0 boosts that class (makes it
        relatively more likely to be predicted), while < 1.0 suppresses it.

        This is equivalent to shifting the decision boundary between classes:
            argmax(probs × thresholds) ≠ argmax(probs)  when thresholds ≠ [1, 1, ..., 1]

        The optimization learns thresholds that maximise Macro F1 on the OOF
        predictions, which directly targets the competition metric.

    Parameters
    ----------
    probs      : np.ndarray shape (N, NUM_CLASSES) — softmax probabilities
    thresholds : np.ndarray shape (NUM_CLASSES,) — per-class multipliers

    Returns
    -------
    preds : np.ndarray shape (N,) — predicted class indices
    """
    return (probs * np.abs(thresholds)).argmax(axis=1)


def _macro_f1_neg(thresholds: np.ndarray, probs: np.ndarray,
                  labels: np.ndarray) -> float:
    """Negated Macro F1 — objective function for scipy.optimize.minimize."""
    return -f1_score(labels, apply_thresholds(probs, thresholds), average='macro')


def robust_threshold_optimization(oof_probs: np.ndarray,
                                   oof_labels: np.ndarray,
                                   n_restarts: int = 30):
    """
    Optimizes per-class thresholds with multiple random restarts.

    Why multiple restarts:
        Nelder-Mead is a local optimizer. The F1 landscape over 6 thresholds
        has many local maxima. Running from multiple starting points and
        keeping the global best gives a more reliable result than a single run.

    Why we use the full OOF set (not nested CV):
        We are optimizing only 6 thresholds on 1342 samples — a highly
        underdetermined problem where nested CV adds implementation complexity
        without meaningful regularization benefit. The thresholds themselves
        are not expressive enough to overfit severely at this scale.

    Restart strategy:
        Restart 0 always starts from uniform thresholds [1.0, ..., 1.0] —
        this is the no-op baseline and guarantees we never return a result
        worse than raw argmax. Subsequent restarts sample from [0.5, 2.0].

    Parameters
    ----------
    oof_probs  : np.ndarray shape (N, NUM_CLASSES)
    oof_labels : np.ndarray shape (N,) — integer class indices
    n_restarts : int — number of optimization attempts (default 30)

    Returns
    -------
    best_thresholds : np.ndarray shape (NUM_CLASSES,)
    best_f1         : float — Macro F1 achieved with best_thresholds
    """
    best_f1     = 0.0
    best_thresh = np.ones(NUM_CLASSES)

    for i in range(n_restarts):
        init = (
            np.ones(NUM_CLASSES)
            if i == 0
            else np.random.uniform(0.5, 2.0, NUM_CLASSES)
        )

        res = minimize(
            _macro_f1_neg, init,
            args=(oof_probs, oof_labels),
            method='Nelder-Mead',
            options={'maxiter': 20_000, 'xatol': 1e-5, 'fatol': 1e-7},
        )

        f1 = -res.fun
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = np.abs(res.x)   # abs() prevents negative multipliers inverting class preference

    return best_thresh, best_f1
