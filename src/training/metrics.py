"""
src/training/metrics.py

Competition metrics and threshold optimization utilities.
Used by both analysis.ipynb (to find optimal thresholds on OOF)
and inference.ipynb (to apply those thresholds to test predictions).

Reference:
  - Nelder-Mead threshold optimization: fas_audit_and_plan.md §P1 Item 6
  - Nested CV design: Efron & Tibshirani (1997)
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score


# ── Core metric ──────────────────────────────────────────────────────────────

def macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    """Macro-averaged F1 score. Competition evaluation metric."""
    return f1_score(labels, preds, average='macro', zero_division=0)


# ── Threshold optimization ────────────────────────────────────────────────────

def _neg_macro_f1_thresh(thresholds, probs, labels):
    """Objective: negative Macro F1 after applying per-class threshold scaling."""
    scaled = probs * np.abs(thresholds)   # abs() prevents negative multipliers
    preds  = scaled.argmax(axis=1)
    return -macro_f1(labels, preds)


def optimize_thresholds(probs: np.ndarray, labels: np.ndarray,
                        n_restarts: int = 30, n_classes: int = 6):
    """Per-class threshold optimization via Nelder-Mead with multiple restarts.

    Threshold acts as a per-class multiplier on softmax probability:
        argmax(prob * threshold)
    Shifting a class threshold up makes the model more reluctant to predict
    that class; shifting it down makes the model more eager.

    Multiple random restarts reduce sensitivity to local minima.

    Args:
        probs:      (N, n_classes) float array of softmax probabilities.
        labels:     (N,) int array of ground-truth class indices.
        n_restarts: Number of random restarts for Nelder-Mead.
        n_classes:  Number of classes.

    Returns:
        best_thresholds: (n_classes,) array of optimal threshold multipliers.
        best_f1:         Macro F1 achieved with those thresholds.
    """
    best_f1, best_t = 0.0, np.ones(n_classes)
    for i in range(n_restarts):
        # Restart 0 always uses the no-op baseline [1,1,...,1] — guarantees result >= raw argmax
        init   = np.ones(n_classes) if i == 0 else np.random.uniform(0.5, 2.0, n_classes)
        result = minimize(
            _neg_macro_f1_thresh, init, args=(probs, labels),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5}
        )
        if -result.fun > best_f1:
            best_f1 = -result.fun
            best_t  = np.abs(result.x)
    return best_t, best_f1


def nested_cv_thresholds(probs: np.ndarray, labels: np.ndarray,
                          n_outer: int = 5, n_restarts: int = 20,
                          n_classes: int = 6, seed: int = 42):
    """Nested CV threshold optimization to prevent overfitting.

    Motivation: Optimizing thresholds on all OOF data is circular —
    the thresholds become tuned to the validation distribution, inflating CV.

    Solution (Efron & Tibshirani, 1997): split OOF into n_outer inner folds,
    optimize on (n_outer - 1) folds, evaluate on held-out fold, average thresholds.
    The OOB F1 values give an honest estimate of how much the thresholds
    will actually help on unseen data (i.e., the LB).

    Args:
        probs:      (N, n_classes) float array of OOF softmax probabilities.
        labels:     (N,) int array of ground-truth class indices.
        n_outer:    Number of outer CV folds.
        n_restarts: Random restarts per inner optimization call.
        n_classes:  Number of classes.
        seed:       Random seed for reproducibility.

    Returns:
        avg_thresholds: (n_classes,) array — average thresholds across folds.
                        Use this for inference.
        oob_f1s:        list of length n_outer — held-out F1 per fold.
                        np.mean(oob_f1s) is the honest LB gain estimate.
        all_thresh:     (n_outer, n_classes) — thresholds from each fold.
                        Large spread = thresholds are unstable, don't trust them.
    """
    rng      = np.random.default_rng(seed)
    idx      = np.arange(len(labels))
    rng.shuffle(idx)
    fold_idx = np.array_split(idx, n_outer)

    all_thresh, oob_f1s = [], []
    for i in range(n_outer):
        val_idx   = fold_idx[i]
        train_idx = np.concatenate([fold_idx[j] for j in range(n_outer) if j != i])
        t, _      = optimize_thresholds(
            probs[train_idx], labels[train_idx],
            n_restarts=n_restarts, n_classes=n_classes
        )
        preds_val = (probs[val_idx] * t).argmax(axis=1)
        oob_f1s.append(macro_f1(labels[val_idx], preds_val))
        all_thresh.append(t)

    return np.mean(all_thresh, axis=0), oob_f1s, np.array(all_thresh)


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Apply threshold multipliers to softmax probs and return predicted class indices.

    Usage in inference.ipynb:
        thresholds = pd.read_csv('reports/thresholds_exp03.csv')['threshold_nested_cv'].values
        preds = apply_thresholds(test_probs, thresholds)
    """
    return (probs * np.abs(thresholds)).argmax(axis=1)
