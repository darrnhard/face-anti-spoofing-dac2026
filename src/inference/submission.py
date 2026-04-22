"""
src/inference/submission.py — Generates competition submission CSVs.

Converts a probability matrix (numpy array) into the competition's required
CSV format (id, label) and saves it to the submissions directory.

Usage:
    from src.inference.submission import make_submission

    sub, preds = make_submission(
        probs       = test_probs_top2,
        test_df     = test_df,
        thresholds  = optimal_thresholds,
        name        = 'v3_top2_thresh',
    )
"""

from pathlib import Path

import numpy as np

from src.inference.threshold import apply_thresholds
from src.utils.config import IDX_TO_CLASS, SUBMISSION_DIR


def make_submission(probs: np.ndarray, test_df,
                    thresholds: np.ndarray = None,
                    name: str = 'submission',
                    submission_dir=None):
    """
    Converts a probability matrix into a competition submission CSV.

    Prediction logic:
        - If thresholds are provided: preds = argmax(probs × thresholds)
        - If thresholds are None:     preds = argmax(probs)  [raw argmax]

    The output CSV is sorted by 'id' to match the competition's expected format.

    Parameters
    ----------
    probs          : np.ndarray shape (N, NUM_CLASSES)
    test_df        : pd.DataFrame — must have an 'id' column
    thresholds     : np.ndarray shape (NUM_CLASSES,) | None
    name           : str — output filename without extension (e.g. 'v3_top2_thresh')
    submission_dir : Path | None — defaults to SUBMISSION_DIR from config

    Returns
    -------
    sub       : pd.DataFrame — the submission DataFrame (columns: id, label)
    preds_idx : np.ndarray shape (N,) — integer predicted class indices
    """
    if submission_dir is None:
        submission_dir = SUBMISSION_DIR

    Path(submission_dir).mkdir(parents=True, exist_ok=True)

    # ── Predict ───────────────────────────────────────────────────────────────
    if thresholds is not None:
        preds_idx = apply_thresholds(probs, thresholds)
    else:
        preds_idx = probs.argmax(axis=1)

    # ── Build DataFrame ───────────────────────────────────────────────────────
    sub = test_df[['id']].copy()
    sub['label'] = [IDX_TO_CLASS[int(p)] for p in preds_idx]
    sub = sub.sort_values('id').reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    path = Path(submission_dir) / f"{name}.csv"
    sub.to_csv(path, index=False)

    return sub, preds_idx
