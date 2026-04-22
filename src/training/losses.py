"""
src/training/losses.py — Loss functions for the FAS competition.

Contains:
    get_class_weights()  — inverse-frequency class weights from a DataFrame
    FocalLoss            — focal loss with per-class alpha weights
    Poly1Loss            — polynomial cross-entropy, alternative to focal loss

Usage:
    from src.training.losses import FocalLoss, Poly1Loss, get_class_weights

    class_weights = get_class_weights(train_df, device)
    criterion = FocalLoss(alpha=class_weights, gamma=1.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import CLASSES, NUM_CLASSES


# =============================================================================
# Class Weight Helper
# =============================================================================

def get_class_weights(train_df, device):
    """
    Computes inverse-frequency class weights from the full training DataFrame.

    Formula: weight_c = N / (num_classes * count_c)
    This gives more weight to rare classes (e.g. fake_printed with only 75 samples)
    and less weight to frequent classes (e.g. realperson with 385 samples).

    Computed from the FULL train_df (not fold-specific) for consistency
    across all folds.

    Parameters
    ----------
    train_df : pd.DataFrame — must have a 'label' column with class name strings
    device   : str | torch.device

    Returns
    -------
    class_weights : torch.FloatTensor of shape (NUM_CLASSES,), on device
    """
    n_total      = len(train_df)
    class_counts = train_df['label'].value_counts()
    weights      = [
        n_total / (NUM_CLASSES * class_counts[c])
        for c in CLASSES
    ]
    return torch.FloatTensor(weights).to(device)


# =============================================================================
# Loss Functions
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss with per-class alpha weights.

    Focal loss down-weights easy / well-classified examples and focuses
    training on hard ones. Used with gamma=1.0 (down from 2.0 in v1).

    Why gamma=1.0 instead of the standard 2.0:
    gamma=2.0 is very aggressive — it heavily suppresses 'easy' examples.
    On a 1342-image dataset where fake_printed has only 75 samples, we
    cannot afford to discard ANY signal. gamma=1.0 still focuses on hard
    examples but preserves more learning signal from well-classified ones.

    Parameters
    ----------
    alpha : torch.Tensor | None — per-class weights of shape (num_classes,)
    gamma : float               — focusing parameter; 0.0 = standard CE
    """

    def __init__(self, alpha=None, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-class weights (class_weights tensor)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt      = torch.exp(-ce_loss)                     # predicted probability for true class
        focal   = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


class Poly1Loss(nn.Module):
    """
    Poly-1 loss: CE + epsilon * (1 - pt)

    Why this exists:
    Recent work shows Poly-1 consistently outperforms both CE and Focal
    on small datasets. The epsilon term gently penalizes low-confidence
    correct predictions without the aggressive discarding that focal loss
    does. Available as an alternative — swap in via setup_training(loss='poly1').

    Reference: Leng et al., "PolyLoss: A Polynomial Expansion Perspective
    of Classification Loss Functions", ICLR 2022.

    Parameters
    ----------
    num_classes : int
    epsilon     : float — weight of the polynomial correction term
    weight      : torch.Tensor | None — per-class weights
    """

    def __init__(self, num_classes=NUM_CLASSES, epsilon=1.0, weight=None):
        super().__init__()
        self.epsilon     = epsilon
        self.num_classes = num_classes
        self.weight      = weight

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, weight=self.weight, reduction='none')
        pt = (
            F.one_hot(labels, self.num_classes).float() *
            F.softmax(logits, dim=-1)
        ).sum(dim=-1)
        return (ce + self.epsilon * (1 - pt)).mean()

class SoftCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with soft (probability distribution) targets.

    Used for pseudo-labeled samples where the training target is the
    ensemble's probability vector rather than a one-hot hard label.
    Confidence weighting: high-confidence pseudo-labels contribute full
    loss; low-confidence ones contribute proportionally less.

    Parameters
    ----------
    logits      : torch.Tensor  shape (B, C) — raw model outputs
    soft_targets: torch.Tensor  shape (B, C) — target probability distribution
    weights     : torch.Tensor  shape (B,) | None — per-sample confidence weights
    """
    def forward(self, logits, soft_targets, weights=None):
        log_probs        = F.log_softmax(logits, dim=1)
        per_sample_loss  = -(soft_targets * log_probs).sum(dim=1)
        if weights is not None:
            per_sample_loss = per_sample_loss * weights
        return per_sample_loss.mean()
