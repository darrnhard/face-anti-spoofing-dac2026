"""
src/training/trainer.py — Training engine for the FAS competition.

Public API:
    setup_training()       — creates optimizer, scheduler, criterion, EMA model
    cutmix_batch()         — applies CutMix augmentation at the batch level
    train_one_epoch()      — runs one training epoch, returns (loss, macro_f1)
    validate()             — evaluates EMA model on val loader
    dinov2_linear_probe()  — trains a linear head on frozen DINOv2 features
    train_fold()           — full 2-stage training for one CV fold

Usage:
    from src.training.trainer import train_fold, dinov2_linear_probe

    # Optional: run linear probe before fine-tuning DINOv2
    dinov2_linear_probe(fold=0, train_df=train_df, class_weights=class_weights)

    # Train one fold
    result = train_fold(
        fold=0, model_key='convnext', train_df=train_df,
        class_weights=class_weights, exp_id='exp02'
    )
    print(result['best_f1'])
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from pathlib import Path

from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

import yaml

from src.data.dataset import FaceSpoofDataset
from src.data.augmentation import get_transforms
from src.models.registry import (
    ARCH_CONFIGS, create_model, get_param_groups, get_llrd_param_groups,
)
from src.training.losses import FocalLoss, SoftCrossEntropyLoss
from src.utils.config import (
    DEVICE, NUM_CLASSES, CLASSES, IDX_TO_CLASS, MODEL_DIR, OOF_DIR, CONFIGS_DIR,
)


# =============================================================================
# Optimizer / Scheduler / Loss / EMA Setup
# =============================================================================

def setup_training(model, model_key, class_weights):
    """
    Creates the full training setup for one architecture.

    Reads all hyperparameters from ARCH_CONFIGS[model_key].

    Parameter groups:
    - LLRD architectures (ConvNeXt, EVA-02, DINOv2): layer-wise / stage-wise
      LR decay via get_llrd_param_groups(). Head group always gets weight_decay=0;
      all other groups get weight_decay=1e-4.
    - EfficientNet-B4: standard 2-group split (backbone + head) with
      weight_decay=1e-4 / 0.0 respectively.

    Scheduler: CosineAnnealingWarmRestarts
      T_0=20 first restart, T_mult=2 doubles each subsequent cycle.
      Gives the model multiple chances to escape local minima.

    Parameters
    ----------
    model         : nn.Module
    model_key     : str           — key into ARCH_CONFIGS
    class_weights : torch.Tensor  — from losses.get_class_weights()

    Returns
    -------
    optimizer  : torch.optim.AdamW
    scheduler  : CosineAnnealingWarmRestarts
    criterion  : FocalLoss
    ema_model  : AveragedModel (EMA wrapper around model)
    """
    cfg = ARCH_CONFIGS[model_key]

    # ── Parameter groups ──────────────────────────────────────────────────────
    if cfg['use_llrd']:
        param_groups = get_llrd_param_groups(
            model, model_key,
            base_lr=cfg['backbone_lr'],
            head_lr=cfg['head_lr'],
            decay_factor=cfg['llrd_factor'],
        )
        # Head group (index 0) gets no weight decay; backbone groups keep 1e-4
        for i, group in enumerate(param_groups):
            group['weight_decay'] = 0.0 if i == 0 else 1e-4
    else:
        backbone_params, head_params = get_param_groups(model, model_key)
        param_groups = [
            {'params': backbone_params, 'lr': cfg['backbone_lr'], 'weight_decay': 1e-4},
            {'params': head_params,     'lr': cfg['head_lr'],     'weight_decay': 0.0},
        ]

    optimizer = torch.optim.AdamW(param_groups)

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7,
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = FocalLoss(alpha=class_weights, gamma=1.0)

    # ── EMA ───────────────────────────────────────────────────────────────────
    # use_buffers=True ensures BN running_mean/running_var are also EMA'd
    # (critical for EfficientNet-B4 which uses BatchNorm; no-op for LayerNorm models)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999), use_buffers=True)

    return optimizer, scheduler, criterion, ema_model


# =============================================================================
# CutMix
# =============================================================================

def cutmix_batch(images, labels, alpha=1.0):
    """
    Applies CutMix augmentation to a single batch in-place.

    CutMix pastes a random rectangular patch from one image onto another,
    blending their labels proportionally to the patch area. Stronger
    regularisation than MixUp because it preserves local spatial structure.

    Parameters
    ----------
    images : torch.Tensor — shape (B, C, H, W)
    labels : torch.Tensor — shape (B,), integer class indices
    alpha  : float        — Beta distribution parameter (1.0 = uniform)

    Returns
    -------
    images   : torch.Tensor — mixed images (same shape, in-place)
    labels_a : torch.Tensor — original labels
    labels_b : torch.Tensor — labels of the pasted patches
    lam      : float        — mixing coefficient (area of original kept)
    """
    B          = images.size(0)
    indices    = torch.randperm(B)
    lam        = np.random.beta(alpha, alpha)
    _, _, H, W = images.shape

    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio); cut_h = int(H * cut_ratio)
    cx = np.random.randint(W);  cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W); x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H); y2 = np.clip(cy + cut_h // 2, 0, H)

    images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return images, labels, labels[indices], lam


# =============================================================================
# BatchNorm Helper
# =============================================================================

def set_bn_eval(module):
    """
    Keeps BatchNorm layers in eval mode during fine-tuning.

    Why this matters:
    When model.train() is called, ALL layers enter training mode including
    BatchNorm. In training mode, BatchNorm recomputes running_mean and
    running_var from each mini-batch. On a ~1,000-image fold with batch
    size 24-64, those mini-batch statistics are noisier and less
    representative than the ImageNet statistics baked in during pretraining.
    Keeping BN in eval mode preserves the pretrained statistics so the
    backbone features stay stable.

    Applied to: EfficientNet-B4 (has BatchNorm2d throughout).
    Not needed for: ConvNeXt, EVA-02, DINOv2 (all use LayerNorm — unaffected
    by this call since the isinstance check finds nothing to put in eval mode).
    Applying it universally is therefore safe and avoids any model_key logic here.

    Usage:
        model.train()
        model.apply(set_bn_eval)   # overrides BN layers back to eval
    """
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        module.eval()


# =============================================================================
# Per-Epoch Training and Validation
# =============================================================================
def train_one_epoch(model, ema_model, loader, criterion, optimizer, scaler,
                    use_cutmix=True, soft_criterion=None):
    """
    Runs one full training epoch.

    When soft_criterion is provided (pseudo-label training):
      - Expects loader to yield 5-tuple: (images, label_idx, soft_probs, is_pseudo, confidence)
      - Real rows (is_pseudo=0) → criterion (FocalLoss) on hard integer labels
      - Pseudo rows (is_pseudo=1) → soft_criterion (SoftCrossEntropyLoss) on
        prob vectors, weighted by confidence
      - CutMix is disabled in this mode to avoid mixing soft label types

    When soft_criterion is None (standard training):
      - Expects loader to yield 2-tuple: (images, labels)
      - CutMix applied with 50% probability if use_cutmix=True
    """
    model.train()
    model.apply(set_bn_eval)
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        if soft_criterion is not None:
            # ── Soft label path (pseudo-label training) ───────────────────────
            images, label_idx, soft_probs, is_pseudo_flag, confidence = batch
            images         = images.to(DEVICE)
            label_idx      = label_idx.to(DEVICE)
            soft_probs     = soft_probs.to(DEVICE)
            is_pseudo_flag = is_pseudo_flag.bool().to(DEVICE)
            confidence     = confidence.to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda"):
                logits  = model(images)
                real_m  = ~is_pseudo_flag
                pseu_m  =  is_pseudo_flag
                loss    = torch.tensor(0.0, device=DEVICE)
                if real_m.any():
                    loss = loss + criterion(logits[real_m], label_idx[real_m])
                if pseu_m.any():
                    loss = loss + soft_criterion(
                        logits[pseu_m],
                        soft_probs[pseu_m],
                        confidence[pseu_m],
                    )

            batch_labels_np = label_idx.cpu().numpy()

        else:
            # ── Hard label path (standard training) ───────────────────────────
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            use_mix = use_cutmix and np.random.rand() < 0.5
            if use_mix:
                images, labels_a, labels_b, lam = cutmix_batch(images, labels)

            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(images)
                if use_mix:
                    loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
                else:
                    loss = criterion(logits, labels)

            batch_labels_np = (
                labels.cpu().numpy() if not use_mix else labels_a.cpu().numpy()
            )

        # ── Backward + step (identical for both paths) ────────────────────────
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        ema_model.update_parameters(model)

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(batch_labels_np)

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, macro_f1

def validate(model, loader, criterion):
    """
    Evaluates the model (typically EMA) on a validation DataLoader.

    Parameters
    ----------
    model     : nn.Module (use ema_model for production evaluation)
    loader    : DataLoader
    criterion : loss function

    Returns
    -------
    avg_loss  : float
    macro_f1  : float
    all_preds : np.ndarray — predicted class indices
    all_labels: np.ndarray — true class indices
    all_probs : np.ndarray — shape (N, NUM_CLASSES), softmax probabilities
    """
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with autocast("cuda"):
                logits = model(images)
                loss   = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            total_loss  += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss  = total_loss / len(loader)
    macro_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    all_probs = np.concatenate(all_probs, axis=0)
    return avg_loss, macro_f1, np.array(all_preds), np.array(all_labels), all_probs


# =============================================================================
# DINOv2 Linear Probe
# =============================================================================

def dinov2_linear_probe(fold, train_df, class_weights, model_dir=None, epochs=20):
    """
    Trains a linear classifier on frozen DINOv2 features.

    Why a separate probe step:
    DINOv2's self-supervised features are the best off-the-shelf visual features
    available. But they are fragile on small datasets — aggressive fine-tuning
    early on collapses the feature space. This probe:
      1. Finds the optimal linear separator on frozen features (fast, ~5 min/fold)
      2. Saves the trained head weights to {model_dir}/probe/
      3. train_fold() loads these weights before full fine-tuning, giving Stage 1
         a much better starting point than random initialisation.

    Parameters
    ----------
    fold          : int
    train_df      : pd.DataFrame
    class_weights : torch.Tensor
    model_dir     : Path | None — defaults to MODEL_DIR from config
    epochs        : int         — probe training epochs (default: 20)

    Returns
    -------
    best_f1 : float — best val Macro F1 during probe training
    """
    if model_dir is None:
        model_dir = MODEL_DIR

    probe_dir = Path(model_dir) / 'probe'
    probe_dir.mkdir(parents=True, exist_ok=True)

    cfg      = ARCH_CONFIGS['dinov2']
    img_size = cfg['img_size']

    print(f"\n{'='*60}")
    print(f"DINOv2 Linear Probe — Fold {fold}")
    print(f"Frozen backbone | {epochs} epochs | lr=1e-3")
    print(f"{'='*60}")

    # Data
    train_tf, val_tf = get_transforms(img_size)
    trn_df = train_df[train_df['fold'] != fold]
    val_df = train_df[train_df['fold'] == fold]

    trn_loader = DataLoader(
        FaceSpoofDataset(trn_df, transform=train_tf, soft_labels=False),
        batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        FaceSpoofDataset(val_df, transform=val_tf, soft_labels=False),  # always hard labels
        batch_size=cfg['batch_size'], shuffle=False, num_workers=4, pin_memory=True,
    )

    # Model — freeze everything except the linear head
    model = create_model('dinov2')
    for name, param in model.named_parameters():
        if name not in ('head.weight', 'head.bias'):
            param.requires_grad = False

    frozen_n    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Frozen:    {frozen_n / 1e6:.1f}M params")
    print(f"  Trainable: {trainable_n / 1e3:.1f}K params (head only)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=0.0,
    )
    criterion = FocalLoss(alpha=class_weights, gamma=1.0)
    scaler    = GradScaler("cuda")
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999), use_buffers=True)

    best_f1 = 0.0
    print(f"\n  Training linear probe...")
    for epoch in range(epochs):
        model.train()
        model.apply(set_bn_eval)   # no-op for DINOv2 (LayerNorm only); consistent with train_one_epoch
        total_loss, all_preds, all_labels = 0.0, [], []
        for images, labels in trn_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            ema_model.update_parameters(model)
            total_loss  += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        tr_loss = total_loss / len(trn_loader)
        tr_f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        vl_loss, vl_f1, _, _, _ = validate(ema_model, val_loader, criterion)

        star = " ★" if vl_f1 > best_f1 else ""
        print(f"    Epoch {epoch+1:02d}/{epochs} | "
              f"Train: {tr_loss:.4f}/{tr_f1:.4f} | Val: {vl_loss:.4f}/{vl_f1:.4f}{star}")

        if vl_f1 > best_f1:
            best_f1 = vl_f1
            probe_path = probe_dir / f"dinov2_probe_fold{fold}.pth"
            torch.save({
                'head_state_dict': {
                    'head.weight': ema_model.module.state_dict()['head.weight'],
                    'head.bias':   ema_model.module.state_dict()['head.bias'],
                },
                'fold':   fold,
                'val_f1': vl_f1,
            }, probe_path)

    print(f"\n  Linear probe best Val F1: {best_f1:.4f}")
    print(f"  Head weights saved → {probe_dir}/dinov2_probe_fold{fold}.pth")
    print(f"  → Will be auto-loaded by train_fold('dinov2', ...)")

    del model, ema_model
    torch.cuda.empty_cache()
    return best_f1


# =============================================================================
# Full Fold Training
# =============================================================================

def train_fold(fold, model_key, train_df, class_weights,
               use_cutmix=True, use_sampler=False, exp_id='exp02',
               model_dir=None, oof_dir=None):
    """
    Full 2-stage training for a single cross-validation fold.

    Stage 1 — Head-only warmup:
        Freezes the backbone. Only the classification head is trained.
        No CutMix (clean gradients help the head converge faster).
        Duration: ARCH_CONFIGS[model_key]['freeze_epochs']

    Stage 2 — Full fine-tuning:
        Unfreezes all layers. Rebuilds optimizer + scheduler for full model.
        First warmup_ep epochs: linear LR warmup (10% → 100% of target LR).
        warmup_ep is read from ARCH_CONFIGS[model_key]['warmup_epochs']:
          CNNs (ConvNeXt, EfficientNet-B4) = 3 epochs
          ViTs (EVA-02, DINOv2, SwinV2, MaxViT, FSFM) = 5 epochs
        Remaining epochs: CosineAnnealingWarmRestarts.
        Duration: epochs - freeze_epochs

    Checkpointing:
        Best EMA model (by val Macro F1) saved to:
            {model_dir}/{exp_id}/{model_key}_fold{fold}_f1{score:.4f}.pth
        Config snapshot saved alongside for reproducibility:
            {model_dir}/{exp_id}/{model_key}_fold{fold}_config.yaml

    OOF predictions:
        Val set predictions from the best checkpoint saved to:
            {oof_dir}/{exp_id}/oof_{model_key}_fold{fold}.csv

    DINOv2 / FSFM special handling:
        If dinov2_linear_probe() was run beforehand, the saved probe head
        weights are loaded into the model before Stage 1 — giving Stage 1
        a much better starting point than random initialisation.

    Parameters
    ----------
    fold          : int
    model_key     : str           — key into ARCH_CONFIGS
    train_df      : pd.DataFrame  — must have 'fold', 'crop_path', 'label', 'label_idx'
    class_weights : torch.Tensor  — from losses.get_class_weights()
    use_cutmix    : bool          — enable CutMix in Stage 2 (default: True)
    use_sampler   : bool          — use WeightedRandomSampler to balance class frequencies
                                    per batch (default: False). Recommended for new
                                    architectures (swinv2, maxvit, fsfm) where macro F1 on
                                    minority classes (fake_printed: 104 samples) matters most.
    exp_id        : str           — experiment identifier for file naming (e.g. 'exp05')
    model_dir     : Path | None   — defaults to MODEL_DIR from config
    oof_dir       : Path | None   — defaults to OOF_DIR from config

    Returns
    -------
    dict with keys:
        best_f1    : float
        val_preds  : np.ndarray
        val_labels : np.ndarray
        val_probs  : np.ndarray  — shape (N, NUM_CLASSES)
    """
    if model_dir is None:
        model_dir = MODEL_DIR
    if oof_dir is None:
        oof_dir = OOF_DIR

    # Create experiment subdirectories
    exp_model_dir = Path(model_dir) / exp_id
    exp_oof_dir   = Path(oof_dir)   / exp_id
    exp_model_dir.mkdir(parents=True, exist_ok=True)
    exp_oof_dir.mkdir(parents=True,   exist_ok=True)

    cfg        = ARCH_CONFIGS[model_key]
    img_size   = cfg['img_size']
    batch_size = cfg['batch_size']
    num_epochs = cfg['epochs']
    freeze_ep  = cfg['freeze_epochs']
    patience   = cfg['patience']
    warmup_ep  = cfg['warmup_epochs']   # CNNs=3, ViTs=5 (per plan §5 LR table)

    print(f"\n{'='*60}")
    print(f"FOLD {fold} — {model_key} ({cfg['timm_name']})")
    print(f"Epochs: {num_epochs} | Freeze: {freeze_ep} | Warmup: {warmup_ep} | CutMix: {use_cutmix}")
    print(f"{'='*60}")

    # ── Data ──────────────────────────────────────────────────────────────────
    # patch_crop: zoom into 50-100% sub-region of the face crop for new architectures.
    # Forces the model to detect local spoof artifacts (moiré, print edges, mask seams)
    # rather than relying on global face appearance.
    _new_arch_keys = {'swinv2', 'maxvit', 'fsfm'}
    patch_crop = model_key in _new_arch_keys
    train_tf, val_tf = get_transforms(img_size, patch_crop=patch_crop)

    trn_df = train_df[(train_df['fold'] != fold) | (train_df['fold'] == -1)]
    val_df = train_df[train_df['fold'] == fold]

    # Auto-detect soft label mode: True when train_pseudo.csv is loaded
    _has_pseudo = 'is_pseudo' in train_df.columns
    trn_dataset = FaceSpoofDataset(trn_df, transform=train_tf, soft_labels=_has_pseudo)

    if use_sampler:
        # WeightedRandomSampler: each sample is drawn with probability proportional to
        # its class weight, effectively oversampling minority classes (e.g. fake_printed).
        # replacement=True is required when num_samples == len(dataset).
        from torch.utils.data import WeightedRandomSampler
        sample_weights = class_weights.cpu()[trn_df['label_idx'].values]
        sampler = WeightedRandomSampler(
            weights=sample_weights.float(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        trn_loader = DataLoader(
            trn_dataset, batch_size=batch_size,
            sampler=sampler, num_workers=4, pin_memory=True,
        )
    else:
        trn_loader = DataLoader(
            trn_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True,
        )

    val_loader = DataLoader(
        FaceSpoofDataset(val_df, transform=val_tf, soft_labels=False),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    # ── Model setup ───────────────────────────────────────────────────────────
    model = create_model(model_key)

    # DINOv2 / FSFM: load linear probe head weights if they exist.
    # FSFM uses the same probe infrastructure as DINOv2 (face-specific features are fragile;
    # a short linear probe stabilises the head before full fine-tuning).
    _probe_archs = {'dinov2', 'fsfm'}
    if model_key in _probe_archs:
        probe_subdir = f"{model_key}_probe" if model_key == 'fsfm' else 'probe'
        probe_path = Path(model_dir) / probe_subdir / f"{model_key}_probe_fold{fold}.pth"
        if probe_path.exists():
            probe_ckpt = torch.load(probe_path, weights_only=False)
            model.load_state_dict(probe_ckpt['head_state_dict'], strict=False)
            print(f"\n  Loaded linear probe head → {probe_path.name}")
        else:
            print(f"\n  [WARN] No probe weights found at {probe_path}")
            print(f"         Run dinov2_linear_probe(fold={fold}, ...) first for best results.")

    optimizer, scheduler, criterion, ema_model = setup_training(model, model_key, class_weights)
    scaler = GradScaler("cuda")

    # Create soft criterion if pseudo-labeled data is present
    _soft_criterion = SoftCrossEntropyLoss().to(DEVICE) if _has_pseudo else None

    total_n    = sum(p.numel() for p in model.parameters())
    _, head_ps = get_param_groups(model, model_key)
    head_n     = sum(p.numel() for p in head_ps)
    print(f"\n  Total:    {total_n / 1e6:.1f}M params")
    print(f"  Head:     {head_n / 1e3:.1f}K params (lr={cfg['head_lr']:.0e})")
    print(f"  Backbone: {(total_n - head_n) / 1e6:.1f}M params (lr={cfg['backbone_lr']:.0e})")
    print(f"  EMA:      decay=0.999")

    # ── Stage 1: Head-only ────────────────────────────────────────────────────
    head_ids = {id(p) for p in get_param_groups(model, model_key)[1]}
    for param in model.parameters():
        param.requires_grad = id(param) in head_ids

    print(f"\n  Stage 1: Head-only ({freeze_ep} epochs, no CutMix)")
    for epoch in range(freeze_ep):
        tr_loss, tr_f1 = train_one_epoch(
            model, ema_model, trn_loader, criterion, optimizer, scaler, use_cutmix=False, soft_criterion=_soft_criterion,
        )
        vl_loss, vl_f1, _, _, _ = validate(ema_model, val_loader, criterion)
        print(f"    Epoch {epoch+1}/{freeze_ep} | "
              f"Train: {tr_loss:.4f}/{tr_f1:.4f} | Val: {vl_loss:.4f}/{vl_f1:.4f}")

    # ── Stage 2: Full fine-tuning ─────────────────────────────────────────────
    for param in model.parameters():
        param.requires_grad = True

    # Rebuild optimizer + scheduler for the full model
    optimizer, scheduler, criterion, ema_model = setup_training(model, model_key, class_weights)
    scaler = GradScaler("cuda")

    # Create soft criterion if pseudo-labeled data is present
    _soft_criterion = SoftCrossEntropyLoss().to(DEVICE) if _has_pseudo else None

    # Warmup: linearly ramp LR from 10% → 100% over warmup_ep epochs
    warmup_target_lrs = [g['lr'] for g in optimizer.param_groups]
    for g in optimizer.param_groups:
        g['lr'] = g['lr'] * 0.1

    print(f"\n  Stage 2: Full fine-tuning ({num_epochs - freeze_ep} epochs, CutMix={use_cutmix})")
    best_f1, best_epoch, patience_counter = 0.0, 0, 0
    ckpt_path = None

    for epoch in range(freeze_ep, num_epochs):
        stage2_ep = epoch - freeze_ep   # 0-indexed within Stage 2

        # Linear warmup during first warmup_ep epochs
        if stage2_ep < warmup_ep:
            scale = 0.1 + 0.9 * (stage2_ep + 1) / warmup_ep
            for g, target_lr in zip(optimizer.param_groups, warmup_target_lrs):
                g['lr'] = target_lr * scale

        tr_loss, tr_f1 = train_one_epoch(
            model, ema_model, trn_loader, criterion, optimizer, scaler, use_cutmix=use_cutmix, soft_criterion=_soft_criterion,
        )
        vl_loss, vl_f1, vl_preds, vl_labels, vl_probs = validate(
            ema_model, val_loader, criterion
        )

        # Step scheduler only after warmup completes
        if stage2_ep >= warmup_ep:
            scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        star   = " ★" if vl_f1 > best_f1 else ""
        print(f"    Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train: {tr_loss:.4f}/{tr_f1:.4f} | "
              f"Val: {vl_loss:.4f}/{vl_f1:.4f} | "
              f"LR: {lr_now:.2e}{star}")

        if vl_f1 > best_f1:
            best_f1          = vl_f1
            best_epoch       = epoch + 1
            patience_counter = 0

            # Delete previous best cheeckpoint before saving new one
            if ckpt_path is not None and ckpt_path.exists():
                ckpt_path.unlink()

            # Save checkpoint
            ckpt_path = exp_model_dir / f"{model_key}_fold{fold}_f1{best_f1:.4f}.pth"
            torch.save({
                'model_state_dict': ema_model.module.state_dict(),
                'fold':      fold,
                'epoch':     epoch,
                'val_f1':    vl_f1,
                'model_key': model_key,
                'exp_id':    exp_id,
                'timm_name': cfg['timm_name'],
            }, ckpt_path)

            # Save config snapshot alongside checkpoint (for reproducibility)
            exp_config_dir = CONFIGS_DIR / exp_id
            exp_config_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = exp_config_dir / f"{model_key}_fold{fold}_config.yaml"
            config_snapshot = {
                'exp_id'    : exp_id,
                'model_key' : model_key,
                'fold'      : fold,
                'best_epoch': best_epoch,
                'best_f1'   : round(float(best_f1), 6),
                'arch_config': {k: v for k, v in cfg.items()},
                'training'  : {
                    'use_cutmix'    : use_cutmix,
                    'use_sampler'   : use_sampler,
                    'patch_crop'    : patch_crop,
                    'warmup_epochs' : warmup_ep,
                    'ema_decay'     : 0.999,
                    'grad_clip_norm': 1.0,
                    'loss'          : 'focal',
                    'focal_gamma'   : 1.0,
                },
            }
            with open(snapshot_path, 'w') as f:
                yaml.dump(config_snapshot, f, default_flow_style=False, sort_keys=False)

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch + 1} (patience={patience})")
            break

    print(f"\n  Best: Epoch {best_epoch}, Val F1: {best_f1:.4f}")

    # ── OOF predictions from best checkpoint ─────────────────────────────────
    ckpt        = torch.load(ckpt_path, weights_only=False)
    model_clean = create_model(model_key)
    model_clean.load_state_dict(ckpt['model_state_dict'])
    model_clean.eval()

    _, _, vl_preds, vl_labels, vl_probs = validate(model_clean, val_loader, criterion)

    oof_df = val_df[['crop_path', 'label', 'label_idx', 'fold']].copy()
    for i, cls in enumerate(CLASSES):
        oof_df[f'prob_{cls}'] = vl_probs[:, i]
    oof_df['pred_idx'] = vl_preds
    oof_df['pred']     = [IDX_TO_CLASS[p] for p in vl_preds]

    oof_path = exp_oof_dir / f"oof_{model_key}_fold{fold}.csv"
    oof_df.to_csv(oof_path, index=False)

    del model, ema_model, model_clean
    torch.cuda.empty_cache()

    return {
        'best_f1'   : best_f1,
        'val_preds' : vl_preds,
        'val_labels': vl_labels,
        'val_probs' : vl_probs,
    }
