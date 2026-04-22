# Session Log — Research Implementation (exp05 New Architectures)

**Competition:** FindIT-DAC 2026 — Face Anti-Spoofing (6-class, Macro F1)  
**Session date:** 2026-04-07  
**Status:** ✅ Complete (code), ⏳ Pending remote training

---

## Context Going Into This Session

- exp04 training complete: 4 architectures × 5 folds = 20 models
- Best OOF F1: **0.9365** (Top-3 ensemble: DINOv2 + EVA-02 + ConvNeXt)
- LB score: **0.71238** — 17-point CV-LB gap suggests cross-dataset generalization problem
- EfficientNet-B4 excluded from primary ensemble (ECE = 0.091)
- All src/ bugs fixed in session 02; codebase ready for new experiments
- Research paper read: *"Pretrained Backbones for Face Anti-Spoofing Competitions with Small, Imbalanced RGB Datasets"*

---

## Research Paper Analysis

### What the paper recommends (full ranking)

| Tier | Architecture | Reason |
|---|---|---|
| A | CLIP ViT-B/16 + LoRA | Best for low-data cross-dataset FAS; LoRA prevents overfitting |
| A | FSFM ViT-B | Face-specific self-supervised pretraining; not a VLM |
| A | S-Adapter (ViT + statistical adapters) | FAS-specific texture features; complex to implement |
| B | SwinV2-Tiny/Base (IN-22k) | WFAS 2023 2nd-place; hierarchical ViT |
| B | ConvNeXt / MaxViT (IN-21k) | WFAS 2023 3rd-place; already have ConvNeXt |
| C | ResNet-34/50 | Classic workhorse; weaker than transformers |
| C | MobileNetV3-Spoof | Lightweight; only for FLOPs-constrained settings |

### Training tricks recommended (Section 6.3)

1. **Patch/region-based training** — crop to 50-75% of face to force local artifact detection
2. **WeightedRandomSampler** — explicit recommendation for minority class with macro F1
3. Data augmentation for spoof cues (already implemented in exp04)
4. Focal/margin losses (already implemented)
5. Face detection + alignment (already done via MTCNN)

### Competition constraints applied

- **CLIP excluded**: VLM not allowed by competition rules
- **S-Adapter excluded**: Engineering complexity too high for competition timeline
- **Pseudo-label deferred**: User handling separately in a future session

---

## What We Decided to Implement

| # | Feature | Priority | Rationale |
|---|---|---|---|
| 1 | FSFM ViT-B | High | Tier A; face-specific pretraining; not VLM; same ViT-B arch as existing models |
| 2 | SwinV2-Base | High | Tier B; timm available; WFAS 2023 winner; IN-22k; easy to integrate |
| 3 | MaxViT-Base | Medium | Tier B; timm available; IN-21k; hybrid CNN/ViT; diversity from SwinV2 |
| 4 | WeightedRandomSampler | High | Paper explicitly recommends; fake_printed has only 104 samples |
| 5 | patch_crop augmentation | Medium | Research Section 6.3; forces local spoof artifact detection |

---

## Model Architecture Research (verified with timm 1.0.26, findit-dac-new env)

### Available pretrained models confirmed

**SwinV2 (chosen):** `swinv2_base_window12_192.ms_in22k`
- 86.9M total params; 6.2K head params (`head.fc.weight/bias`)
- `model.layers` (len=4, Swin stages) → handled by existing ViT LLRD branch
- Input: 192px (window12 requires this)
- Layer sizes: [0.40M, 1.72M, 57.43M, 27.33M]

**MaxViT (chosen):** `maxvit_base_tf_224.in21k`
- 118.7M total params; 596.7K head params (NormMlpClassifierHead: norm + pre_logits.fc + fc)
- `model.stages` (len=4) → new MaxViT branch added to `get_llrd_param_groups()`
- Input: 224px; `model.norm` is Identity (no overlap with head)
- IN-21k only (not fine-tuned) chosen for better cross-domain generalization

**FSFM (architecture base):** `vit_base_patch16_224`
- Standard timm ViT-B — same parameter naming as DINOv2/EVA-02 (`head.weight/head.bias`)
- FSFM weights loaded separately with `strict=False` (head stays randomly initialized)
- `model.blocks` (len=12) → reuses existing ViT LLRD branch unchanged

### Other models considered but not chosen

| Model | Reason not chosen |
|---|---|
| `swinv2_large_window12_192.ms_in22k` | Larger; higher overfitting risk on 1.4k dataset |
| `swinv2_base_window12to16_192to256.ms_in22k_ft_in1k` | IN-1k fine-tuned (less diverse than pure IN-22k) |
| `maxvit_small_tf_224.in1k` | IN-1k only; weaker than base IN-21k |
| `maxvit_base_tf_384.in21k_ft_in1k` | 384px input — too expensive |

---

## Files Changed

### `src/data/augmentation.py`

**Change:** Added `patch_crop` parameter to `get_transforms()`.

```python
# Before
def get_transforms(img_size=224):
    train_transform = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        ...
    ])

# After
def get_transforms(img_size=224, patch_crop=False):
    crop_scale = (0.5, 1.0) if patch_crop else (0.8, 1.0)
    train_transform = T.Compose([
        T.RandomResizedCrop(img_size, scale=crop_scale),
        ...
    ])
```

- `patch_crop=False` by default → all existing exp04 behavior preserved
- `patch_crop=True` enabled for `swinv2`, `maxvit`, `fsfm` in trainer.py

---

### `src/models/registry.py`

**Added imports:** `import torch`, `from pathlib import Path`

**Added 3 entries to `ARCH_CONFIGS`:**

| Key | timm_name | img_size | batch_size | backbone_lr | head_lr | LLRD | Notes |
|---|---|---|---|---|---|---|---|
| `swinv2` | `swinv2_base_window12_192.ms_in22k` | 192 | 64 | 2e-5 | 5e-4 | 0.80 | |
| `maxvit` | `maxvit_base_tf_224.in21k` | 224 | 32 | 2e-5 | 5e-4 | 0.85 | stage-wise |
| `fsfm` | `vit_base_patch16_224` | 224 | 48 | 2e-5 | 1e-3 | 0.80 | + `fsfm_ckpt`, `probe_epochs=20` |

**Extended `create_model()`:** Added FSFM case — creates `vit_base_patch16_224` with `pretrained=False`, then loads FSFM checkpoint with `strict=False`. Prints key loading summary. Gracefully warns if checkpoint not set.

**Extended `get_param_groups()`:** Added cases for `swinv2`, `maxvit`; added `'fsfm'` to existing `eva02/dinov2` branch.

| Architecture | Head condition | Reason |
|---|---|---|
| `swinv2` | `name.startswith('head.fc')` | ClassifierHead wraps Linear as `head.fc` |
| `maxvit` | `name.startswith('head.')` | NormMlpClassifierHead: norm + pre_logits + fc all in head |
| `fsfm` | `name in ('head.weight', 'head.bias')` | Plain Linear, same as DINOv2 |

**Extended `get_llrd_param_groups()`:**
- Added `elif model_key == 'maxvit':` branch — stage-wise LLRD over `model.stages` (4 stages)
- Fixed ViT branch `else:` — head detection now covers both plain Linear (`head.weight/bias`) and SwinV2's ClassifierHead (`head.fc.*`)

---

### `src/training/trainer.py`

**Change 1 — `use_sampler` parameter:**
```python
# Signature change
def train_fold(fold, model_key, train_df, class_weights,
               use_cutmix=True, use_sampler=False, exp_id='exp02', ...):

# DataLoader construction now branches on use_sampler
if use_sampler:
    sample_weights = class_weights.cpu()[trn_df['label_idx'].values]
    sampler = WeightedRandomSampler(weights=sample_weights.float(),
                                    num_samples=len(sample_weights),
                                    replacement=True)
    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, sampler=sampler, ...)
else:
    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, ...)
```

**Change 2 — `patch_crop` auto-detection:**
```python
_new_arch_keys = {'swinv2', 'maxvit', 'fsfm'}
patch_crop = model_key in _new_arch_keys
train_tf, val_tf = get_transforms(img_size, patch_crop=patch_crop)
```

**Change 3 — FSFM probe loading:**
Generalized DINOv2 probe loading to also handle `fsfm`. Uses separate `fsfm_probe/` subdirectory to avoid conflicts with DINOv2's `probe/`.

**Change 4 — Config snapshot:**
Added `use_sampler` and `patch_crop` to saved YAML config for reproducibility.

---

### `notebooks/experiments/exp05-new-architectures.ipynb` *(new)*

Mirrors exp04 structure. 5 sections:

| Section | Content |
|---|---|
| 1 — Setup | Imports, data load, class weights, EXP_ID = 'exp05' |
| 1b — FSFM setup | Checkpoint path setup + key inspection cell (conditional) |
| 1c — Verification | Smoke-check new arch param counts and LLRD group LRs |
| 2 — SwinV2 | 5-fold training with `use_sampler=True` |
| 3 — MaxViT | 5-fold training with `use_sampler=True` |
| 4 — FSFM | Linear probe + 5-fold training (skips gracefully if no checkpoint) |
| 5 — OOF eval | Per-arch OOF F1 + ensemble ablation vs 0.9365 baseline |

---

## Sanity Check Results

Verified with findit-dac-new Python before session close:

```
patch_crop=False scale: (0.8, 1.0)   ✓
patch_crop=True  scale: (0.5, 1.0)   ✓

swinv2: timm=swinv2_base_window12_192.ms_in22k  img=192  bs=64   ✓
maxvit: timm=maxvit_base_tf_224.in21k           img=224  bs=32   ✓
fsfm:   timm=vit_base_patch16_224               img=224  bs=48   ✓

swinv2: backbone=86.9M head=6.2K  LLRD groups=6
  LRs: ['5.0e-04', '2.0e-05', '1.6e-05', '1.3e-05', '1.0e-05', '8.2e-06']   ✓
maxvit: backbone=118.1M head=596.7K  LLRD groups=6
  LRs: ['5.0e-04', '2.0e-05', '1.7e-05', '1.4e-05', '1.2e-05', '1.0e-05']   ✓
```

LLRD verified: head group always highest LR (5e-4), backbone groups decay correctly toward early stages.

---

## Notes for Next Session / Before Training

### FSFM checkpoint (required before Section 4 of exp05)
1. Visit https://github.com/fsfm-3c/fsfm-3c or https://fsfm-3c.github.io
2. Download ViT-B pretrained checkpoint
3. Upload to Vast.ai: `/workspace/fas-competition/models/pretrained/fsfm_vit_b.pth`
4. Run the key inspection cell in exp05 Section 1b to confirm timm-compatible key naming
5. If keys have a prefix (e.g. `encoder.blocks.0...`), strip it with the provided dict comprehension

### Remote training sequence

```bash
./scripts/sync_to_remote.sh <IP> <PORT>
# On Vast.ai:
jupyter nbconvert --to notebook --execute notebooks/experiments/exp05-new-architectures.ipynb
# Sync results back:
./scripts/sync_from_remote.sh <IP> <PORT>
```

### Expected outcomes

| Architecture | Expected OOF F1 | Ensemble impact |
|---|---|---|
| SwinV2-Base | 0.90–0.93 | New hierarchical ViT diversity; IN-22k pretraining |
| MaxViT-Base | 0.89–0.92 | Hybrid CNN/ViT diversity; IN-21k pretraining |
| FSFM ViT-B | 0.92–0.95 | Face-specific features; potentially best single model |

Target: at least one new arch gives ensemble OOF > **0.9365** baseline.

### If FSFM is unavailable

Skip Section 4 entirely. SwinV2 + MaxViT alone add two new architectures to the ensemble
(4-5 model total). The notebook handles this gracefully with `if ARCH_CONFIGS['fsfm']['fsfm_ckpt'] is None:` guards.

---

## Files Changed (summary)

```
src/data/augmentation.py              ← patch_crop parameter added
src/models/registry.py                ← swinv2/maxvit/fsfm configs + model factory + param groups
src/training/trainer.py               ← use_sampler, patch_crop, FSFM probe, config snapshot
notebooks/experiments/
  exp05-new-architectures.ipynb       ← created (new experiment notebook)
```
