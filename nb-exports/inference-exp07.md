# 11 — Inference & Submission — exp07

**Purpose:** Load exp07 checkpoints (25 total: 5 archs × 5 folds), run 3-view TTA inference on the
test set, build multiple ensemble strategies (including a cross-experiment mega-ensemble with exp06),
and generate 3 submission CSVs.

**Run locally** (GTX 1050 Ti) after syncing from Vast.ai:
```bash
bash scripts/sync_from_remote.sh
```

---

### Locked findings from analysis-exp07.ipynb

| Decision | Evidence | Value |
|---|---|---|
| **Primary ensemble** | Pruned mega (exp06-all5 + exp07-top3) — highest OOF diversity | `pruned_mega_argmax` |
| **Secondary ensemble** | exp07 top-4 (drop convnext): LOO Δ=+0.0053 | `top4_argmax` |
| **Baseline** | exp07 all-5 equal weight | `all5_argmax` |
| **No thresholds** | Nested CV OOB=0.9216 < argmax 0.9252; overfitting signal=+0.0108 | Argmax only |
| **Drop convnext from exp07** | LOO: dropping convnext gives +0.0053 OOF gain | Prune convnext |
| **Pruned mega composition** | LOO on mega: drop exp07/convnext (+0.0004) + exp07/effnet_b4 (+0.0013) | exp06×5 + exp07×3 |

> **⚠️ TTA upgrade:** This notebook uses **3-view TTA** (original + hflip + scale-centercrop)
> vs 2-view in exp06. The scale-centercrop view resizes to 115% then center-crops back to the
> native img_size. This is applied via an in-notebook patch — no changes to predict.py needed.
>
> **🔌 swinv2 batch size fix:** `INFER_BATCH` in predict.py was missing 'swinv2'. Patched here.

**Submission plan (3 slots):**
1. `pruned_mega_argmax` — exp06 (all-5) + exp07 (eva02 + dinov2 + swinv2) → primary
2. `top4_argmax` — exp07 top-4 (drop convnext), argmax → exp07-only best
3. `all5_argmax` — exp07 all-5 equal weight, argmax → calibration baseline


## 1. Setup


```python
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re as _re
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.metrics import f1_score

PROJECT_ROOT = Path().resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    DEVICE, TRAIN_CSV, TEST_CSV, CROP_TRAIN_DIR, CROP_TEST_DIR,
    N_FOLDS, CLASSES, NUM_CLASSES, IDX_TO_CLASS,
    MODEL_DIR, OOF_DIR, SUBMISSION_DIR,
    print_env,
)
from src.utils.seed import set_seed
from src.models.loader import load_model_from_checkpoint
import src.inference.predict as _predict_mod   # imported as module for monkey-patching
from src.inference.predict import predict_test
from src.inference.submission import make_submission

set_seed(42)
print_env()

```

    Environment  : local
    Project dir  : /home/darrnhard/ML/Competition/FindIT-DAC
      interim    : /home/darrnhard/ML/Competition/FindIT-DAC/data/interim
        train    : /home/darrnhard/ML/Competition/FindIT-DAC/data/interim/train
        test     : /home/darrnhard/ML/Competition/FindIT-DAC/data/interim/test
      processed  : /home/darrnhard/ML/Competition/FindIT-DAC/data/processed
        crops    : /home/darrnhard/ML/Competition/FindIT-DAC/data/processed/crops
        train csv: /home/darrnhard/ML/Competition/FindIT-DAC/data/processed/train_clean.csv
        test csv : /home/darrnhard/ML/Competition/FindIT-DAC/data/processed/test.csv
    Model dir    : /home/darrnhard/ML/Competition/FindIT-DAC/models
    OOF dir      : /home/darrnhard/ML/Competition/FindIT-DAC/oof
    Submissions  : /home/darrnhard/ML/Competition/FindIT-DAC/submissions
    Device       : cuda
    GPU          : NVIDIA GeForce GTX 1050 Ti
    VRAM         : 4.2 GB


### 1a. Patches: swinv2 batch size + 3-view TTA

Two runtime patches applied here so `predict.py` does not need to be edited:

1. **`INFER_BATCH['swinv2'] = 4`** — swinv2 was absent from the dict in predict.py (latent bug; exp06
   inference worked because it patched the module at notebook level too).  
2. **`_build_tta_transforms` override** — replaces the 2-view function with a 3-view version that adds
   a scale-centercrop pass (resize 115% → CenterCrop). This is applied by replacing the function
   reference in the predict module's namespace; Python resolves module-level names at call-time, so
   `predict_test` will use the patched version automatically.

**Why 3-view?** `fake_screen→realperson` is the dominant error pattern (17/41 hard-wrongs in exp07).
Screen moire artifacts are spatial-frequency signals. A scale-shifted view makes the model evaluate
the same face at a slightly different effective pixel-per-feature density, producing probability
estimates that average better across architectures. (Liu et al. 2020 *NASFAS*; Wang et al. 2022 *SSDG*.)



```python
import torchvision.transforms as T
from src.data.augmentation import IMAGENET_MEAN, IMAGENET_STD

# ── Patch 1: swinv2 batch size (bug fix) ──────────────────────────────────────
_predict_mod.INFER_BATCH['swinv2'] = 4

print('INFER_BATCH after patch:')
for k, v in _predict_mod.INFER_BATCH.items():
    print(f'  {k:<14}: batch_size={v}')

# ── Patch 2: 3-view TTA ────────────────────────────────────────────────────────
def _tta_3view(img_size: int):
    """
    3-view TTA transforms for one checkpoint.

    View 1 — original:   Resize → ToTensor → Normalize
    View 2 — hflip:      HFlip → Resize → ToTensor → Normalize
    View 3 — scale-crop: Resize(115%) → CenterCrop(img_size) → ToTensor → Normalize
        The scale-crop view zooms into the face center at a slightly different
        effective resolution. For face crops, a 15% upscale keeps the full face
        in frame (InsightFace crops contain the face with margin; 15% is well
        inside the safe zoom range). This introduces no border artifacts because
        we resize first then crop — never pad.
    """
    base_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    hflip_tf = T.Compose([
        T.RandomHorizontalFlip(p=1.0),   # p=1.0 → always flip
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    scale_tf = T.Compose([
        T.Resize((int(img_size * 1.15), int(img_size * 1.15))),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return [base_tf, hflip_tf, scale_tf]

_predict_mod._build_tta_transforms = _tta_3view
print()
print('TTA: 3-view (original + hflip + scale-centercrop) ✅')
print()

# ── Sanity: verify patch is live ───────────────────────────────────────────────
# predict_test calls _build_tta_transforms at runtime from the module namespace.
# Patching the module attribute is sufficient; no import cache issues here.
test_tfs = _predict_mod._build_tta_transforms(224)
assert len(test_tfs) == 3, 'TTA patch failed: expected 3 transforms'
print(f'Patch verified: _build_tta_transforms(224) returns {len(test_tfs)} views ✅')

```

    INFER_BATCH after patch:
      convnext      : batch_size=4
      eva02         : batch_size=4
      dinov2        : batch_size=4
      effnet_b4     : batch_size=8
      swinv2        : batch_size=4
    
    TTA: 3-view (original + hflip + scale-centercrop) ✅
    
    Patch verified: _build_tta_transforms(224) returns 3 views ✅



```python
EXP_ID    = 'exp07'
ALL_ARCHS = ['convnext', 'eva02', 'dinov2', 'effnet_b4', 'swinv2']
prob_cols = [f'prob_{cls}' for cls in CLASSES]

# ══════════════════════════════════════════════════════════════════════════════
# LOCKED FINDINGS FROM analysis-exp07.ipynb
# ══════════════════════════════════════════════════════════════════════════════
ANALYSIS = {
    # Section 2: Individual OOF Macro F1 per arch
    'oof_f1': {
        'convnext'  : 0.9073,   # Δ=-0.0111 vs exp06 ↓
        'eva02'     : 0.9293,   # Δ=+0.0026 vs exp06 ↑  ← best single arch
        'dinov2'    : 0.9254,   # Δ=-0.0067 vs exp06 ↓
        'effnet_b4' : 0.8826,   # Δ=-0.0076 vs exp06 ↓
        'swinv2'    : 0.9131,   # Δ=+0.0042 vs exp06 ↑
    },
    # Section 13: ECE (Expected Calibration Error) — all improved vs exp06
    'ece': {
        'convnext'  : 0.0452,   # ⚠️ moderate  (exp06: 0.0527)
        'eva02'     : 0.0312,   # ✅ good       (exp06: 0.0324)
        'dinov2'    : 0.0280,   # ✅ good       (exp06: 0.0422)
        'effnet_b4' : 0.0207,   # ✅ excellent  (exp06: 0.0305)
        'swinv2'    : 0.0246,   # ✅ good       (exp06: 0.0261)
        'ensemble'  : 0.0532,   # ⚠️ moderate  (exp06: 0.0642)
    },
    # Section 7: Ensemble OOF strategies
    'ensemble_oof': {
        'all5_equal'         : 0.9252,
        'top4_drop_convnext' : 0.9306,   # best exp07-only
    },
    # Section 12: Threshold decision
    'threshold': {
        'argmax'              : 0.9252,
        'naive_oof'           : 0.9324,
        'nested_cv_oob'       : 0.9216,   # OOB < argmax by 0.0036 → thresholds HURT
        'overfitting_signal'  : +0.0108,
        'expected_lb_gain'    : -0.0037,  # negative → use argmax
        'decision'            : 'NO_THRESHOLD',
    },
    # Section 7 LOO (exp07 all-5)
    'loo_exp07': {
        'convnext'  : 0.9306,  # Δ=+0.0053 ← DROP (helps)
        'eva02'     : 0.9176,  # Δ=-0.0076 ← keep
        'dinov2'    : 0.9198,  # Δ=-0.0054 ← keep
        'effnet_b4' : 0.9274,  # Δ=+0.0021 ← borderline
        'swinv2'    : 0.9261,  # Δ=+0.0009 ← borderline
    },
    # Section 14 LOO on mega-ensemble (exp06 all-5 + exp07 all-5)
    'loo_mega': {
        'mega_all10'              : 0.9306,
        'drop_exp07_convnext'     : 0.9311,  # Δ=+0.0004 → prune
        'drop_exp07_effnet_b4'    : 0.9319,  # Δ=+0.0013 → prune
        'drop_exp07_eva02'        : 0.9296,  # Δ=-0.0010 → keep
        'drop_exp07_dinov2'       : 0.9260,  # Δ=-0.0046 → keep (most valuable)
        'drop_exp07_swinv2'       : 0.9307,  # Δ=+0.0001 → borderline, keep
        'drop_exp06_convnext'     : 0.9310,  # Δ=+0.0004 → borderline, keep exp06
        'drop_exp06_eva02'        : 0.9291,  # Δ=-0.0016 → keep
        'drop_exp06_dinov2'       : 0.9269,  # Δ=-0.0037 → keep
    },
    # Pruned mega composition (drop exp07/convnext + exp07/effnet_b4 from mega)
    # = exp06 (all 5) + exp07 (eva02 + dinov2 + swinv2) = 8 models total
    'pruned_mega_archs': {
        'exp06': ['convnext', 'eva02', 'dinov2', 'effnet_b4', 'swinv2'],
        'exp07': ['eva02', 'dinov2', 'swinv2'],   # dropped convnext + effnet_b4
    },
    'best_single_arch': 'eva02',
}
# ══════════════════════════════════════════════════════════════════════════════

SUBMISSION_DIR.mkdir(exist_ok=True)
print(f'Experiment  : {EXP_ID}')
print(f'All archs   : {ALL_ARCHS}  ({len(ALL_ARCHS)*N_FOLDS} checkpoints total)')
print(f'Device      : {DEVICE}')
print(f'OOF dir     : {OOF_DIR / EXP_ID}')
print(f'Model dir   : {MODEL_DIR / EXP_ID}')
print(f'Submit dir  : {SUBMISSION_DIR}')
print()
print('Strategy summary (from analysis):')
print(f'  Primary  : pruned_mega_argmax    exp06×5 + exp07×3 (eva02, dinov2, swinv2)')
print(f'  Secondary: top4_argmax           exp07 top-4 drop convnext (OOF={ANALYSIS["ensemble_oof"]["top4_drop_convnext"]:.4f})')
print(f'  Baseline : all5_argmax           exp07 all-5 (OOF={ANALYSIS["ensemble_oof"]["all5_equal"]:.4f})')
print(f'  Threshold: {ANALYSIS["threshold"]["decision"]} (signal={ANALYSIS["threshold"]["overfitting_signal"]:+.4f}, expected Δ={ANALYSIS["threshold"]["expected_lb_gain"]:+.4f})')

```

    Experiment  : exp07
    All archs   : ['convnext', 'eva02', 'dinov2', 'effnet_b4', 'swinv2']  (25 checkpoints total)
    Device      : cuda
    OOF dir     : /home/darrnhard/ML/Competition/FindIT-DAC/oof/exp07
    Model dir   : /home/darrnhard/ML/Competition/FindIT-DAC/models/exp07
    Submit dir  : /home/darrnhard/ML/Competition/FindIT-DAC/submissions
    
    Strategy summary (from analysis):
      Primary  : pruned_mega_argmax    exp06×5 + exp07×3 (eva02, dinov2, swinv2)
      Secondary: top4_argmax           exp07 top-4 drop convnext (OOF=0.9306)
      Baseline : all5_argmax           exp07 all-5 (OOF=0.9252)
      Threshold: NO_THRESHOLD (signal=+0.0108, expected Δ=-0.0037)


## 2. Load Data & Confirm OOF Analysis

Re-load OOF CSVs from both exp07 and exp06 and recompute all ensemble F1s before committing GPU time.
Any mismatch > 0.001 vs locked analysis values stops the notebook.

**exp07 OOF scope:** 1,464 real rows (pseudo rows had `fold=-1`, never validated).  
**exp06 OOF scope:** Same 1,464 real rows aligned by crop_path.



```python
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

train_df['crop_path'] = train_df['crop_path'].apply(
    lambda p: str(CROP_TRAIN_DIR / Path(p).name)
)
test_df['crop_path'] = test_df['crop_path'].apply(
    lambda p: str(CROP_TEST_DIR / Path(p).name)
)

assert Path(train_df['crop_path'].iloc[0]).exists(), \
    f'Train crops not found — check CROP_TRAIN_DIR: {CROP_TRAIN_DIR}'
assert Path(test_df['crop_path'].iloc[0]).exists(), \
    f'Test crops not found — check CROP_TEST_DIR: {CROP_TEST_DIR}'

print(f'Train (real rows) : {len(train_df)} images')
print(f'Test              : {len(test_df)} images')

```

    Train (real rows) : 1464 images
    Test              : 404 images



```python
# ── Load exp07 OOF CSVs ────────────────────────────────────────────────────────
oof_data_07 = {}
oof_base_07 = OOF_DIR / EXP_ID

for arch in ALL_ARCHS:
    fold_dfs = []
    for fold in range(N_FOLDS):
        path = oof_base_07 / f'oof_{arch}_fold{fold}.csv'
        if path.exists():
            df = pd.read_csv(path)
            if 'crop_path' in df.columns:
                df['crop_path'] = df['crop_path'].apply(
                    lambda p: str(CROP_TRAIN_DIR / Path(p).name)
                )
            fold_dfs.append(df)
        else:
            print(f'  ⚠️  Missing: {path.name}')
    if fold_dfs:
        oof_data_07[arch] = pd.concat(fold_dfs, ignore_index=True)

print('exp07 OOF per-arch F1 verification:')
print(f'  {"Arch":<14} {"Rows":>6}  {"F1(CSV)":>9}  {"F1(locked)":>10}  {"Match":>8}')
print('  ' + '-' * 58)
for arch in ALL_ARCHS:
    df      = oof_data_07[arch]
    f1_csv  = f1_score(df['label_idx'], df['pred_idx'], average='macro')
    locked  = ANALYSIS['oof_f1'][arch]
    match   = '✅' if abs(f1_csv - locked) < 0.002 else f'⚠️  MISMATCH ({f1_csv:.4f})'
    print(f'  {arch:<14} {len(df):>6}  {f1_csv:>9.4f}  {locked:>10.4f}  {match}')

```

    exp07 OOF per-arch F1 verification:
      Arch             Rows    F1(CSV)  F1(locked)     Match
      ----------------------------------------------------------
      convnext         1464     0.9077      0.9073  ✅
      eva02            1464     0.9296      0.9293  ✅
      dinov2           1464     0.9257      0.9254  ✅
      effnet_b4        1464     0.8829      0.8826  ✅
      swinv2           1464     0.9136      0.9131  ✅



```python
# ── Compute ensemble OOF strategies ───────────────────────────────────────────
base_df     = oof_data_07['convnext'].sort_values('crop_path').reset_index(drop=True)
true_labels = base_df['label_idx'].values

oof_prbs_07 = {}
for arch in ALL_ARCHS:
    aligned = oof_data_07[arch].set_index('crop_path').loc[base_df['crop_path']]
    oof_prbs_07[arch] = aligned[prob_cols].values.astype('float32')

def ens_f1_subset_07(arch_list):
    avg = np.mean([oof_prbs_07[a] for a in arch_list], axis=0)
    return f1_score(true_labels, avg.argmax(axis=1), average='macro'), avg

f1_all5, _   = ens_f1_subset_07(ALL_ARCHS)
top4_archs   = [a for a in ALL_ARCHS if a != 'convnext']
f1_top4, _   = ens_f1_subset_07(top4_archs)

print('exp07 Ensemble OOF F1 verification:')
print(f'  {"Strategy":<40} {"F1(CSV)":>9}  {"F1(locked)":>10}  {"Match":>8}')
print('  ' + '-' * 72)
for name, f1_val, locked in [
    ('All-5 equal weight',       f1_all5, ANALYSIS['ensemble_oof']['all5_equal']),
    ('Top-4 (drop convnext)',     f1_top4, ANALYSIS['ensemble_oof']['top4_drop_convnext']),
]:
    match = '✅' if abs(f1_val - locked) < 0.002 else '⚠️  MISMATCH'
    print(f'  {name:<40} {f1_val:>9.4f}  {locked:>10.4f}  {match}')

print()
print('Threshold decision confirmation:')
print(f'  Argmax OOF F1        : {ANALYSIS["threshold"]["argmax"]:.4f}')
print(f'  Nested CV OOB        : {ANALYSIS["threshold"]["nested_cv_oob"]:.4f}  ← OOB < argmax by {ANALYSIS["threshold"]["argmax"] - ANALYSIS["threshold"]["nested_cv_oob"]:.4f}')
print(f'  Overfitting signal   : {ANALYSIS["threshold"]["overfitting_signal"]:+.4f}  (>0.005 → high)')
print(f'  Expected LB Δ        : {ANALYSIS["threshold"]["expected_lb_gain"]:+.4f}  ← NEGATIVE → thresholds hurt')
print(f'  Decision             : {ANALYSIS["threshold"]["decision"]}')

```

    exp07 Ensemble OOF F1 verification:
      Strategy                                   F1(CSV)  F1(locked)     Match
      ------------------------------------------------------------------------
      All-5 equal weight                          0.9252      0.9252  ✅
      Top-4 (drop convnext)                       0.9306      0.9306  ✅
    
    Threshold decision confirmation:
      Argmax OOF F1        : 0.9252
      Nested CV OOB        : 0.9216  ← OOB < argmax by 0.0036
      Overfitting signal   : +0.0108  (>0.005 → high)
      Expected LB Δ        : -0.0037  ← NEGATIVE → thresholds hurt
      Decision             : NO_THRESHOLD



```python
# ── Load exp06 OOF CSVs for cross-experiment mega-ensemble OOF check ──────────
oof_base_06 = OOF_DIR / 'exp06'
oof_data_06 = {}

for arch in ALL_ARCHS:
    fold_dfs = []
    for fold in range(N_FOLDS):
        path = oof_base_06 / f'oof_{arch}_fold{fold}.csv'
        if path.exists():
            df = pd.read_csv(path)
            if 'crop_path' in df.columns:
                df['crop_path'] = df['crop_path'].apply(
                    lambda p: str(CROP_TRAIN_DIR / Path(p).name)
                )
            fold_dfs.append(df)
        else:
            print(f'  ⚠️  Missing exp06: {path.name}')
    if fold_dfs:
        oof_data_06[arch] = pd.concat(fold_dfs, ignore_index=True)

oof_prbs_06 = {}
for arch in ALL_ARCHS:
    if arch in oof_data_06:
        aligned = oof_data_06[arch].set_index('crop_path').loc[base_df['crop_path']]
        oof_prbs_06[arch] = aligned[prob_cols].values.astype('float32')

print(f'exp06 OOF loaded: {len(oof_prbs_06)} archs')
print()

# ── Verify mega-ensemble OOF and compute pruned mega OOF ──────────────────────
# Full mega: exp06 all-5 + exp07 all-5 = 10 probability arrays
all10_probs  = list(oof_prbs_07.values()) + list(oof_prbs_06.values())
mega_probs   = np.mean(all10_probs, axis=0)
f1_mega_all10 = f1_score(true_labels, mega_probs.argmax(axis=1), average='macro')

# Pruned mega: exp06 all-5 + exp07 (eva02 + dinov2 + swinv2) = 8 arrays
# Rationale: LOO on mega showed dropping exp07/convnext (+0.0004) and
# exp07/effnet_b4 (+0.0013) both improve OOF. Combined pruning targets ~+0.0013.
EXP07_KEEP  = ANALYSIS['pruned_mega_archs']['exp07']   # ['eva02', 'dinov2', 'swinv2']
pruned_prbs = ([oof_prbs_07[a] for a in EXP07_KEEP] +
               list(oof_prbs_06.values()))              # 3 + 5 = 8 arrays
pruned_mega_probs = np.mean(pruned_prbs, axis=0)
f1_pruned = f1_score(true_labels, pruned_mega_probs.argmax(axis=1), average='macro')

print('Cross-Experiment Ensemble OOF Verification:')
print(f'  {"Strategy":<50} {"OOF F1":>8}  Notes')
print('  ' + '-' * 75)
print(f'  {"exp07 all-5":<50} {f1_all5:>8.4f}  baseline (locked={ANALYSIS["ensemble_oof"]["all5_equal"]:.4f})')
print(f'  {"exp07 top-4 (drop convnext)":<50} {f1_top4:>8.4f}  best exp07-only')
print(f'  {"Mega all-10 (exp06+exp07)":<50} {f1_mega_all10:>8.4f}  locked={ANALYSIS["loo_mega"]["mega_all10"]:.4f}')
print(f'  {"Pruned mega (exp06×5 + exp07×3)":<50} {f1_pruned:>8.4f}  ← primary submission target')
print()
print(f'  exp07 pruned-mega keeps: {EXP07_KEEP} ({len(EXP07_KEEP)} exp07 + 5 exp06 = 8 total)')
print()
# Verify mega_all10 within tolerance
mega_match = '✅' if abs(f1_mega_all10 - ANALYSIS['loo_mega']['mega_all10']) < 0.002 else '⚠️  MISMATCH'
print(f'  Mega all-10 locked match: {mega_match}')

```

    exp06 OOF loaded: 5 archs
    
    Cross-Experiment Ensemble OOF Verification:
      Strategy                                             OOF F1  Notes
      ---------------------------------------------------------------------------
      exp07 all-5                                          0.9252  baseline (locked=0.9252)
      exp07 top-4 (drop convnext)                          0.9306  best exp07-only
      Mega all-10 (exp06+exp07)                            0.9306  locked=0.9306
      Pruned mega (exp06×5 + exp07×3)                      0.9349  ← primary submission target
    
      exp07 pruned-mega keeps: ['eva02', 'dinov2', 'swinv2'] (3 exp07 + 5 exp06 = 8 total)
    
      Mega all-10 locked match: ✅


## 3. Model Checkpoint Sanity Check

Verify all 25 exp07 checkpoints load correctly before committing to inference.


```python
import torch

print(f'Device: {DEVICE}')
print(f'Checking {len(ALL_ARCHS) * N_FOLDS} checkpoints...\n')

load_status = {}
for arch in ALL_ARCHS:
    arch_ok = True
    for fold in range(N_FOLDS):
        try:
            m = load_model_from_checkpoint(arch, fold=fold, exp_id=EXP_ID)
            params = sum(p.numel() for p in m.parameters()) / 1e6
            del m
            torch.cuda.empty_cache()
            print(f'  {arch} fold {fold}: {params:.1f}M params ✅')
        except FileNotFoundError as e:
            print(f'  {arch} fold {fold}: ✗ MISSING — {e}')
            arch_ok = False
        except Exception as e:
            print(f'  {arch} fold {fold}: ✗ ERROR — {e}')
            arch_ok = False
    load_status[arch] = arch_ok
    print()

print('Load status:')
all_ok = True
for arch, ok in load_status.items():
    status = '✅ all folds OK' if ok else '❌ MISSING/ERROR'
    print(f'  {arch:<14}: {status}')
    if not ok:
        all_ok = False

if not all_ok:
    raise RuntimeError(
        'Fix missing checkpoints before inference:\n'
        '  bash scripts/sync_from_remote.sh'
    )
else:
    print('\n✅ All 25 checkpoints verified. Safe to proceed with inference.')

```

    Device: cuda
    Checking 25 checkpoints...
    
      convnext fold 0: 87.6M params ✅
      convnext fold 1: 87.6M params ✅
      convnext fold 2: 87.6M params ✅
      convnext fold 3: 87.6M params ✅
      convnext fold 4: 87.6M params ✅
    
      eva02 fold 0: 85.8M params ✅
      eva02 fold 1: 85.8M params ✅
      eva02 fold 2: 85.8M params ✅
      eva02 fold 3: 85.8M params ✅
      eva02 fold 4: 85.8M params ✅
    
      dinov2 fold 0: 85.7M params ✅
      dinov2 fold 1: 85.7M params ✅
      dinov2 fold 2: 85.7M params ✅
      dinov2 fold 3: 85.7M params ✅
      dinov2 fold 4: 85.7M params ✅
    
      effnet_b4 fold 0: 17.6M params ✅
      effnet_b4 fold 1: 17.6M params ✅
      effnet_b4 fold 2: 17.6M params ✅
      effnet_b4 fold 3: 17.6M params ✅
      effnet_b4 fold 4: 17.6M params ✅
    
      swinv2 fold 0: 86.9M params ✅
      swinv2 fold 1: 86.9M params ✅
      swinv2 fold 2: 86.9M params ✅
      swinv2 fold 3: 86.9M params ✅
      swinv2 fold 4: 86.9M params ✅
    
    Load status:
      convnext      : ✅ all folds OK
      eva02         : ✅ all folds OK
      dinov2        : ✅ all folds OK
      effnet_b4     : ✅ all folds OK
      swinv2        : ✅ all folds OK
    
    ✅ All 25 checkpoints verified. Safe to proceed with inference.


## 4. Test Inference with 3-View TTA

25 models × 3 TTA views = 75 forward passes on 404 test images.

**TTA views:**
- View 1: Original — `Resize(img_size) → ToTensor → Normalize`
- View 2: H-flip   — `HFlip → Resize(img_size) → ToTensor → Normalize`
- View 3: Scale-crop — `Resize(img_size×1.15) → CenterCrop(img_size) → ToTensor → Normalize`

**Memory note (GTX 1050 Ti, 4.2 GB VRAM):** One model at a time, deleted after inference.
SwinV2 uses 192px (smaller than 224px). Estimated runtime: **~35–45 min** (vs ~25 min at 2-view TTA).

**Why swinv2 uses batch_size=4:** 192px × 86.9M params. At batch=8, the 3rd TTA view
(which resizes to 220px before CenterCrop) is still well within VRAM limits, but we keep
batch=4 to match the conservative GTX 1050 Ti profile.



```python
print('Starting test inference (3-view TTA)...')
print(f'Test set  : {len(test_df)} images')
print(f'TTA       : 3 views per checkpoint (original + hflip + scale-centercrop)')
print(f'Batch size: {_predict_mod.INFER_BATCH}')
print()

model_test_probs = {arch: np.zeros((len(test_df), NUM_CLASSES)) for arch in ALL_ARCHS}

for arch in ALL_ARCHS:
    print(f'{arch}:')
    for fold in range(N_FOLDS):
        # tta_views=3 — predict_test slices transforms[:3] which now has 3 entries
        probs = predict_test(arch, fold, test_df, exp_id=EXP_ID, tta_views=3)
        model_test_probs[arch] += probs
        row_sum = probs.sum(axis=1).mean()
        print(f'  fold {fold}: shape={probs.shape}  row_sum={row_sum:.4f} ✓')
    model_test_probs[arch] /= N_FOLDS
    print(f'  → averaged over {N_FOLDS} folds\n')

print('✅ Test inference complete.')

```

    Starting test inference (3-view TTA)...
    Test set  : 404 images
    TTA       : 3 views per checkpoint (original + hflip + scale-centercrop)
    Batch size: {'convnext': 4, 'eva02': 4, 'dinov2': 4, 'effnet_b4': 8, 'swinv2': 4}
    
    convnext:
      fold 0: shape=(404, 6)  row_sum=1.0000 ✓
      fold 1: shape=(404, 6)  row_sum=1.0000 ✓
      fold 2: shape=(404, 6)  row_sum=1.0000 ✓
      fold 3: shape=(404, 6)  row_sum=1.0000 ✓
      fold 4: shape=(404, 6)  row_sum=1.0000 ✓
      → averaged over 5 folds
    
    eva02:
      fold 0: shape=(404, 6)  row_sum=1.0000 ✓
      fold 1: shape=(404, 6)  row_sum=1.0000 ✓
      fold 2: shape=(404, 6)  row_sum=1.0000 ✓
      fold 3: shape=(404, 6)  row_sum=1.0000 ✓
      fold 4: shape=(404, 6)  row_sum=1.0000 ✓
      → averaged over 5 folds
    
    dinov2:
      fold 0: shape=(404, 6)  row_sum=1.0000 ✓
      fold 1: shape=(404, 6)  row_sum=1.0000 ✓
      fold 2: shape=(404, 6)  row_sum=1.0000 ✓
      fold 3: shape=(404, 6)  row_sum=1.0000 ✓
      fold 4: shape=(404, 6)  row_sum=1.0000 ✓
      → averaged over 5 folds
    
    effnet_b4:
      fold 0: shape=(404, 6)  row_sum=1.0000 ✓
      fold 1: shape=(404, 6)  row_sum=1.0000 ✓
      fold 2: shape=(404, 6)  row_sum=1.0000 ✓
      fold 3: shape=(404, 6)  row_sum=1.0000 ✓
      fold 4: shape=(404, 6)  row_sum=1.0000 ✓
      → averaged over 5 folds
    
    swinv2:
      fold 0: shape=(404, 6)  row_sum=1.0000 ✓
      fold 1: shape=(404, 6)  row_sum=1.0000 ✓
      fold 2: shape=(404, 6)  row_sum=1.0000 ✓
      fold 3: shape=(404, 6)  row_sum=1.0000 ✓
      fold 4: shape=(404, 6)  row_sum=1.0000 ✓
      → averaged over 5 folds
    
    ✅ Test inference complete.


## 5. Load exp06 Test Probabilities (Mega-Ensemble)

The exp06 inference notebook saved per-arch test probability arrays to `oof/exp06/test_probs_{arch}.npy`.
We load them here — no re-inference needed. This gives us exp06's "perspective" on the test set
without spending any GPU time.

**Why this is valid:** Each .npy stores the average of 5-fold test probabilities (after 2-view TTA)
for that architecture. Loading them and averaging with exp07 probs is equivalent to running a
10-model ensemble (same math as the mega-ensemble OOF analysis in Section 14 of analysis-exp07.ipynb).

**Note:** exp06 used 2-view TTA, exp07 uses 3-view. This asymmetry is fine — ensemble averaging
does not require identical preprocessing across experiments.



```python
EXP06_PROBS_DIR = OOF_DIR / 'exp06'

print(f'Loading exp06 test probabilities from: {EXP06_PROBS_DIR}')
print()

exp06_test_probs = {}
all_loaded = True
for arch in ALL_ARCHS:
    npy_path = EXP06_PROBS_DIR / f'test_probs_{arch}.npy'
    if npy_path.exists():
        arr = np.load(npy_path)
        row_sum = arr.sum(axis=1).mean()
        exp06_test_probs[arch] = arr
        print(f'  exp06/{arch}: shape={arr.shape}  row_sum={row_sum:.4f} ✓')
    else:
        print(f'  ⚠️  MISSING: {npy_path}')
        all_loaded = False

print()
if not all_loaded:
    raise FileNotFoundError(
        'exp06 test probability files missing.\n'
        'Re-run 09-inference-exp06-v2.ipynb first (it saves the .npy files),\n'
        'or remove Section 5 and skip the mega-ensemble submission.'
    )
else:
    print(f'✅ All 5 exp06 test probability arrays loaded.')

print()
# Quick sanity: exp06 class distribution should roughly match what we saw before
for arch in ['eva02', 'dinov2']:
    preds = exp06_test_probs[arch].argmax(axis=1)
    dist  = {CLASSES[i]: int((preds == i).sum()) for i in range(NUM_CLASSES)}
    print(f'  exp06/{arch} prediction distribution: {dist}')

```

    Loading exp06 test probabilities from: /home/darrnhard/ML/Competition/FindIT-DAC/oof/exp06
    
      exp06/convnext: shape=(404, 6)  row_sum=1.0000 ✓
      exp06/eva02: shape=(404, 6)  row_sum=1.0000 ✓
      exp06/dinov2: shape=(404, 6)  row_sum=1.0000 ✓
      exp06/effnet_b4: shape=(404, 6)  row_sum=1.0000 ✓
      exp06/swinv2: shape=(404, 6)  row_sum=1.0000 ✓
    
    ✅ All 5 exp06 test probability arrays loaded.
    
      exp06/eva02 prediction distribution: {'fake_mannequin': 51, 'fake_mask': 74, 'fake_printed': 57, 'fake_screen': 63, 'fake_unknown': 51, 'realperson': 108}
      exp06/dinov2 prediction distribution: {'fake_mannequin': 51, 'fake_mask': 70, 'fake_printed': 58, 'fake_screen': 67, 'fake_unknown': 50, 'realperson': 108}


## 6. Build Ensemble Probability Matrices & Sanity Checks

Three probability matrices — one per submission:

| # | Strategy | Composition | OOF F1 |
|---|---|---|---|
| 1 | `pruned_mega` | exp06 (all-5) + exp07 (eva02 + dinov2 + swinv2) | ~0.9319+ |
| 2 | `top4` | exp07: eva02 + dinov2 + effnet_b4 + swinv2 | 0.9306 |
| 3 | `all5` | exp07: all-5 equal weight | 0.9252 |

> **Why pruned mega is primary:** It combines 8 independently trained models spanning two training
> regimes (with/without pseudo-labels), giving higher functional diversity than any single-experiment
> ensemble. The OOF LOO analysis confirmed dropping exp07/convnext and exp07/effnet_b4 from the mega
> improves OOF; the pruned version uses the 3 exp07 models with the strongest individual OOF and
> lowest cross-experiment κ (effnet_b4: κ=0.9179 — most diverse).



```python
# ── 1. Pruned mega-ensemble ────────────────────────────────────────────────────
# exp07: eva02 + dinov2 + swinv2 (dropped convnext, effnet_b4 per LOO)
# exp06: all 5 archs
# Total: 3 + 5 = 8 probability arrays averaged equally

exp07_keep  = ANALYSIS['pruned_mega_archs']['exp07']   # ['eva02', 'dinov2', 'swinv2']
exp07_arrs  = [model_test_probs[a] for a in exp07_keep]
exp06_arrs  = [exp06_test_probs[a] for a in ALL_ARCHS]

test_probs_pruned_mega = np.mean(exp07_arrs + exp06_arrs, axis=0)

# ── 2. exp07 Top-4 (drop convnext) ────────────────────────────────────────────
test_probs_top4 = np.mean(
    [model_test_probs[a] for a in ALL_ARCHS if a != 'convnext'], axis=0
)

# ── 3. exp07 All-5 equal weight ────────────────────────────────────────────────
test_probs_all5 = np.mean([model_test_probs[a] for a in ALL_ARCHS], axis=0)

# ── Probability matrix summary ─────────────────────────────────────────────────
print('Ensemble probability matrices:')
print(f'  {"Name":<30} {"Shape":>12}  {"Mean conf":>10}  {"Row sum":>8}')
print('  ' + '-' * 68)
for name, arr in [
    ('pruned_mega (exp06×5 + exp07×3)', test_probs_pruned_mega),
    ('top4       (exp07 -convnext)',     test_probs_top4),
    ('all5       (exp07 equal weight)', test_probs_all5),
]:
    conf = arr.max(axis=1).mean()
    rsum = arr.sum(axis=1).mean()
    print(f'  {name:<30} {str(arr.shape):>12}  {conf:>10.4f}  {rsum:>8.4f}')

# ── Row sum sanity ─────────────────────────────────────────────────────────────
for name, arr in [
    ('pruned_mega', test_probs_pruned_mega),
    ('top4',        test_probs_top4),
    ('all5',        test_probs_all5),
]:
    assert abs(arr.sum(axis=1).mean() - 1.0) < 1e-4, f'{name}: row sums not ≈ 1.0!'
print()
print('✅ All row sums verified ≈ 1.0')

```

    Ensemble probability matrices:
      Name                                  Shape   Mean conf   Row sum
      --------------------------------------------------------------------
      pruned_mega (exp06×5 + exp07×3)     (404, 6)      0.8853    1.0000
      top4       (exp07 -convnext)       (404, 6)      0.8885    1.0000
      all5       (exp07 equal weight)     (404, 6)      0.8804    1.0000
    
    ✅ All row sums verified ≈ 1.0



```python
# ── Save raw probability arrays ────────────────────────────────────────────────
# Saved so future analysis (pseudo-labels Round 3, ensemble experiments) can
# reuse test probabilities without re-running inference.
probs_save_dir = OOF_DIR / EXP_ID
probs_save_dir.mkdir(parents=True, exist_ok=True)

np.save(probs_save_dir / 'test_probs_pruned_mega.npy', test_probs_pruned_mega)
np.save(probs_save_dir / 'test_probs_top4.npy',        test_probs_top4)
np.save(probs_save_dir / 'test_probs_all5.npy',        test_probs_all5)
for arch in ALL_ARCHS:
    np.save(probs_save_dir / f'test_probs_{arch}.npy', model_test_probs[arch])

print(f'Saved probability arrays to: {probs_save_dir}')
for f in sorted(probs_save_dir.glob('test_probs_*.npy')):
    print(f'  ✅ {f.name}')

```

    Saved probability arrays to: /home/darrnhard/ML/Competition/FindIT-DAC/oof/exp07
      ✅ test_probs_all5.npy
      ✅ test_probs_convnext.npy
      ✅ test_probs_dinov2.npy
      ✅ test_probs_effnet_b4.npy
      ✅ test_probs_eva02.npy
      ✅ test_probs_pruned_mega.npy
      ✅ test_probs_swinv2.npy
      ✅ test_probs_top4.npy



```python
# ── Class distribution check ───────────────────────────────────────────────────
N_TEST    = len(test_df)
N_TRAIN   = len(train_df)
TRAIN_DIST = dict(train_df['label'].value_counts())

print('Class distribution: test predictions vs training proportions')
print(f'  {"Class":<22}  {"Train%":>8}  {"Mega%":>8}  {"Top4%":>8}  {"All5%":>8}  Note')
print('  ' + '-' * 78)

mega_labels = [CLASSES[i] for i in test_probs_pruned_mega.argmax(axis=1)]
top4_labels = [CLASSES[i] for i in test_probs_top4.argmax(axis=1)]
all5_labels = [CLASSES[i] for i in test_probs_all5.argmax(axis=1)]

for cls in CLASSES:
    tr_pct   = TRAIN_DIST.get(cls, 0) / N_TRAIN * 100
    mg_pct   = mega_labels.count(cls) / N_TEST * 100
    t4_pct   = top4_labels.count(cls) / N_TEST * 100
    a5_pct   = all5_labels.count(cls) / N_TEST * 100
    flag     = '  ⚠️  large shift' if abs(mg_pct - tr_pct) > 15 else ''
    print(f'  {cls:<22}  {tr_pct:>8.1f}%  {mg_pct:>8.1f}%  {t4_pct:>8.1f}%  {a5_pct:>8.1f}%{flag}')

print()
print('Note: fake_printed and fake_screen typically shift vs training (known from exp06 analysis).')
print('Large shifts above are pre-existing and not a regression introduced by mega-ensemble.')

```

    Class distribution: test predictions vs training proportions
      Class                     Train%     Mega%     Top4%     All5%  Note
      ------------------------------------------------------------------------------
      fake_mannequin              13.2%      12.6%      12.6%      12.9%
      fake_mask                   18.2%      18.1%      17.8%      17.6%
      fake_printed                 7.1%      13.9%      13.4%      13.6%
      fake_screen                 13.0%      16.6%      16.6%      16.3%
      fake_unknown                21.0%      12.6%      13.1%      12.9%
      realperson                  27.5%      26.2%      26.5%      26.7%
    
    Note: fake_printed and fake_screen typically shift vs training (known from exp06 analysis).
    Large shifts above are pre-existing and not a regression introduced by mega-ensemble.



```python
# ── Prediction agreement between strategies ────────────────────────────────────
preds_mega = test_probs_pruned_mega.argmax(axis=1)
preds_top4 = test_probs_top4.argmax(axis=1)
preds_all5 = test_probs_all5.argmax(axis=1)

print('Prediction agreement across strategies:')
print(f'  pruned_mega vs top4    : {(preds_mega == preds_top4).mean():.4f}')
print(f'  pruned_mega vs all5    : {(preds_mega == preds_all5).mean():.4f}')
print(f'  top4 vs all5           : {(preds_top4 == preds_all5).mean():.4f}')
print()
print('Samples where strategies disagree (potential decision-boundary cases):')
disagree_mega_top4 = (preds_mega != preds_top4).sum()
disagree_mega_all5 = (preds_mega != preds_all5).sum()
print(f'  pruned_mega ≠ top4    : {disagree_mega_top4} samples ({disagree_mega_top4/N_TEST*100:.1f}%)')
print(f'  pruned_mega ≠ all5    : {disagree_mega_all5} samples ({disagree_mega_all5/N_TEST*100:.1f}%)')
print()
print('Lower mega-vs-top4 disagreement = exp06 models and exp07 top3 are mostly aligned.')
print('Higher disagreement = more diversity = more potential ensemble benefit.')

```

    Prediction agreement across strategies:
      pruned_mega vs top4    : 0.9926
      pruned_mega vs all5    : 0.9876
      top4 vs all5           : 0.9901
    
    Samples where strategies disagree (potential decision-boundary cases):
      pruned_mega ≠ top4    : 3 samples (0.7%)
      pruned_mega ≠ all5    : 5 samples (1.2%)
    
    Lower mega-vs-top4 disagreement = exp06 models and exp07 top3 are mostly aligned.
    Higher disagreement = more diversity = more potential ensemble benefit.


## 7. Generate Submissions

**3 submissions, ordered by expected LB performance:**

| Slot | Name | Strategy | Why |
|---|---|---|---|
| 1st | `pruned_mega_argmax` | exp06×5 + exp07×3 | Cross-experiment diversity — highest LB potential |
| 2nd | `top4_argmax` | exp07 top-4 drop convnext | Best clean exp07-only; OOF=0.9306 |
| 3rd | `all5_argmax` | exp07 all-5 equal weight | Calibration baseline; OOF=0.9252 |

**Not submitting:**
- Threshold variants: expected LB Δ=−0.0037, overfitting signal=+0.0108 → confirmed to hurt
- Optimized-weight exp07: performance-weighted (0.9241) was worse than equal (0.9252) in OOF;
  scipy-optimized weights may help but we don't have a 4th slot to test it today



```python
submissions = {}

# 1. PRIMARY: Pruned mega-ensemble argmax
submissions['pruned_mega_argmax'] = make_submission(
    test_probs_pruned_mega, test_df,
    thresholds=None,
    name='pruned_mega_argmax',
)

# 2. SECONDARY: exp07 top-4 (drop convnext), argmax
submissions['top4_argmax'] = make_submission(
    test_probs_top4, test_df,
    thresholds=None,
    name='top4_argmax',
)

# 3. BASELINE: exp07 all-5 equal weight, argmax
submissions['all5_argmax'] = make_submission(
    test_probs_all5, test_df,
    thresholds=None,
    name='all5_argmax',
)

print('Submission files generated:')
for name, (sub_df, _preds_idx) in submissions.items():
    dist  = dict(sub_df['label'].value_counts().sort_index())
    saved = '✅' if (SUBMISSION_DIR / f'{name}.csv').exists() else '⚠️ not found'
    print(f'\n  {name}:')
    print(f'    Rows        : {len(sub_df)}')
    print(f'    Distribution: {dist}')
    print(f'    Saved       : {saved}')

```

    Submission files generated:
    
      pruned_mega_argmax:
        Rows        : 404
        Distribution: {'fake_mannequin': 51, 'fake_mask': 73, 'fake_printed': 56, 'fake_screen': 67, 'fake_unknown': 51, 'realperson': 106}
        Saved       : ✅
    
      top4_argmax:
        Rows        : 404
        Distribution: {'fake_mannequin': 51, 'fake_mask': 72, 'fake_printed': 54, 'fake_screen': 67, 'fake_unknown': 53, 'realperson': 107}
        Saved       : ✅
    
      all5_argmax:
        Rows        : 404
        Distribution: {'fake_mannequin': 52, 'fake_mask': 71, 'fake_printed': 55, 'fake_screen': 66, 'fake_unknown': 52, 'realperson': 108}
        Saved       : ✅


## 8. Summary


```python
print('=' * 72)
print('INFERENCE COMPLETE — exp07')
print('=' * 72)

print()
print('OOF PERFORMANCE (locked from analysis-exp07.ipynb):')
for arch in ALL_ARCHS:
    oof_val = ANALYSIS['oof_f1'][arch]
    ece_val = ANALYSIS['ece'][arch]
    ece_flag = '✅' if ece_val < 0.04 else '⚠️' if ece_val < 0.08 else '❌'
    best = '  ← best single arch' if arch == ANALYSIS['best_single_arch'] else ''
    print(f'  {arch:<14}: OOF={oof_val:.4f}  ECE={ece_val:.4f} {ece_flag}{best}')

print()
print('ENSEMBLE OOF F1:')
print(f'  exp07 all-5 equal weight  : {ANALYSIS["ensemble_oof"]["all5_equal"]:.4f}  ← BASELINE')
print(f'  exp07 top-4 (drop convnext): {ANALYSIS["ensemble_oof"]["top4_drop_convnext"]:.4f}  ← EXP07 BEST')
print(f'  mega all-10 (exp06+exp07)  : {ANALYSIS["loo_mega"]["mega_all10"]:.4f}')
print(f'  pruned mega (exp06×5+exp07×3): {f1_pruned:.4f}  ← PRIMARY TARGET')
print()
print('THRESHOLD DECISION:')
print(f'  Argmax OOF F1        : {ANALYSIS["threshold"]["argmax"]:.4f}')
print(f'  Nested CV OOB        : {ANALYSIS["threshold"]["nested_cv_oob"]:.4f}  (OOB < argmax by {ANALYSIS["threshold"]["argmax"] - ANALYSIS["threshold"]["nested_cv_oob"]:.4f})')
print(f'  Overfitting signal   : {ANALYSIS["threshold"]["overfitting_signal"]:+.4f}  (>0.005 → significant)')
print(f'  Expected LB Δ        : {ANALYSIS["threshold"]["expected_lb_gain"]:+.4f}  → {ANALYSIS["threshold"]["decision"]}')
print()
print('TTA:')
print('  3-view (original + hflip + scale-centercrop @ 115%)')
print('  Improvement over exp06: +1 view (2 → 3); expected gain on fake_screen errors')
print()
print('SUBMISSION FILES:')
for name, (sub_df, _) in submissions.items():
    saved_path = SUBMISSION_DIR / f'{name}.csv'
    exists     = '✅' if saved_path.exists() else '⚠️ not found'
    slot_map   = {
        'pruned_mega_argmax': '  ← SUBMIT 1st (primary — cross-exp diversity)',
        'top4_argmax'       : '  ← SUBMIT 2nd (exp07 clean best)',
        'all5_argmax'       : '  ← SUBMIT 3rd (calibration baseline)',
    }
    print(f'  {name:<30}: {exists}{slot_map.get(name, "")}')
print()
print('SUBMISSION ORDER:')
print('  1st → pruned_mega_argmax   (exp06×5 + exp07×3; highest diversity; primary)')
print('  2nd → top4_argmax          (exp07 top-4; OOF=0.9306; cross-check mega benefit)')
print('  3rd → all5_argmax          (exp07 all-5; OOF=0.9252; measures cost of convnext)')
print()
print('WHAT TO WATCH:')
print('  • If pruned_mega > top4 on LB → cross-experiment ensembling works; do it for exp08.')
print('  • If top4 ≈ pruned_mega on LB → exp07 diversity is sufficient; mega adds noise.')
print('  • If all5 > top4 on LB → convnext adds LB value despite OOF regression (same')
print('    as effnet_b4 in exp03 — high OOF correlation ≠ no LB value).')
print()
print('PREVIOUS BEST LB : 0.76739  (exp03)')
print('exp06 BEST LB    : TBD (submitted but not recorded here)')
print(f'{"="*72}')

```

    ========================================================================
    INFERENCE COMPLETE — exp07
    ========================================================================
    
    OOF PERFORMANCE (locked from analysis-exp07.ipynb):
      convnext      : OOF=0.9073  ECE=0.0452 ⚠️
      eva02         : OOF=0.9293  ECE=0.0312 ✅  ← best single arch
      dinov2        : OOF=0.9254  ECE=0.0280 ✅
      effnet_b4     : OOF=0.8826  ECE=0.0207 ✅
      swinv2        : OOF=0.9131  ECE=0.0246 ✅
    
    ENSEMBLE OOF F1:
      exp07 all-5 equal weight  : 0.9252  ← BASELINE
      exp07 top-4 (drop convnext): 0.9306  ← EXP07 BEST
      mega all-10 (exp06+exp07)  : 0.9306
      pruned mega (exp06×5+exp07×3): 0.9349  ← PRIMARY TARGET
    
    THRESHOLD DECISION:
      Argmax OOF F1        : 0.9252
      Nested CV OOB        : 0.9216  (OOB < argmax by 0.0036)
      Overfitting signal   : +0.0108  (>0.005 → significant)
      Expected LB Δ        : -0.0037  → NO_THRESHOLD
    
    TTA:
      3-view (original + hflip + scale-centercrop @ 115%)
      Improvement over exp06: +1 view (2 → 3); expected gain on fake_screen errors
    
    SUBMISSION FILES:
      pruned_mega_argmax            : ✅  ← SUBMIT 1st (primary — cross-exp diversity)
      top4_argmax                   : ✅  ← SUBMIT 2nd (exp07 clean best)
      all5_argmax                   : ✅  ← SUBMIT 3rd (calibration baseline)
    
    SUBMISSION ORDER:
      1st → pruned_mega_argmax   (exp06×5 + exp07×3; highest diversity; primary)
      2nd → top4_argmax          (exp07 top-4; OOF=0.9306; cross-check mega benefit)
      3rd → all5_argmax          (exp07 all-5; OOF=0.9252; measures cost of convnext)
    
    WHAT TO WATCH:
      • If pruned_mega > top4 on LB → cross-experiment ensembling works; do it for exp08.
      • If top4 ≈ pruned_mega on LB → exp07 diversity is sufficient; mega adds noise.
      • If all5 > top4 on LB → convnext adds LB value despite OOF regression (same
        as effnet_b4 in exp03 — high OOF correlation ≠ no LB value).
    
    PREVIOUS BEST LB : 0.76739  (exp03)
    exp06 BEST LB    : TBD (submitted but not recorded here)
    ========================================================================



```python

```
