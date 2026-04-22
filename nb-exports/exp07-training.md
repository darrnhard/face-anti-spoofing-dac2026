# exp07 — 5-Architecture Ensemble with Round 2 Soft Pseudo-Labels

**Experiment:** `exp07`  
**Architectures:** ConvNeXt-Base · EVA-02-Base · DINOv2-Base · EfficientNet-B4 · SwinV2-Base  
**Training data:** `train_pseudo_exp07.csv` — real + Round 2 pseudo-labeled test images

## What changed vs exp06

| Change | exp06 | exp07 | Rationale |
|---|---|---|---|
| Pseudo-label source | exp04_all4_swinv2 (LB=0.78555) | exp06_all5 (OOF=0.9355, LB=0.78555) | Better-calibrated ensemble |
| fake_screen override | None | 0.85 (if confirmed visually) | Weakest class (F1=0.9040) needs more data |
| Augmentation | exp06 pipeline | + GaussianNoise, RandomRotation, improved MoirePattern | FLAT-2D→REAL was #1 failure (34/96 errors) |
| Training epochs | convnext=70, others=60 | convnext=80, others=70 | More data + SwinV2 hit epoch ceiling in exp06 fold 4 |
| Everything else | — | Identical to exp06 | No other variables changed |

## Prerequisites
1. `10-pseudo-label-r2.ipynb` has been run → `data/processed/train_pseudo_exp07.csv` exists
2. `src/data/augmentation.py` — updated (GaussianNoise, RandomRotation, RGB MoirePattern)
3. `src/data/dataset.py` — soft label support already present from exp06
4. `src/training/losses.py` — SoftCrossEntropyLoss already present from exp06
5. `src/training/trainer.py` — soft_criterion support already present from exp06

This notebook is intentionally thin — all logic lives in `src/`.


```python
! pip install -r /workspace/fas-competition/requirements.txt -q
```


```python
! pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 1 · Setup


```python
import sys
from pathlib import Path

PROJECT_ROOT = Path().resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.utils.config import (
    DEVICE, CROP_TRAIN_DIR, CROP_TEST_DIR,
    N_FOLDS, CLASSES, MODEL_DIR, OOF_DIR,
    PROCESSED_DIR, print_env,
)
from src.utils.seed import set_seed
from src.training.losses import get_class_weights
from src.training.trainer import train_fold, dinov2_linear_probe
from src.models.registry import ARCH_CONFIGS, create_model, get_param_groups

set_seed(42)
print_env()

EXP_ID = 'exp07'
```

    Environment  : remote
    Project dir  : /workspace/fas-competition
      interim    : /workspace/fas-competition/data/interim
        train    : /workspace/fas-competition/data/interim/train
        test     : /workspace/fas-competition/data/interim/test
      processed  : /workspace/fas-competition/data/processed
        crops    : /workspace/fas-competition/data/processed/crops
        train csv: /workspace/fas-competition/data/processed/train_clean.csv
        test csv : /workspace/fas-competition/data/processed/test.csv
    Model dir    : /workspace/fas-competition/models
    OOF dir      : /workspace/fas-competition/oof
    Submissions  : /workspace/fas-competition/submissions
    Device       : cuda
    GPU          : NVIDIA GeForce RTX 3090
    VRAM         : 25.3 GB


## 2 · Epoch Overrides

Patching ARCH_CONFIGS at runtime — registry.py is unchanged.

**Rationale for +30 epochs across all architectures:**
- Round 2 pseudo labels add more training images → each epoch is more informative
- Better augmentation (GaussianNoise, RandomRotation) means the model needs more passes
  to see the full diversity of augmented samples
- SwinV2 fold 4 hit the epoch ceiling in exp06 (best_epoch=60, total_epochs=60)
- Early stopping (patience=15) is still active — models that converge early will stop


```python
EPOCH_OVERRIDES = {
    'convnext'  : 100,   # was 70 in exp06
    'eva02'     : 90,   # was 60 in exp06
    'dinov2'    : 90,   # was 60 in exp06
    'effnet_b4' : 90,   # was 60 in exp06
    'swinv2'    : 90,   # was 60 in exp06 — hit ceiling at fold 4 (best=epoch 60/60)
}

print('Epoch overrides applied to ARCH_CONFIGS:')
for arch, n_epochs in EPOCH_OVERRIDES.items():
    prev = ARCH_CONFIGS[arch]['epochs']
    ARCH_CONFIGS[arch]['epochs'] = n_epochs
    print(f'  {arch:<14}: {prev} → {n_epochs}')

print()
print('Note: early stopping (patience=15) still active.')
print('Models that converge before the ceiling will stop early.')
```

    Epoch overrides applied to ARCH_CONFIGS:
      convnext      : 70 → 100
      eva02         : 60 → 90
      dinov2        : 60 → 90
      effnet_b4     : 60 → 90
      swinv2        : 60 → 90
    
    Note: early stopping (patience=15) still active.
    Models that converge before the ceiling will stop early.


## 3 · Load Data


```python
# Load train_pseudo_exp07.csv (real + Round 2 pseudo rows)
PSEUDO_CSV = PROCESSED_DIR / 'train_pseudo_exp07.csv'
assert PSEUDO_CSV.exists(), (
    f'train_pseudo_exp07.csv not found at {PSEUDO_CSV}\n'
    f'Run 10-pseudo-label-r2.ipynb first.'
)

train_df = pd.read_csv(PSEUDO_CSV)

# ── Remap crop paths to current environment ───────────────────────────────────
def remap_crop_path(row):
    fname = Path(row['crop_path']).name
    if row['is_pseudo']:
        return str(CROP_TEST_DIR / fname)
    else:
        return str(CROP_TRAIN_DIR / fname)

train_df['crop_path'] = train_df.apply(remap_crop_path, axis=1)

# ── Stats ─────────────────────────────────────────────────────────────────────
n_real   = (~train_df['is_pseudo']).sum()
n_pseudo = train_df['is_pseudo'].sum()
print(f'Train rows   : {len(train_df):,}  ({n_real:,} real + {n_pseudo:,} pseudo)')
print(f'Folds        : {sorted(train_df["fold"].unique())}  (-1 = pseudo, never in val)')
print()

# ── Sanity checks ─────────────────────────────────────────────────────────────
assert 'is_pseudo' in train_df.columns, 'is_pseudo column missing'
assert (train_df[train_df['is_pseudo']]['fold'] == -1).all(), \
    'BUG: pseudo rows have fold != -1'
real_folds = sorted(train_df[~train_df['is_pseudo']]['fold'].unique())
assert real_folds == [0, 1, 2, 3, 4], f'Real folds should be 0-4, got {real_folds}'
assert Path(train_df[~train_df['is_pseudo']].iloc[0]['crop_path']).exists(), \
    'First real crop not found'
assert Path(train_df[train_df['is_pseudo']].iloc[0]['crop_path']).exists(), \
    'First pseudo crop not found'

print('✅ All sanity checks passed.')
print()
print('Class distribution (real rows only):')
print(train_df[~train_df['is_pseudo']]['label'].value_counts().sort_index().to_string())
print()
print('Pseudo-label class distribution (Round 2):')
print(train_df[train_df['is_pseudo']]['label'].value_counts().sort_index().to_string())
```

    Train rows   : 1,718  (1,464 real + 254 pseudo)
    Folds        : [-1, 0, 1, 2, 3, 4]  (-1 = pseudo, never in val)
    
    ✅ All sanity checks passed.
    
    Class distribution (real rows only):
    label
    fake_mannequin    193
    fake_mask         266
    fake_printed      104
    fake_screen       191
    fake_unknown      307
    realperson        403
    
    Pseudo-label class distribution (Round 2):
    label
    fake_mannequin    44
    fake_mask         39
    fake_printed      36
    fake_screen       37
    fake_unknown      39
    realperson        59


## 4 · Class Weights — Real Rows Only

**Critical:** class weights must be computed from real labeled rows only.
Pseudo rows must not inflate class counts — their distribution reflects the
ensemble's test predictions, not the true training distribution.


```python
real_df      = train_df[~train_df['is_pseudo']].reset_index(drop=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_weights = get_class_weights(real_df, device)

print('Class weights (computed from real rows only):')
for i, cls in enumerate(CLASSES):
    n   = (real_df['label'] == cls).sum()
    bar = '█' * int(class_weights[i].item() * 10)
    print(f'  {cls:<22}: n={n:>4}  weight={class_weights[i]:.4f}  {bar}')
print()
print(f'Device: {DEVICE}')
```

    Class weights (computed from real rows only):
      fake_mannequin        : n= 193  weight=1.2642  ████████████
      fake_mask             : n= 266  weight=0.9173  █████████
      fake_printed          : n= 104  weight=2.3462  ███████████████████████
      fake_screen           : n= 191  weight=1.2775  ████████████
      fake_unknown          : n= 307  weight=0.7948  ███████
      realperson            : n= 403  weight=0.6055  ██████
    
    Device: cuda


## 5 · DINOv2 Linear Probe

Same probe infrastructure as exp06 — trains the classification head on frozen
DINOv2 features before full fine-tuning. Run once per fold, results cached to disk.


```python
print('Running DINOv2 linear probe (5 folds × 20 epochs)...')
print('This initializes the head before full fine-tuning.')
print()

for fold in range(N_FOLDS):
    dinov2_linear_probe(
        fold         = fold,
        train_df     = train_df,
        class_weights= class_weights,
    )

print()
print('✅ DINOv2 linear probe complete.')
```

    Running DINOv2 linear probe (5 folds × 20 epochs)...
    This initializes the head before full fine-tuning.
    
    
    ============================================================
    DINOv2 Linear Probe — Fold 0
    Frozen backbone | 20 epochs | lr=1e-3
    ============================================================



    model.safetensors:   0%|          | 0.00/346M [00:00<?, ?B/s]


    
      Frozen:    85.7M params
      Trainable: 4.6K params (head only)
    
      Training linear probe...
        Epoch 01/20 | Train: 0.8229/0.6021 | Val: 1.2634/0.2943 ★
        Epoch 02/20 | Train: 0.4194/0.7838 | Val: 1.1880/0.3719 ★
        Epoch 03/20 | Train: 0.3532/0.8039 | Val: 1.1079/0.4445 ★
        Epoch 04/20 | Train: 0.2888/0.8253 | Val: 1.0277/0.5201 ★
        Epoch 05/20 | Train: 0.2557/0.8441 | Val: 0.9488/0.6019 ★
        Epoch 06/20 | Train: 0.2432/0.8570 | Val: 0.8740/0.6685 ★
        Epoch 07/20 | Train: 0.2307/0.8628 | Val: 0.8036/0.7307 ★
        Epoch 08/20 | Train: 0.2322/0.8727 | Val: 0.7381/0.7779 ★
        Epoch 09/20 | Train: 0.2031/0.8809 | Val: 0.6782/0.7991 ★
        Epoch 10/20 | Train: 0.1909/0.8824 | Val: 0.6241/0.8166 ★
        Epoch 11/20 | Train: 0.1971/0.8910 | Val: 0.5751/0.8497 ★
        Epoch 12/20 | Train: 0.1913/0.8932 | Val: 0.5325/0.8595 ★
        Epoch 13/20 | Train: 0.1690/0.8943 | Val: 0.4945/0.8787 ★
        Epoch 14/20 | Train: 0.1797/0.8816 | Val: 0.4614/0.8891 ★
        Epoch 15/20 | Train: 0.1705/0.8949 | Val: 0.4319/0.8915 ★
        Epoch 16/20 | Train: 0.1761/0.8792 | Val: 0.4059/0.8981 ★
        Epoch 17/20 | Train: 0.1490/0.9073 | Val: 0.3830/0.9079 ★
        Epoch 18/20 | Train: 0.1589/0.8989 | Val: 0.3626/0.9047
        Epoch 19/20 | Train: 0.1551/0.9141 | Val: 0.3451/0.9013
        Epoch 20/20 | Train: 0.1489/0.9132 | Val: 0.3300/0.9073
    
      Linear probe best Val F1: 0.9079
      Head weights saved → /workspace/fas-competition/models/probe/dinov2_probe_fold0.pth
      → Will be auto-loaded by train_fold('dinov2', ...)
    
    ============================================================
    DINOv2 Linear Probe — Fold 1
    Frozen backbone | 20 epochs | lr=1e-3
    ============================================================
    
      Frozen:    85.7M params
      Trainable: 4.6K params (head only)
    
      Training linear probe...
        Epoch 01/20 | Train: 0.8820/0.5730 | Val: 1.5096/0.0872 ★
        Epoch 02/20 | Train: 0.3889/0.7989 | Val: 1.4265/0.1332 ★
        Epoch 03/20 | Train: 0.3306/0.8237 | Val: 1.3364/0.1788 ★
        Epoch 04/20 | Train: 0.2900/0.8393 | Val: 1.2454/0.2660 ★
        Epoch 05/20 | Train: 0.2331/0.8658 | Val: 1.1556/0.3685 ★
        Epoch 06/20 | Train: 0.2315/0.8698 | Val: 1.0688/0.4373 ★
        Epoch 07/20 | Train: 0.2117/0.8860 | Val: 0.9864/0.5051 ★
        Epoch 08/20 | Train: 0.1909/0.8934 | Val: 0.9099/0.5754 ★
        Epoch 09/20 | Train: 0.1792/0.8942 | Val: 0.8390/0.6130 ★
        Epoch 10/20 | Train: 0.1793/0.8977 | Val: 0.7754/0.6417 ★
        Epoch 11/20 | Train: 0.1615/0.8991 | Val: 0.7179/0.6820 ★
        Epoch 12/20 | Train: 0.1618/0.9066 | Val: 0.6664/0.7063 ★
        Epoch 13/20 | Train: 0.1790/0.8931 | Val: 0.6195/0.7216 ★
        Epoch 14/20 | Train: 0.1596/0.9012 | Val: 0.5783/0.7509 ★
        Epoch 15/20 | Train: 0.1576/0.9085 | Val: 0.5417/0.7784 ★
        Epoch 16/20 | Train: 0.1391/0.9164 | Val: 0.5083/0.7826 ★
        Epoch 17/20 | Train: 0.1676/0.8992 | Val: 0.4791/0.7998 ★
        Epoch 18/20 | Train: 0.1513/0.9051 | Val: 0.4533/0.8080 ★
        Epoch 19/20 | Train: 0.1325/0.9135 | Val: 0.4300/0.8267 ★
        Epoch 20/20 | Train: 0.1287/0.9151 | Val: 0.4092/0.8327 ★
    
      Linear probe best Val F1: 0.8327
      Head weights saved → /workspace/fas-competition/models/probe/dinov2_probe_fold1.pth
      → Will be auto-loaded by train_fold('dinov2', ...)
    
    ============================================================
    DINOv2 Linear Probe — Fold 2
    Frozen backbone | 20 epochs | lr=1e-3
    ============================================================


    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.


    
      Frozen:    85.7M params
      Trainable: 4.6K params (head only)
    
      Training linear probe...
        Epoch 01/20 | Train: 0.8729/0.5856 | Val: 1.4147/0.2076 ★
        Epoch 02/20 | Train: 0.4135/0.7888 | Val: 1.3415/0.2512 ★
        Epoch 03/20 | Train: 0.3292/0.8146 | Val: 1.2609/0.3117 ★
        Epoch 04/20 | Train: 0.2929/0.8297 | Val: 1.1792/0.3907 ★
        Epoch 05/20 | Train: 0.2692/0.8408 | Val: 1.0990/0.4804 ★
        Epoch 06/20 | Train: 0.2606/0.8661 | Val: 1.0213/0.5633 ★
        Epoch 07/20 | Train: 0.2364/0.8563 | Val: 0.9473/0.6277 ★
        Epoch 08/20 | Train: 0.2222/0.8695 | Val: 0.8776/0.6693 ★
        Epoch 09/20 | Train: 0.2050/0.8738 | Val: 0.8129/0.7260 ★
        Epoch 10/20 | Train: 0.2002/0.8887 | Val: 0.7531/0.7429 ★
        Epoch 11/20 | Train: 0.1788/0.8928 | Val: 0.6979/0.7757 ★
        Epoch 12/20 | Train: 0.1727/0.8947 | Val: 0.6471/0.7957 ★
        Epoch 13/20 | Train: 0.1670/0.9069 | Val: 0.6009/0.8126 ★
        Epoch 14/20 | Train: 0.1689/0.9088 | Val: 0.5586/0.8295 ★
        Epoch 15/20 | Train: 0.1489/0.9148 | Val: 0.5206/0.8437 ★
        Epoch 16/20 | Train: 0.1702/0.8910 | Val: 0.4866/0.8437
        Epoch 17/20 | Train: 0.1549/0.9067 | Val: 0.4561/0.8437
        Epoch 18/20 | Train: 0.1464/0.9094 | Val: 0.4293/0.8408
        Epoch 19/20 | Train: 0.1350/0.9138 | Val: 0.4052/0.8565 ★
        Epoch 20/20 | Train: 0.1493/0.9016 | Val: 0.3839/0.8695 ★
    
      Linear probe best Val F1: 0.8695
      Head weights saved → /workspace/fas-competition/models/probe/dinov2_probe_fold2.pth
      → Will be auto-loaded by train_fold('dinov2', ...)
    
    ============================================================
    DINOv2 Linear Probe — Fold 3
    Frozen backbone | 20 epochs | lr=1e-3
    ============================================================
    
      Frozen:    85.7M params
      Trainable: 4.6K params (head only)
    
      Training linear probe...
        Epoch 01/20 | Train: 0.8504/0.5954 | Val: 1.2986/0.2866 ★
        Epoch 02/20 | Train: 0.4067/0.7886 | Val: 1.2270/0.3454 ★
        Epoch 03/20 | Train: 0.3197/0.8323 | Val: 1.1490/0.4162 ★
        Epoch 04/20 | Train: 0.2892/0.8519 | Val: 1.0704/0.4721 ★
        Epoch 05/20 | Train: 0.2591/0.8562 | Val: 0.9926/0.5404 ★
        Epoch 06/20 | Train: 0.2344/0.8697 | Val: 0.9175/0.5697 ★
        Epoch 07/20 | Train: 0.2313/0.8669 | Val: 0.8474/0.6121 ★
        Epoch 08/20 | Train: 0.2160/0.8781 | Val: 0.7829/0.6685 ★
        Epoch 09/20 | Train: 0.2005/0.8842 | Val: 0.7234/0.6984 ★
        Epoch 10/20 | Train: 0.1853/0.8876 | Val: 0.6685/0.7262 ★
        Epoch 11/20 | Train: 0.1726/0.8948 | Val: 0.6191/0.7494 ★
        Epoch 12/20 | Train: 0.1631/0.9072 | Val: 0.5751/0.7701 ★
        Epoch 13/20 | Train: 0.1774/0.8956 | Val: 0.5353/0.8020 ★
        Epoch 14/20 | Train: 0.1525/0.9026 | Val: 0.4993/0.8027 ★
        Epoch 15/20 | Train: 0.1570/0.9057 | Val: 0.4669/0.8196 ★
        Epoch 16/20 | Train: 0.1503/0.9207 | Val: 0.4378/0.8285 ★
        Epoch 17/20 | Train: 0.1663/0.8949 | Val: 0.4120/0.8306 ★
        Epoch 18/20 | Train: 0.1227/0.9322 | Val: 0.3898/0.8381 ★
        Epoch 19/20 | Train: 0.1337/0.9201 | Val: 0.3704/0.8409 ★
        Epoch 20/20 | Train: 0.1321/0.9152 | Val: 0.3530/0.8481 ★
    
      Linear probe best Val F1: 0.8481
      Head weights saved → /workspace/fas-competition/models/probe/dinov2_probe_fold3.pth
      → Will be auto-loaded by train_fold('dinov2', ...)
    
    ============================================================
    DINOv2 Linear Probe — Fold 4
    Frozen backbone | 20 epochs | lr=1e-3
    ============================================================
    
      Frozen:    85.7M params
      Trainable: 4.6K params (head only)
    
      Training linear probe...
        Epoch 01/20 | Train: 0.8618/0.5925 | Val: 1.3403/0.2132 ★
        Epoch 02/20 | Train: 0.4161/0.7867 | Val: 1.2646/0.2618 ★
        Epoch 03/20 | Train: 0.3358/0.8194 | Val: 1.1814/0.3452 ★
        Epoch 04/20 | Train: 0.2724/0.8538 | Val: 1.0966/0.4223 ★
        Epoch 05/20 | Train: 0.2616/0.8475 | Val: 1.0131/0.5073 ★
        Epoch 06/20 | Train: 0.2531/0.8550 | Val: 0.9323/0.6022 ★
        Epoch 07/20 | Train: 0.2359/0.8652 | Val: 0.8550/0.6918 ★
        Epoch 08/20 | Train: 0.2024/0.8789 | Val: 0.7827/0.7434 ★
        Epoch 09/20 | Train: 0.1989/0.8854 | Val: 0.7158/0.7622 ★
        Epoch 10/20 | Train: 0.1772/0.8875 | Val: 0.6546/0.7796 ★
        Epoch 11/20 | Train: 0.1918/0.8879 | Val: 0.5989/0.7855 ★
        Epoch 12/20 | Train: 0.1693/0.9048 | Val: 0.5491/0.8164 ★
        Epoch 13/20 | Train: 0.1513/0.9166 | Val: 0.5044/0.8421 ★
        Epoch 14/20 | Train: 0.1587/0.9027 | Val: 0.4641/0.8471 ★
        Epoch 15/20 | Train: 0.1540/0.9071 | Val: 0.4281/0.8671 ★
        Epoch 16/20 | Train: 0.1470/0.9129 | Val: 0.3969/0.8770 ★
        Epoch 17/20 | Train: 0.1567/0.9020 | Val: 0.3689/0.8970 ★
        Epoch 18/20 | Train: 0.1328/0.9151 | Val: 0.3445/0.9026 ★
        Epoch 19/20 | Train: 0.1510/0.9068 | Val: 0.3231/0.9046 ★
        Epoch 20/20 | Train: 0.1523/0.9078 | Val: 0.3039/0.9073 ★
    
      Linear probe best Val F1: 0.9073
      Head weights saved → /workspace/fas-competition/models/probe/dinov2_probe_fold4.pth
      → Will be auto-loaded by train_fold('dinov2', ...)
    
    ✅ DINOv2 linear probe complete.


## 6 · Train ConvNeXt-Base  (80 epochs)


```python
convnext_results = {}

for fold in range(N_FOLDS):
    result = train_fold(
        fold         = fold,
        model_key    = 'convnext',
        train_df     = train_df,
        class_weights= class_weights,
        use_cutmix   = True,
        use_sampler  = False,
        exp_id       = EXP_ID,
    )
    convnext_results[fold] = result
    print(f'  ConvNeXt fold {fold}: val F1 = {result["best_f1"]:.4f}')

scores = [r['best_f1'] for r in convnext_results.values()]
print(f'\nConvNeXt summary: mean={np.mean(scores):.4f} ± {np.std(scores):.4f}')
print(f'  exp06 baseline: mean=0.9184 ± 0.0149')
```

    
    ============================================================
    FOLD 0 — convnext (convnext_base.fb_in22k)
    Epochs: 100 | Freeze: 5 | Warmup: 3 | CutMix: True
    ============================================================
    
      Total:    87.6M params
      Head:     6.2K params (lr=5e-04)
      Backbone: 87.6M params (lr=5e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (5 epochs, no CutMix)
        Epoch 1/5 | Train: 2.5624/0.4699 | Val: 1.4224/0.2100
        Epoch 2/5 | Train: 1.6617/0.6997 | Val: 1.3928/0.2446
        Epoch 3/5 | Train: 1.2746/0.7411 | Val: 1.3500/0.2764
        Epoch 4/5 | Train: 1.0983/0.7735 | Val: 1.2995/0.3266
        Epoch 5/5 | Train: 0.9529/0.7888 | Val: 1.2442/0.3823
    
      Stage 2: Full fine-tuning (95 epochs, CutMix=True)
        Epoch 06/100 | Train: 0.8201/0.8069 | Val: 0.3066/0.8679 | LR: 2.00e-04 ★
        Epoch 07/100 | Train: 0.6193/0.8541 | Val: 0.3034/0.8679 | LR: 3.50e-04
        Epoch 08/100 | Train: 0.5653/0.8761 | Val: 0.2985/0.8703 | LR: 5.00e-04 ★
        Epoch 09/100 | Train: 0.3951/0.9027 | Val: 0.2932/0.8740 | LR: 4.97e-04 ★
        Epoch 10/100 | Train: 0.3369/0.9326 | Val: 0.2868/0.8740 | LR: 4.88e-04
        Epoch 11/100 | Train: 0.3028/0.9460 | Val: 0.2803/0.8802 | LR: 4.73e-04 ★
        Epoch 12/100 | Train: 0.2890/0.9572 | Val: 0.2739/0.8802 | LR: 4.52e-04
        Epoch 13/100 | Train: 0.2825/0.9665 | Val: 0.2676/0.8867 | LR: 4.27e-04 ★
        Epoch 14/100 | Train: 0.2892/0.9613 | Val: 0.2610/0.8954 | LR: 3.97e-04 ★
        Epoch 15/100 | Train: 0.2404/0.9837 | Val: 0.2548/0.8979 | LR: 3.64e-04 ★
        Epoch 16/100 | Train: 0.2620/0.9799 | Val: 0.2487/0.8979 | LR: 3.27e-04
        Epoch 17/100 | Train: 0.2317/0.9823 | Val: 0.2432/0.9050 | LR: 2.89e-04 ★
        Epoch 18/100 | Train: 0.2258/0.9875 | Val: 0.2383/0.9074 | LR: 2.50e-04 ★
        Epoch 19/100 | Train: 0.2030/0.9897 | Val: 0.2338/0.9074 | LR: 2.11e-04
        Epoch 20/100 | Train: 0.2157/0.9882 | Val: 0.2293/0.9074 | LR: 1.73e-04
        Epoch 21/100 | Train: 0.2118/0.9869 | Val: 0.2253/0.9074 | LR: 1.37e-04
        Epoch 22/100 | Train: 0.1989/0.9922 | Val: 0.2214/0.9074 | LR: 1.03e-04
        Epoch 23/100 | Train: 0.1891/0.9944 | Val: 0.2178/0.9074 | LR: 7.33e-05
        Epoch 24/100 | Train: 0.1939/0.9930 | Val: 0.2145/0.9074 | LR: 4.78e-05
        Epoch 25/100 | Train: 0.1904/0.9952 | Val: 0.2114/0.9105 | LR: 2.73e-05 ★
        Epoch 26/100 | Train: 0.1913/0.9949 | Val: 0.2085/0.9071 | LR: 1.23e-05
        Epoch 27/100 | Train: 0.1936/0.9965 | Val: 0.2058/0.9071 | LR: 3.18e-06
        Epoch 28/100 | Train: 0.1928/0.9946 | Val: 0.2032/0.9071 | LR: 5.00e-04
        Epoch 29/100 | Train: 0.2077/0.9927 | Val: 0.2003/0.9095 | LR: 4.99e-04
        Epoch 30/100 | Train: 0.2086/0.9877 | Val: 0.1978/0.9089 | LR: 4.97e-04
        Epoch 31/100 | Train: 0.2272/0.9883 | Val: 0.1951/0.9089 | LR: 4.93e-04
        Epoch 32/100 | Train: 0.2040/0.9927 | Val: 0.1925/0.9089 | LR: 4.88e-04
        Epoch 33/100 | Train: 0.2086/0.9882 | Val: 0.1900/0.9150 | LR: 4.81e-04 ★
        Epoch 34/100 | Train: 0.1955/0.9927 | Val: 0.1877/0.9150 | LR: 4.73e-04
        Epoch 35/100 | Train: 0.2063/0.9924 | Val: 0.1857/0.9150 | LR: 4.63e-04
        Epoch 36/100 | Train: 0.2016/0.9911 | Val: 0.1835/0.9150 | LR: 4.52e-04
        Epoch 37/100 | Train: 0.1967/0.9905 | Val: 0.1818/0.9150 | LR: 4.40e-04
        Epoch 38/100 | Train: 0.1903/0.9952 | Val: 0.1807/0.9150 | LR: 4.27e-04
        Epoch 39/100 | Train: 0.2099/0.9943 | Val: 0.1798/0.9150 | LR: 4.12e-04
        Epoch 40/100 | Train: 0.1930/0.9946 | Val: 0.1787/0.9184 | LR: 3.97e-04 ★
        Epoch 41/100 | Train: 0.1838/0.9975 | Val: 0.1774/0.9175 | LR: 3.81e-04
        Epoch 42/100 | Train: 0.1844/0.9983 | Val: 0.1760/0.9175 | LR: 3.64e-04
        Epoch 43/100 | Train: 0.1956/0.9963 | Val: 0.1746/0.9140 | LR: 3.46e-04
        Epoch 44/100 | Train: 0.2077/0.9938 | Val: 0.1735/0.9175 | LR: 3.27e-04
        Epoch 45/100 | Train: 0.1860/0.9915 | Val: 0.1726/0.9140 | LR: 3.08e-04
        Epoch 46/100 | Train: 0.1780/0.9964 | Val: 0.1719/0.9140 | LR: 2.89e-04
        Epoch 47/100 | Train: 0.1812/0.9966 | Val: 0.1714/0.9175 | LR: 2.70e-04
        Epoch 48/100 | Train: 0.1827/0.9956 | Val: 0.1706/0.9175 | LR: 2.50e-04
        Epoch 49/100 | Train: 0.1799/0.9969 | Val: 0.1699/0.9175 | LR: 2.30e-04
        Epoch 50/100 | Train: 0.1756/0.9984 | Val: 0.1691/0.9175 | LR: 2.11e-04
        Epoch 51/100 | Train: 0.1847/0.9982 | Val: 0.1684/0.9175 | LR: 1.92e-04
        Epoch 52/100 | Train: 0.1781/0.9966 | Val: 0.1679/0.9175 | LR: 1.73e-04
        Epoch 53/100 | Train: 0.1742/1.0000 | Val: 0.1673/0.9175 | LR: 1.54e-04
        Epoch 54/100 | Train: 0.1766/0.9978 | Val: 0.1668/0.9175 | LR: 1.37e-04
        Epoch 55/100 | Train: 0.1744/0.9980 | Val: 0.1665/0.9175 | LR: 1.19e-04
        Early stopping at epoch 55 (patience=15)
    
      Best: Epoch 40, Val F1: 0.9184
      ConvNeXt fold 0: val F1 = 0.9184
    
    ============================================================
    FOLD 1 — convnext (convnext_base.fb_in22k)
    Epochs: 100 | Freeze: 5 | Warmup: 3 | CutMix: True
    ============================================================
    
      Total:    87.6M params
      Head:     6.2K params (lr=5e-04)
      Backbone: 87.6M params (lr=5e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (5 epochs, no CutMix)
        Epoch 1/5 | Train: 2.6815/0.4397 | Val: 1.4975/0.1145
        Epoch 2/5 | Train: 1.7111/0.7090 | Val: 1.4672/0.1197
        Epoch 3/5 | Train: 1.2656/0.7498 | Val: 1.4247/0.1401
        Epoch 4/5 | Train: 1.0952/0.7752 | Val: 1.3749/0.1526
        Epoch 5/5 | Train: 0.9193/0.8026 | Val: 1.3206/0.2253
    
      Stage 2: Full fine-tuning (95 epochs, CutMix=True)
        Epoch 06/100 | Train: 0.8122/0.8207 | Val: 0.3997/0.7975 | LR: 2.00e-04 ★
        Epoch 07/100 | Train: 0.6019/0.8607 | Val: 0.3958/0.7999 | LR: 3.50e-04 ★
        Epoch 08/100 | Train: 0.4562/0.8951 | Val: 0.3903/0.8025 | LR: 5.00e-04 ★
        Epoch 09/100 | Train: 0.3690/0.9378 | Val: 0.3835/0.7986 | LR: 4.97e-04
        Epoch 10/100 | Train: 0.3365/0.9361 | Val: 0.3767/0.7986 | LR: 4.88e-04
        Epoch 11/100 | Train: 0.3284/0.9459 | Val: 0.3695/0.8020 | LR: 4.73e-04
        Epoch 12/100 | Train: 0.2705/0.9713 | Val: 0.3625/0.8044 | LR: 4.52e-04 ★
        Epoch 13/100 | Train: 0.2631/0.9743 | Val: 0.3559/0.8072 | LR: 4.27e-04 ★
        Epoch 14/100 | Train: 0.2646/0.9720 | Val: 0.3506/0.8071 | LR: 3.97e-04
        Epoch 15/100 | Train: 0.2474/0.9801 | Val: 0.3449/0.8104 | LR: 3.64e-04 ★
        Epoch 16/100 | Train: 0.2295/0.9821 | Val: 0.3392/0.8210 | LR: 3.27e-04 ★
        Epoch 17/100 | Train: 0.2166/0.9870 | Val: 0.3338/0.8210 | LR: 2.89e-04
        Epoch 18/100 | Train: 0.2337/0.9867 | Val: 0.3281/0.8210 | LR: 2.50e-04
        Epoch 19/100 | Train: 0.2225/0.9889 | Val: 0.3230/0.8314 | LR: 2.11e-04 ★
        Epoch 20/100 | Train: 0.2085/0.9899 | Val: 0.3186/0.8314 | LR: 1.73e-04
        Epoch 21/100 | Train: 0.1981/0.9926 | Val: 0.3141/0.8349 | LR: 1.37e-04 ★
        Epoch 22/100 | Train: 0.2111/0.9880 | Val: 0.3101/0.8451 | LR: 1.03e-04 ★
        Epoch 23/100 | Train: 0.2198/0.9925 | Val: 0.3064/0.8451 | LR: 7.33e-05
        Epoch 24/100 | Train: 0.1899/0.9954 | Val: 0.3029/0.8411 | LR: 4.78e-05
        Epoch 25/100 | Train: 0.1991/0.9922 | Val: 0.2994/0.8411 | LR: 2.73e-05
        Epoch 26/100 | Train: 0.1931/0.9954 | Val: 0.2963/0.8511 | LR: 1.23e-05 ★
        Epoch 27/100 | Train: 0.1981/0.9948 | Val: 0.2933/0.8535 | LR: 3.18e-06 ★
        Epoch 28/100 | Train: 0.1940/0.9939 | Val: 0.2907/0.8535 | LR: 5.00e-04
        Epoch 29/100 | Train: 0.1953/0.9941 | Val: 0.2879/0.8560 | LR: 4.99e-04 ★
        Epoch 30/100 | Train: 0.2163/0.9833 | Val: 0.2853/0.8560 | LR: 4.97e-04
        Epoch 31/100 | Train: 0.2085/0.9956 | Val: 0.2831/0.8590 | LR: 4.93e-04 ★
        Epoch 32/100 | Train: 0.2071/0.9932 | Val: 0.2809/0.8590 | LR: 4.88e-04
        Epoch 33/100 | Train: 0.1963/0.9936 | Val: 0.2788/0.8590 | LR: 4.81e-04
        Epoch 34/100 | Train: 0.1993/0.9904 | Val: 0.2770/0.8724 | LR: 4.73e-04 ★
        Epoch 35/100 | Train: 0.2225/0.9911 | Val: 0.2753/0.8724 | LR: 4.63e-04
        Epoch 36/100 | Train: 0.1982/0.9880 | Val: 0.2734/0.8724 | LR: 4.52e-04
        Epoch 37/100 | Train: 0.2039/0.9890 | Val: 0.2720/0.8724 | LR: 4.40e-04
        Epoch 38/100 | Train: 0.1983/0.9940 | Val: 0.2710/0.8724 | LR: 4.27e-04
        Epoch 39/100 | Train: 0.2006/0.9933 | Val: 0.2699/0.8724 | LR: 4.12e-04
        Epoch 40/100 | Train: 0.1955/0.9936 | Val: 0.2691/0.8699 | LR: 3.97e-04
        Epoch 41/100 | Train: 0.1914/0.9939 | Val: 0.2683/0.8699 | LR: 3.81e-04
        Epoch 42/100 | Train: 0.1814/0.9986 | Val: 0.2678/0.8704 | LR: 3.64e-04
        Epoch 43/100 | Train: 0.1960/0.9967 | Val: 0.2674/0.8731 | LR: 3.46e-04 ★
        Epoch 44/100 | Train: 0.2001/0.9932 | Val: 0.2666/0.8731 | LR: 3.27e-04
        Epoch 45/100 | Train: 0.2031/0.9944 | Val: 0.2657/0.8731 | LR: 3.08e-04
        Epoch 46/100 | Train: 0.1787/0.9971 | Val: 0.2649/0.8756 | LR: 2.89e-04 ★
        Epoch 47/100 | Train: 0.1864/0.9957 | Val: 0.2642/0.8756 | LR: 2.70e-04
        Epoch 48/100 | Train: 0.1709/0.9991 | Val: 0.2634/0.8756 | LR: 2.50e-04
        Epoch 49/100 | Train: 0.1792/0.9967 | Val: 0.2628/0.8780 | LR: 2.30e-04 ★
        Epoch 50/100 | Train: 0.1864/0.9966 | Val: 0.2621/0.8805 | LR: 2.11e-04 ★
        Epoch 51/100 | Train: 0.1774/0.9995 | Val: 0.2609/0.8829 | LR: 1.92e-04 ★
        Epoch 52/100 | Train: 0.1763/0.9994 | Val: 0.2598/0.8899 | LR: 1.73e-04 ★
        Epoch 53/100 | Train: 0.1780/1.0000 | Val: 0.2591/0.8921 | LR: 1.54e-04 ★
        Epoch 54/100 | Train: 0.1749/0.9964 | Val: 0.2582/0.8921 | LR: 1.37e-04
        Epoch 55/100 | Train: 0.1830/0.9984 | Val: 0.2575/0.8946 | LR: 1.19e-04 ★
        Epoch 56/100 | Train: 0.1742/0.9991 | Val: 0.2570/0.8992 | LR: 1.03e-04 ★
        Epoch 57/100 | Train: 0.1720/1.0000 | Val: 0.2564/0.8992 | LR: 8.77e-05
        Epoch 58/100 | Train: 0.1675/1.0000 | Val: 0.2565/0.9039 | LR: 7.33e-05 ★
        Epoch 59/100 | Train: 0.1721/0.9985 | Val: 0.2563/0.9039 | LR: 6.00e-05
        Epoch 60/100 | Train: 0.1710/0.9987 | Val: 0.2563/0.9039 | LR: 4.78e-05
        Epoch 61/100 | Train: 0.1778/0.9980 | Val: 0.2559/0.9039 | LR: 3.69e-05
        Epoch 62/100 | Train: 0.1743/1.0000 | Val: 0.2556/0.9039 | LR: 2.73e-05
        Epoch 63/100 | Train: 0.1753/0.9983 | Val: 0.2553/0.9039 | LR: 1.91e-05
        Epoch 64/100 | Train: 0.1763/0.9989 | Val: 0.2551/0.9039 | LR: 1.23e-05
        Epoch 65/100 | Train: 0.1716/0.9995 | Val: 0.2549/0.9039 | LR: 7.01e-06
        Epoch 66/100 | Train: 0.1727/0.9993 | Val: 0.2550/0.9008 | LR: 3.18e-06
        Epoch 67/100 | Train: 0.1724/0.9993 | Val: 0.2551/0.9008 | LR: 8.71e-07
        Epoch 68/100 | Train: 0.1762/0.9977 | Val: 0.2550/0.9008 | LR: 5.00e-04
        Epoch 69/100 | Train: 0.1816/0.9963 | Val: 0.2548/0.9008 | LR: 5.00e-04
        Epoch 70/100 | Train: 0.1870/0.9942 | Val: 0.2539/0.9008 | LR: 4.99e-04
        Epoch 71/100 | Train: 0.1759/0.9989 | Val: 0.2539/0.9008 | LR: 4.98e-04
        Epoch 72/100 | Train: 0.1770/0.9986 | Val: 0.2532/0.9008 | LR: 4.97e-04
        Epoch 73/100 | Train: 0.1885/0.9980 | Val: 0.2529/0.9008 | LR: 4.95e-04
        Early stopping at epoch 73 (patience=15)
    
      Best: Epoch 58, Val F1: 0.9039
      ConvNeXt fold 1: val F1 = 0.9039
    
    ============================================================
    FOLD 2 — convnext (convnext_base.fb_in22k)
    Epochs: 100 | Freeze: 5 | Warmup: 3 | CutMix: True
    ============================================================
    
      Total:    87.6M params
      Head:     6.2K params (lr=5e-04)
      Backbone: 87.6M params (lr=5e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (5 epochs, no CutMix)
        Epoch 1/5 | Train: 2.7350/0.3835 | Val: 1.5000/0.1351
        Epoch 2/5 | Train: 1.7704/0.6869 | Val: 1.4734/0.1497
        Epoch 3/5 | Train: 1.3036/0.7486 | Val: 1.4338/0.1701
        Epoch 4/5 | Train: 1.1302/0.7833 | Val: 1.3862/0.2170
        Epoch 5/5 | Train: 0.9788/0.8138 | Val: 1.3339/0.2495
    
      Stage 2: Full fine-tuning (95 epochs, CutMix=True)
        Epoch 06/100 | Train: 0.7908/0.8327 | Val: 0.3653/0.8127 | LR: 2.00e-04 ★
        Epoch 07/100 | Train: 0.6393/0.8484 | Val: 0.3625/0.8161 | LR: 3.50e-04 ★
        Epoch 08/100 | Train: 0.4601/0.8885 | Val: 0.3576/0.8229 | LR: 5.00e-04 ★
        Epoch 09/100 | Train: 0.4138/0.9141 | Val: 0.3516/0.8265 | LR: 4.97e-04 ★
        Epoch 10/100 | Train: 0.3428/0.9398 | Val: 0.3450/0.8282 | LR: 4.88e-04 ★
        Epoch 11/100 | Train: 0.3241/0.9472 | Val: 0.3383/0.8260 | LR: 4.73e-04
        Epoch 12/100 | Train: 0.2798/0.9671 | Val: 0.3321/0.8271 | LR: 4.52e-04
        Epoch 13/100 | Train: 0.2811/0.9644 | Val: 0.3259/0.8308 | LR: 4.27e-04 ★
        Epoch 14/100 | Train: 0.2554/0.9696 | Val: 0.3200/0.8308 | LR: 3.97e-04
        Epoch 15/100 | Train: 0.2421/0.9846 | Val: 0.3142/0.8276 | LR: 3.64e-04
        Epoch 16/100 | Train: 0.2357/0.9872 | Val: 0.3083/0.8337 | LR: 3.27e-04 ★
        Epoch 17/100 | Train: 0.2146/0.9875 | Val: 0.3022/0.8337 | LR: 2.89e-04
        Epoch 18/100 | Train: 0.2068/0.9894 | Val: 0.2966/0.8478 | LR: 2.50e-04 ★
        Epoch 19/100 | Train: 0.2207/0.9889 | Val: 0.2914/0.8515 | LR: 2.11e-04 ★
        Epoch 20/100 | Train: 0.2153/0.9879 | Val: 0.2861/0.8507 | LR: 1.73e-04
        Epoch 21/100 | Train: 0.2042/0.9901 | Val: 0.2810/0.8507 | LR: 1.37e-04
        Epoch 22/100 | Train: 0.2075/0.9915 | Val: 0.2762/0.8507 | LR: 1.03e-04
        Epoch 23/100 | Train: 0.1927/0.9958 | Val: 0.2719/0.8569 | LR: 7.33e-05 ★
        Epoch 24/100 | Train: 0.1974/0.9879 | Val: 0.2676/0.8616 | LR: 4.78e-05 ★
        Epoch 25/100 | Train: 0.2077/0.9928 | Val: 0.2638/0.8624 | LR: 2.73e-05 ★
        Epoch 26/100 | Train: 0.1924/0.9944 | Val: 0.2599/0.8686 | LR: 1.23e-05 ★
        Epoch 27/100 | Train: 0.1912/0.9975 | Val: 0.2563/0.8724 | LR: 3.18e-06 ★
        Epoch 28/100 | Train: 0.1784/0.9969 | Val: 0.2526/0.8748 | LR: 5.00e-04 ★
        Epoch 29/100 | Train: 0.2046/0.9957 | Val: 0.2495/0.8748 | LR: 4.99e-04
        Epoch 30/100 | Train: 0.1980/0.9891 | Val: 0.2463/0.8848 | LR: 4.97e-04 ★
        Epoch 31/100 | Train: 0.1984/0.9905 | Val: 0.2433/0.8874 | LR: 4.93e-04 ★
        Epoch 32/100 | Train: 0.2000/0.9922 | Val: 0.2400/0.8874 | LR: 4.88e-04
        Epoch 33/100 | Train: 0.1810/0.9955 | Val: 0.2370/0.8874 | LR: 4.81e-04
        Epoch 34/100 | Train: 0.2076/0.9913 | Val: 0.2345/0.8874 | LR: 4.73e-04
        Epoch 35/100 | Train: 0.2054/0.9930 | Val: 0.2321/0.8912 | LR: 4.63e-04 ★
        Epoch 36/100 | Train: 0.2374/0.9867 | Val: 0.2297/0.8936 | LR: 4.52e-04 ★
        Epoch 37/100 | Train: 0.2029/0.9937 | Val: 0.2280/0.8936 | LR: 4.40e-04
        Epoch 38/100 | Train: 0.1939/0.9975 | Val: 0.2260/0.8936 | LR: 4.27e-04
        Epoch 39/100 | Train: 0.1950/0.9942 | Val: 0.2240/0.8936 | LR: 4.12e-04
        Epoch 40/100 | Train: 0.2016/0.9931 | Val: 0.2221/0.8936 | LR: 3.97e-04
        Epoch 41/100 | Train: 0.1910/0.9932 | Val: 0.2207/0.9000 | LR: 3.81e-04 ★
        Epoch 42/100 | Train: 0.1872/0.9979 | Val: 0.2192/0.9025 | LR: 3.64e-04 ★
        Epoch 43/100 | Train: 0.1913/0.9982 | Val: 0.2179/0.9025 | LR: 3.46e-04
        Epoch 44/100 | Train: 0.1872/0.9968 | Val: 0.2168/0.9025 | LR: 3.27e-04
        Epoch 45/100 | Train: 0.1980/0.9930 | Val: 0.2159/0.9025 | LR: 3.08e-04
        Epoch 46/100 | Train: 0.1893/0.9963 | Val: 0.2152/0.9025 | LR: 2.89e-04
        Epoch 47/100 | Train: 0.1823/0.9962 | Val: 0.2147/0.8987 | LR: 2.70e-04
        Epoch 48/100 | Train: 0.1926/0.9978 | Val: 0.2139/0.8987 | LR: 2.50e-04
        Epoch 49/100 | Train: 0.1802/0.9983 | Val: 0.2135/0.8987 | LR: 2.30e-04
        Epoch 50/100 | Train: 0.1781/0.9976 | Val: 0.2131/0.8987 | LR: 2.11e-04
        Epoch 51/100 | Train: 0.1871/0.9952 | Val: 0.2130/0.8915 | LR: 1.92e-04
        Epoch 52/100 | Train: 0.1756/1.0000 | Val: 0.2125/0.8915 | LR: 1.73e-04
        Epoch 53/100 | Train: 0.1834/0.9967 | Val: 0.2124/0.8915 | LR: 1.54e-04
        Epoch 54/100 | Train: 0.1756/0.9992 | Val: 0.2120/0.8915 | LR: 1.37e-04
        Epoch 55/100 | Train: 0.1742/0.9978 | Val: 0.2118/0.8964 | LR: 1.19e-04
        Epoch 56/100 | Train: 0.1719/0.9988 | Val: 0.2115/0.8964 | LR: 1.03e-04
        Epoch 57/100 | Train: 0.1728/0.9993 | Val: 0.2110/0.8964 | LR: 8.77e-05
        Early stopping at epoch 57 (patience=15)
    
      Best: Epoch 42, Val F1: 0.9025
      ConvNeXt fold 2: val F1 = 0.9025
    
    ============================================================
    FOLD 3 — convnext (convnext_base.fb_in22k)
    Epochs: 100 | Freeze: 5 | Warmup: 3 | CutMix: True
    ============================================================
    
      Total:    87.6M params
      Head:     6.2K params (lr=5e-04)
      Backbone: 87.6M params (lr=5e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (5 epochs, no CutMix)
        Epoch 1/5 | Train: 2.6801/0.4174 | Val: 1.4792/0.1649
        Epoch 2/5 | Train: 1.7111/0.7083 | Val: 1.4518/0.1789
        Epoch 3/5 | Train: 1.2709/0.7580 | Val: 1.4128/0.2077
        Epoch 4/5 | Train: 1.0567/0.7882 | Val: 1.3660/0.2258
        Epoch 5/5 | Train: 0.9293/0.8026 | Val: 1.3148/0.2748
    
      Stage 2: Full fine-tuning (95 epochs, CutMix=True)
        Epoch 06/100 | Train: 0.7948/0.8280 | Val: 0.4208/0.7879 | LR: 2.00e-04 ★
        Epoch 07/100 | Train: 0.5809/0.8604 | Val: 0.4177/0.7879 | LR: 3.50e-04
        Epoch 08/100 | Train: 0.4792/0.8971 | Val: 0.4131/0.7879 | LR: 5.00e-04
        Epoch 09/100 | Train: 0.3729/0.9236 | Val: 0.4076/0.7913 | LR: 4.97e-04 ★
        Epoch 10/100 | Train: 0.3266/0.9431 | Val: 0.4015/0.7941 | LR: 4.88e-04 ★
        Epoch 11/100 | Train: 0.3095/0.9497 | Val: 0.3950/0.7941 | LR: 4.73e-04
        Epoch 12/100 | Train: 0.2883/0.9593 | Val: 0.3893/0.7941 | LR: 4.52e-04
        Epoch 13/100 | Train: 0.2467/0.9704 | Val: 0.3830/0.7941 | LR: 4.27e-04
        Epoch 14/100 | Train: 0.2455/0.9817 | Val: 0.3767/0.7970 | LR: 3.97e-04 ★
        Epoch 15/100 | Train: 0.2596/0.9746 | Val: 0.3705/0.8007 | LR: 3.64e-04 ★
        Epoch 16/100 | Train: 0.2199/0.9871 | Val: 0.3642/0.8090 | LR: 3.27e-04 ★
        Epoch 17/100 | Train: 0.2286/0.9883 | Val: 0.3585/0.8090 | LR: 2.89e-04
        Epoch 18/100 | Train: 0.2154/0.9828 | Val: 0.3528/0.8090 | LR: 2.50e-04
        Epoch 19/100 | Train: 0.2107/0.9890 | Val: 0.3474/0.8136 | LR: 2.11e-04 ★
        Epoch 20/100 | Train: 0.2009/0.9904 | Val: 0.3423/0.8132 | LR: 1.73e-04
        Epoch 21/100 | Train: 0.2126/0.9866 | Val: 0.3375/0.8169 | LR: 1.37e-04 ★
        Epoch 22/100 | Train: 0.2052/0.9917 | Val: 0.3332/0.8251 | LR: 1.03e-04 ★
        Epoch 23/100 | Train: 0.2002/0.9927 | Val: 0.3290/0.8260 | LR: 7.33e-05 ★
        Epoch 24/100 | Train: 0.1953/0.9933 | Val: 0.3250/0.8291 | LR: 4.78e-05 ★
        Epoch 25/100 | Train: 0.2074/0.9935 | Val: 0.3213/0.8346 | LR: 2.73e-05 ★
        Epoch 26/100 | Train: 0.1904/0.9967 | Val: 0.3177/0.8346 | LR: 1.23e-05
        Epoch 27/100 | Train: 0.1907/0.9926 | Val: 0.3141/0.8346 | LR: 3.18e-06
        Epoch 28/100 | Train: 0.1950/0.9971 | Val: 0.3108/0.8346 | LR: 5.00e-04
        Epoch 29/100 | Train: 0.2049/0.9932 | Val: 0.3077/0.8386 | LR: 4.99e-04 ★
        Epoch 30/100 | Train: 0.2131/0.9954 | Val: 0.3047/0.8414 | LR: 4.97e-04 ★
        Epoch 31/100 | Train: 0.2149/0.9895 | Val: 0.3023/0.8423 | LR: 4.93e-04 ★
        Epoch 32/100 | Train: 0.2222/0.9882 | Val: 0.3006/0.8361 | LR: 4.88e-04
        Epoch 33/100 | Train: 0.1863/0.9911 | Val: 0.2983/0.8361 | LR: 4.81e-04
        Epoch 34/100 | Train: 0.2045/0.9907 | Val: 0.2964/0.8418 | LR: 4.73e-04
        Epoch 35/100 | Train: 0.2092/0.9885 | Val: 0.2950/0.8418 | LR: 4.63e-04
        Epoch 36/100 | Train: 0.1989/0.9938 | Val: 0.2936/0.8453 | LR: 4.52e-04 ★
        Epoch 37/100 | Train: 0.1875/0.9937 | Val: 0.2925/0.8453 | LR: 4.40e-04
        Epoch 38/100 | Train: 0.1895/0.9961 | Val: 0.2910/0.8453 | LR: 4.27e-04
        Epoch 39/100 | Train: 0.2097/0.9916 | Val: 0.2893/0.8475 | LR: 4.12e-04 ★
        Epoch 40/100 | Train: 0.1945/0.9944 | Val: 0.2882/0.8475 | LR: 3.97e-04
        Epoch 41/100 | Train: 0.1971/0.9952 | Val: 0.2870/0.8471 | LR: 3.81e-04
        Epoch 42/100 | Train: 0.1873/0.9951 | Val: 0.2855/0.8471 | LR: 3.64e-04
        Epoch 43/100 | Train: 0.1801/0.9947 | Val: 0.2839/0.8528 | LR: 3.46e-04 ★
        Epoch 44/100 | Train: 0.1837/0.9945 | Val: 0.2823/0.8564 | LR: 3.27e-04 ★
        Epoch 45/100 | Train: 0.2006/0.9970 | Val: 0.2815/0.8592 | LR: 3.08e-04 ★
        Epoch 46/100 | Train: 0.1777/0.9981 | Val: 0.2805/0.8624 | LR: 2.89e-04 ★
        Epoch 47/100 | Train: 0.1833/0.9964 | Val: 0.2801/0.8624 | LR: 2.70e-04
        Epoch 48/100 | Train: 0.1830/0.9968 | Val: 0.2794/0.8624 | LR: 2.50e-04
        Epoch 49/100 | Train: 0.1735/0.9980 | Val: 0.2795/0.8624 | LR: 2.30e-04
        Epoch 50/100 | Train: 0.1743/0.9943 | Val: 0.2792/0.8624 | LR: 2.11e-04
        Epoch 51/100 | Train: 0.1704/0.9974 | Val: 0.2793/0.8597 | LR: 1.92e-04
        Epoch 52/100 | Train: 0.1772/0.9975 | Val: 0.2787/0.8597 | LR: 1.73e-04
        Epoch 53/100 | Train: 0.1815/0.9987 | Val: 0.2787/0.8624 | LR: 1.54e-04
        Epoch 54/100 | Train: 0.1738/0.9979 | Val: 0.2783/0.8657 | LR: 1.37e-04 ★
        Epoch 55/100 | Train: 0.1787/0.9982 | Val: 0.2779/0.8697 | LR: 1.19e-04 ★
        Epoch 56/100 | Train: 0.1745/0.9979 | Val: 0.2778/0.8697 | LR: 1.03e-04
        Epoch 57/100 | Train: 0.1739/0.9978 | Val: 0.2778/0.8697 | LR: 8.77e-05
        Epoch 58/100 | Train: 0.1712/1.0000 | Val: 0.2776/0.8697 | LR: 7.33e-05
        Epoch 59/100 | Train: 0.1753/0.9967 | Val: 0.2776/0.8670 | LR: 6.00e-05
        Epoch 60/100 | Train: 0.1773/0.9980 | Val: 0.2776/0.8670 | LR: 4.78e-05
        Epoch 61/100 | Train: 0.1725/0.9981 | Val: 0.2778/0.8691 | LR: 3.69e-05
        Epoch 62/100 | Train: 0.1717/0.9993 | Val: 0.2778/0.8691 | LR: 2.73e-05
        Epoch 63/100 | Train: 0.1718/1.0000 | Val: 0.2779/0.8691 | LR: 1.91e-05
        Epoch 64/100 | Train: 0.1716/1.0000 | Val: 0.2781/0.8691 | LR: 1.23e-05
        Epoch 65/100 | Train: 0.1701/0.9992 | Val: 0.2781/0.8691 | LR: 7.01e-06
        Epoch 66/100 | Train: 0.1743/0.9991 | Val: 0.2781/0.8691 | LR: 3.18e-06
        Epoch 67/100 | Train: 0.1694/0.9989 | Val: 0.2780/0.8691 | LR: 8.71e-07
        Epoch 68/100 | Train: 0.1745/0.9984 | Val: 0.2780/0.8691 | LR: 5.00e-04
        Epoch 69/100 | Train: 0.1762/0.9974 | Val: 0.2783/0.8691 | LR: 5.00e-04
        Epoch 70/100 | Train: 0.1798/0.9971 | Val: 0.2782/0.8718 | LR: 4.99e-04 ★
        Epoch 71/100 | Train: 0.1924/0.9968 | Val: 0.2780/0.8718 | LR: 4.98e-04
        Epoch 72/100 | Train: 0.1861/0.9943 | Val: 0.2772/0.8753 | LR: 4.97e-04 ★
        Epoch 73/100 | Train: 0.1837/0.9957 | Val: 0.2770/0.8753 | LR: 4.95e-04
        Epoch 74/100 | Train: 0.1773/0.9975 | Val: 0.2769/0.8753 | LR: 4.93e-04
        Epoch 75/100 | Train: 0.1852/0.9959 | Val: 0.2776/0.8780 | LR: 4.91e-04 ★
        Epoch 76/100 | Train: 0.1800/0.9966 | Val: 0.2778/0.8780 | LR: 4.88e-04
        Epoch 77/100 | Train: 0.1818/0.9980 | Val: 0.2778/0.8780 | LR: 4.85e-04
        Epoch 78/100 | Train: 0.1871/0.9976 | Val: 0.2780/0.8807 | LR: 4.81e-04 ★
        Epoch 79/100 | Train: 0.1787/0.9988 | Val: 0.2785/0.8807 | LR: 4.77e-04
        Epoch 80/100 | Train: 0.1817/0.9945 | Val: 0.2790/0.8842 | LR: 4.73e-04 ★
        Epoch 81/100 | Train: 0.1733/0.9978 | Val: 0.2797/0.8842 | LR: 4.68e-04
        Epoch 82/100 | Train: 0.1731/0.9975 | Val: 0.2815/0.8842 | LR: 4.63e-04
        Epoch 83/100 | Train: 0.1772/1.0000 | Val: 0.2827/0.8815 | LR: 4.58e-04
        Epoch 84/100 | Train: 0.1786/0.9981 | Val: 0.2843/0.8754 | LR: 4.52e-04
        Epoch 85/100 | Train: 0.1746/0.9978 | Val: 0.2853/0.8754 | LR: 4.46e-04
        Epoch 86/100 | Train: 0.1749/0.9979 | Val: 0.2860/0.8754 | LR: 4.40e-04
        Epoch 87/100 | Train: 0.1675/0.9993 | Val: 0.2869/0.8781 | LR: 4.34e-04
        Epoch 88/100 | Train: 0.1693/0.9980 | Val: 0.2882/0.8781 | LR: 4.27e-04
        Epoch 89/100 | Train: 0.1720/0.9991 | Val: 0.2895/0.8721 | LR: 4.20e-04
        Epoch 90/100 | Train: 0.1803/0.9967 | Val: 0.2902/0.8721 | LR: 4.12e-04
        Epoch 91/100 | Train: 0.1762/1.0000 | Val: 0.2918/0.8721 | LR: 4.05e-04
        Epoch 92/100 | Train: 0.1772/0.9976 | Val: 0.2928/0.8721 | LR: 3.97e-04
        Epoch 93/100 | Train: 0.1749/0.9980 | Val: 0.2933/0.8721 | LR: 3.89e-04
        Epoch 94/100 | Train: 0.1759/0.9977 | Val: 0.2939/0.8721 | LR: 3.81e-04
        Epoch 95/100 | Train: 0.1754/1.0000 | Val: 0.2949/0.8786 | LR: 3.72e-04
        Early stopping at epoch 95 (patience=15)
    
      Best: Epoch 80, Val F1: 0.8842
      ConvNeXt fold 3: val F1 = 0.8842
    
    ============================================================
    FOLD 4 — convnext (convnext_base.fb_in22k)
    Epochs: 100 | Freeze: 5 | Warmup: 3 | CutMix: True
    ============================================================
    
      Total:    87.6M params
      Head:     6.2K params (lr=5e-04)
      Backbone: 87.6M params (lr=5e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (5 epochs, no CutMix)
        Epoch 1/5 | Train: 2.6699/0.4264 | Val: 1.4474/0.1810
        Epoch 2/5 | Train: 1.7012/0.7062 | Val: 1.4197/0.2187
        Epoch 3/5 | Train: 1.2611/0.7512 | Val: 1.3792/0.2331
        Epoch 4/5 | Train: 1.0511/0.7836 | Val: 1.3306/0.2588
        Epoch 5/5 | Train: 0.9404/0.7962 | Val: 1.2769/0.3307
    
      Stage 2: Full fine-tuning (95 epochs, CutMix=True)
        Epoch 06/100 | Train: 0.8141/0.8288 | Val: 0.3289/0.8693 | LR: 2.00e-04 ★
        Epoch 07/100 | Train: 0.6144/0.8569 | Val: 0.3261/0.8693 | LR: 3.50e-04
        Epoch 08/100 | Train: 0.4761/0.8872 | Val: 0.3215/0.8730 | LR: 5.00e-04 ★
        Epoch 09/100 | Train: 0.4030/0.9172 | Val: 0.3159/0.8814 | LR: 4.97e-04 ★
        Epoch 10/100 | Train: 0.3385/0.9361 | Val: 0.3096/0.8840 | LR: 4.88e-04 ★
        Epoch 11/100 | Train: 0.3324/0.9467 | Val: 0.3032/0.8872 | LR: 4.73e-04 ★
        Epoch 12/100 | Train: 0.2788/0.9666 | Val: 0.2959/0.8872 | LR: 4.52e-04
        Epoch 13/100 | Train: 0.2629/0.9714 | Val: 0.2884/0.8899 | LR: 4.27e-04 ★
        Epoch 14/100 | Train: 0.2487/0.9729 | Val: 0.2813/0.8899 | LR: 3.97e-04
        Epoch 15/100 | Train: 0.2458/0.9814 | Val: 0.2748/0.8969 | LR: 3.64e-04 ★
        Epoch 16/100 | Train: 0.2267/0.9815 | Val: 0.2682/0.8995 | LR: 3.27e-04 ★
        Epoch 17/100 | Train: 0.2279/0.9846 | Val: 0.2619/0.8995 | LR: 2.89e-04
        Epoch 18/100 | Train: 0.2165/0.9857 | Val: 0.2556/0.9022 | LR: 2.50e-04 ★
        Epoch 19/100 | Train: 0.2195/0.9821 | Val: 0.2497/0.9014 | LR: 2.11e-04
        Epoch 20/100 | Train: 0.2180/0.9871 | Val: 0.2438/0.9014 | LR: 1.73e-04
        Epoch 21/100 | Train: 0.2055/0.9891 | Val: 0.2381/0.9014 | LR: 1.37e-04
        Epoch 22/100 | Train: 0.1975/0.9953 | Val: 0.2330/0.9049 | LR: 1.03e-04 ★
        Epoch 23/100 | Train: 0.1938/0.9931 | Val: 0.2279/0.9049 | LR: 7.33e-05
        Epoch 24/100 | Train: 0.1911/0.9952 | Val: 0.2232/0.9085 | LR: 4.78e-05 ★
        Epoch 25/100 | Train: 0.1925/0.9948 | Val: 0.2186/0.9109 | LR: 2.73e-05 ★
        Epoch 26/100 | Train: 0.1908/0.9942 | Val: 0.2143/0.9135 | LR: 1.23e-05 ★
        Epoch 27/100 | Train: 0.1885/0.9976 | Val: 0.2102/0.9135 | LR: 3.18e-06
        Epoch 28/100 | Train: 0.2044/0.9949 | Val: 0.2063/0.9135 | LR: 5.00e-04
        Epoch 29/100 | Train: 0.2098/0.9891 | Val: 0.2027/0.9135 | LR: 4.99e-04
        Epoch 30/100 | Train: 0.2194/0.9794 | Val: 0.1989/0.9135 | LR: 4.97e-04
        Epoch 31/100 | Train: 0.2206/0.9913 | Val: 0.1955/0.9135 | LR: 4.93e-04
        Epoch 32/100 | Train: 0.1967/0.9957 | Val: 0.1920/0.9162 | LR: 4.88e-04 ★
        Epoch 33/100 | Train: 0.2076/0.9944 | Val: 0.1893/0.9162 | LR: 4.81e-04
        Epoch 34/100 | Train: 0.2015/0.9964 | Val: 0.1869/0.9162 | LR: 4.73e-04
        Epoch 35/100 | Train: 0.2030/0.9925 | Val: 0.1847/0.9162 | LR: 4.63e-04
        Epoch 36/100 | Train: 0.1937/0.9956 | Val: 0.1825/0.9174 | LR: 4.52e-04 ★
        Epoch 37/100 | Train: 0.1962/0.9946 | Val: 0.1804/0.9174 | LR: 4.40e-04
        Epoch 38/100 | Train: 0.1942/0.9943 | Val: 0.1785/0.9174 | LR: 4.27e-04
        Epoch 39/100 | Train: 0.1892/0.9916 | Val: 0.1768/0.9174 | LR: 4.12e-04
        Epoch 40/100 | Train: 0.1894/0.9957 | Val: 0.1753/0.9168 | LR: 3.97e-04
        Epoch 41/100 | Train: 0.1882/0.9953 | Val: 0.1735/0.9168 | LR: 3.81e-04
        Epoch 42/100 | Train: 0.1861/0.9966 | Val: 0.1718/0.9168 | LR: 3.64e-04
        Epoch 43/100 | Train: 0.1868/0.9967 | Val: 0.1705/0.9168 | LR: 3.46e-04
        Epoch 44/100 | Train: 0.1840/0.9972 | Val: 0.1694/0.9168 | LR: 3.27e-04
        Epoch 45/100 | Train: 0.1812/0.9980 | Val: 0.1681/0.9168 | LR: 3.08e-04
        Epoch 46/100 | Train: 0.1878/0.9944 | Val: 0.1665/0.9168 | LR: 2.89e-04
        Epoch 47/100 | Train: 0.1836/0.9971 | Val: 0.1654/0.9168 | LR: 2.70e-04
        Epoch 48/100 | Train: 0.1819/0.9982 | Val: 0.1644/0.9168 | LR: 2.50e-04
        Epoch 49/100 | Train: 0.1808/0.9944 | Val: 0.1632/0.9168 | LR: 2.30e-04
        Epoch 50/100 | Train: 0.1781/0.9989 | Val: 0.1623/0.9203 | LR: 2.11e-04 ★
        Epoch 51/100 | Train: 0.1736/0.9975 | Val: 0.1614/0.9203 | LR: 1.92e-04
        Epoch 52/100 | Train: 0.1771/0.9973 | Val: 0.1605/0.9241 | LR: 1.73e-04 ★
        Epoch 53/100 | Train: 0.1743/0.9991 | Val: 0.1596/0.9273 | LR: 1.54e-04 ★
        Epoch 54/100 | Train: 0.1808/0.9967 | Val: 0.1590/0.9273 | LR: 1.37e-04
        Epoch 55/100 | Train: 0.1796/0.9984 | Val: 0.1583/0.9273 | LR: 1.19e-04
        Epoch 56/100 | Train: 0.1808/0.9975 | Val: 0.1575/0.9273 | LR: 1.03e-04
        Epoch 57/100 | Train: 0.1743/0.9976 | Val: 0.1568/0.9273 | LR: 8.77e-05
        Epoch 58/100 | Train: 0.1731/0.9990 | Val: 0.1562/0.9273 | LR: 7.33e-05
        Epoch 59/100 | Train: 0.1723/0.9978 | Val: 0.1557/0.9273 | LR: 6.00e-05
        Epoch 60/100 | Train: 0.1804/1.0000 | Val: 0.1551/0.9241 | LR: 4.78e-05
        Epoch 61/100 | Train: 0.1752/1.0000 | Val: 0.1547/0.9241 | LR: 3.69e-05
        Epoch 62/100 | Train: 0.1737/0.9991 | Val: 0.1541/0.9241 | LR: 2.73e-05
        Epoch 63/100 | Train: 0.1722/1.0000 | Val: 0.1535/0.9241 | LR: 1.91e-05
        Epoch 64/100 | Train: 0.1727/1.0000 | Val: 0.1530/0.9241 | LR: 1.23e-05
        Epoch 65/100 | Train: 0.1725/1.0000 | Val: 0.1525/0.9241 | LR: 7.01e-06
        Epoch 66/100 | Train: 0.1693/0.9988 | Val: 0.1522/0.9241 | LR: 3.18e-06
        Epoch 67/100 | Train: 0.1716/0.9979 | Val: 0.1518/0.9241 | LR: 8.71e-07
        Epoch 68/100 | Train: 0.1713/1.0000 | Val: 0.1515/0.9265 | LR: 5.00e-04
        Early stopping at epoch 68 (patience=15)
    
      Best: Epoch 53, Val F1: 0.9273
      ConvNeXt fold 4: val F1 = 0.9273
    
    ConvNeXt summary: mean=0.9073 ± 0.0148
      exp06 baseline: mean=0.9184 ± 0.0149


## 7 · Train EVA-02-Base  (70 epochs)


```python
eva02_results = {}

for fold in range(N_FOLDS):
    result = train_fold(
        fold         = fold,
        model_key    = 'eva02',
        train_df     = train_df,
        class_weights= class_weights,
        use_cutmix   = True,
        use_sampler  = False,
        exp_id       = EXP_ID,
    )
    eva02_results[fold] = result
    print(f'  EVA-02 fold {fold}: val F1 = {result["best_f1"]:.4f}')

scores = [r['best_f1'] for r in eva02_results.values()]
print(f'\nEVA-02 summary: mean={np.mean(scores):.4f} ± {np.std(scores):.4f}')
print(f'  exp06 baseline: mean=0.9267 ± 0.0239')
```

    
    ============================================================
    FOLD 0 — eva02 (eva02_base_patch14_224.mim_in22k)
    Epochs: 90 | Freeze: 10 | Warmup: 5 | CutMix: True
    ============================================================



    model.safetensors:   0%|          | 0.00/343M [00:00<?, ?B/s]


    
      Total:    85.8M params
      Head:     4.6K params (lr=5e-04)
      Backbone: 85.8M params (lr=2e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (10 epochs, no CutMix)
        Epoch 1/10 | Train: 2.1786/0.5305 | Val: 1.3690/0.7326
        Epoch 2/10 | Train: 1.2700/0.7320 | Val: 1.3031/0.7560
        Epoch 3/10 | Train: 1.0911/0.7443 | Val: 1.2266/0.7900
        Epoch 4/10 | Train: 0.9316/0.7750 | Val: 1.1481/0.8033
        Epoch 5/10 | Train: 0.8874/0.7875 | Val: 1.0694/0.8044
        Epoch 6/10 | Train: 0.8720/0.8031 | Val: 0.9944/0.8127
        Epoch 7/10 | Train: 0.8511/0.7880 | Val: 0.9242/0.8255
        Epoch 8/10 | Train: 0.7780/0.8185 | Val: 0.8597/0.8245
        Epoch 9/10 | Train: 0.7615/0.8064 | Val: 0.7994/0.8323
        Epoch 10/10 | Train: 0.7098/0.8293 | Val: 0.7445/0.8357
    
      Stage 2: Full fine-tuning (80 epochs, CutMix=True)
        Epoch 11/90 | Train: 0.6587/0.8409 | Val: 0.3739/0.8190 | LR: 1.40e-04 ★
        Epoch 12/90 | Train: 0.5965/0.8519 | Val: 0.3719/0.8190 | LR: 2.30e-04
        Epoch 13/90 | Train: 0.5271/0.8824 | Val: 0.3705/0.8226 | LR: 3.20e-04 ★
        Epoch 14/90 | Train: 0.4325/0.9041 | Val: 0.3691/0.8222 | LR: 4.10e-04
        Epoch 15/90 | Train: 0.3985/0.9133 | Val: 0.3662/0.8291 | LR: 5.00e-04 ★
        Epoch 16/90 | Train: 0.3897/0.9321 | Val: 0.3618/0.8291 | LR: 4.97e-04
        Epoch 17/90 | Train: 0.3280/0.9489 | Val: 0.3571/0.8319 | LR: 4.88e-04 ★
        Epoch 18/90 | Train: 0.3031/0.9634 | Val: 0.3520/0.8283 | LR: 4.73e-04
        Epoch 19/90 | Train: 0.3136/0.9509 | Val: 0.3470/0.8364 | LR: 4.52e-04 ★
        Epoch 20/90 | Train: 0.2986/0.9624 | Val: 0.3422/0.8364 | LR: 4.27e-04
        Epoch 21/90 | Train: 0.2913/0.9635 | Val: 0.3372/0.8443 | LR: 3.97e-04 ★
        Epoch 22/90 | Train: 0.2580/0.9701 | Val: 0.3317/0.8452 | LR: 3.64e-04 ★
        Epoch 23/90 | Train: 0.2688/0.9691 | Val: 0.3266/0.8433 | LR: 3.27e-04
        Epoch 24/90 | Train: 0.2459/0.9782 | Val: 0.3218/0.8434 | LR: 2.89e-04
        Epoch 25/90 | Train: 0.2386/0.9798 | Val: 0.3182/0.8522 | LR: 2.50e-04 ★
        Epoch 26/90 | Train: 0.2302/0.9845 | Val: 0.3141/0.8526 | LR: 2.11e-04 ★
        Epoch 27/90 | Train: 0.2299/0.9855 | Val: 0.3108/0.8526 | LR: 1.73e-04
        Epoch 28/90 | Train: 0.2167/0.9865 | Val: 0.3069/0.8526 | LR: 1.37e-04
        Epoch 29/90 | Train: 0.2241/0.9825 | Val: 0.3036/0.8526 | LR: 1.03e-04
        Epoch 30/90 | Train: 0.2264/0.9833 | Val: 0.2998/0.8602 | LR: 7.33e-05 ★
        Epoch 31/90 | Train: 0.2348/0.9863 | Val: 0.2962/0.8602 | LR: 4.78e-05
        Epoch 32/90 | Train: 0.2105/0.9890 | Val: 0.2931/0.8646 | LR: 2.73e-05 ★
        Epoch 33/90 | Train: 0.2178/0.9838 | Val: 0.2901/0.8674 | LR: 1.23e-05 ★
        Epoch 34/90 | Train: 0.2083/0.9898 | Val: 0.2874/0.8674 | LR: 3.18e-06
        Epoch 35/90 | Train: 0.2024/0.9924 | Val: 0.2848/0.8651 | LR: 5.00e-04
        Epoch 36/90 | Train: 0.2287/0.9842 | Val: 0.2827/0.8651 | LR: 4.99e-04
        Epoch 37/90 | Train: 0.2436/0.9816 | Val: 0.2805/0.8685 | LR: 4.97e-04 ★
        Epoch 38/90 | Train: 0.2286/0.9835 | Val: 0.2774/0.8750 | LR: 4.93e-04 ★
        Epoch 39/90 | Train: 0.2434/0.9784 | Val: 0.2750/0.8750 | LR: 4.88e-04
        Epoch 40/90 | Train: 0.2195/0.9893 | Val: 0.2726/0.8750 | LR: 4.81e-04
        Epoch 41/90 | Train: 0.2260/0.9852 | Val: 0.2693/0.8741 | LR: 4.73e-04
        Epoch 42/90 | Train: 0.2327/0.9856 | Val: 0.2672/0.8798 | LR: 4.63e-04 ★
        Epoch 43/90 | Train: 0.2371/0.9846 | Val: 0.2642/0.8798 | LR: 4.52e-04
        Epoch 44/90 | Train: 0.2139/0.9906 | Val: 0.2615/0.8798 | LR: 4.40e-04
        Epoch 45/90 | Train: 0.2170/0.9884 | Val: 0.2590/0.8798 | LR: 4.27e-04
        Epoch 46/90 | Train: 0.2149/0.9903 | Val: 0.2567/0.8821 | LR: 4.12e-04 ★
        Epoch 47/90 | Train: 0.2136/0.9903 | Val: 0.2545/0.8821 | LR: 3.97e-04
        Epoch 48/90 | Train: 0.2032/0.9937 | Val: 0.2519/0.8821 | LR: 3.81e-04
        Epoch 49/90 | Train: 0.2028/0.9934 | Val: 0.2504/0.8821 | LR: 3.64e-04
        Epoch 50/90 | Train: 0.2079/0.9932 | Val: 0.2496/0.8821 | LR: 3.46e-04
        Epoch 51/90 | Train: 0.2128/0.9874 | Val: 0.2491/0.8844 | LR: 3.27e-04 ★
        Epoch 52/90 | Train: 0.2054/0.9906 | Val: 0.2476/0.8844 | LR: 3.08e-04
        Epoch 53/90 | Train: 0.1987/0.9927 | Val: 0.2461/0.8903 | LR: 2.89e-04 ★
        Epoch 54/90 | Train: 0.2024/0.9895 | Val: 0.2448/0.8993 | LR: 2.70e-04 ★
        Epoch 55/90 | Train: 0.1946/0.9924 | Val: 0.2432/0.8993 | LR: 2.50e-04
        Epoch 56/90 | Train: 0.1868/0.9967 | Val: 0.2412/0.9084 | LR: 2.30e-04 ★
        Epoch 57/90 | Train: 0.1945/0.9920 | Val: 0.2396/0.9084 | LR: 2.11e-04
        Epoch 58/90 | Train: 0.1898/0.9966 | Val: 0.2378/0.9088 | LR: 1.92e-04 ★
        Epoch 59/90 | Train: 0.1800/0.9944 | Val: 0.2353/0.9114 | LR: 1.73e-04 ★
        Epoch 60/90 | Train: 0.1894/0.9949 | Val: 0.2328/0.9114 | LR: 1.54e-04
        Epoch 61/90 | Train: 0.1830/0.9951 | Val: 0.2306/0.9109 | LR: 1.37e-04
        Epoch 62/90 | Train: 0.1814/0.9947 | Val: 0.2284/0.9114 | LR: 1.19e-04
        Epoch 63/90 | Train: 0.1867/0.9975 | Val: 0.2264/0.9109 | LR: 1.03e-04
        Epoch 64/90 | Train: 0.1814/0.9971 | Val: 0.2237/0.9172 | LR: 8.77e-05 ★
        Epoch 65/90 | Train: 0.1800/0.9977 | Val: 0.2217/0.9172 | LR: 7.33e-05
        Epoch 66/90 | Train: 0.1841/0.9962 | Val: 0.2197/0.9197 | LR: 6.00e-05 ★
        Epoch 67/90 | Train: 0.1781/0.9979 | Val: 0.2181/0.9197 | LR: 4.78e-05
        Epoch 68/90 | Train: 0.1899/0.9967 | Val: 0.2169/0.9197 | LR: 3.69e-05
        Epoch 69/90 | Train: 0.1868/0.9951 | Val: 0.2155/0.9197 | LR: 2.73e-05
        Epoch 70/90 | Train: 0.1808/0.9968 | Val: 0.2141/0.9197 | LR: 1.91e-05
        Epoch 71/90 | Train: 0.1809/0.9956 | Val: 0.2130/0.9233 | LR: 1.23e-05 ★
        Epoch 72/90 | Train: 0.1798/0.9953 | Val: 0.2119/0.9233 | LR: 7.01e-06
        Epoch 73/90 | Train: 0.1799/0.9962 | Val: 0.2108/0.9233 | LR: 3.18e-06
        Epoch 74/90 | Train: 0.1795/0.9977 | Val: 0.2096/0.9233 | LR: 8.71e-07
        Epoch 75/90 | Train: 0.1736/0.9987 | Val: 0.2085/0.9269 | LR: 5.00e-04 ★
        Epoch 76/90 | Train: 0.1879/0.9915 | Val: 0.2078/0.9269 | LR: 5.00e-04
        Epoch 77/90 | Train: 0.2169/0.9882 | Val: 0.2073/0.9330 | LR: 4.99e-04 ★
        Epoch 78/90 | Train: 0.2034/0.9904 | Val: 0.2068/0.9330 | LR: 4.98e-04
        Epoch 79/90 | Train: 0.2078/0.9912 | Val: 0.2067/0.9330 | LR: 4.97e-04
        Epoch 80/90 | Train: 0.2065/0.9930 | Val: 0.2062/0.9330 | LR: 4.95e-04
        Epoch 81/90 | Train: 0.1982/0.9932 | Val: 0.2068/0.9330 | LR: 4.93e-04
        Epoch 82/90 | Train: 0.2185/0.9898 | Val: 0.2074/0.9330 | LR: 4.91e-04
        Epoch 83/90 | Train: 0.1930/0.9917 | Val: 0.2068/0.9330 | LR: 4.88e-04
        Epoch 84/90 | Train: 0.1967/0.9944 | Val: 0.2075/0.9330 | LR: 4.85e-04
        Epoch 85/90 | Train: 0.1935/0.9933 | Val: 0.2079/0.9330 | LR: 4.81e-04
        Epoch 86/90 | Train: 0.1933/0.9982 | Val: 0.2083/0.9330 | LR: 4.77e-04
        Epoch 87/90 | Train: 0.1898/0.9970 | Val: 0.2076/0.9330 | LR: 4.73e-04
        Epoch 88/90 | Train: 0.1861/0.9960 | Val: 0.2076/0.9330 | LR: 4.68e-04
        Epoch 89/90 | Train: 0.1927/0.9941 | Val: 0.2075/0.9330 | LR: 4.63e-04
        Epoch 90/90 | Train: 0.1931/0.9918 | Val: 0.2069/0.9330 | LR: 4.58e-04
    
      Best: Epoch 77, Val F1: 0.9330
      EVA-02 fold 0: val F1 = 0.9330
    
    ============================================================
    FOLD 1 — eva02 (eva02_base_patch14_224.mim_in22k)
    Epochs: 90 | Freeze: 10 | Warmup: 5 | CutMix: True
    ============================================================
    
      Total:    85.8M params
      Head:     4.6K params (lr=5e-04)
      Backbone: 85.8M params (lr=2e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (10 epochs, no CutMix)
        Epoch 1/10 | Train: 2.1892/0.5783 | Val: 1.3624/0.6602
        Epoch 2/10 | Train: 1.2454/0.7489 | Val: 1.2970/0.7226
        Epoch 3/10 | Train: 1.0482/0.7702 | Val: 1.2188/0.7500
        Epoch 4/10 | Train: 0.9431/0.7953 | Val: 1.1387/0.7564
        Epoch 5/10 | Train: 0.8600/0.8108 | Val: 1.0604/0.7564
        Epoch 6/10 | Train: 0.8194/0.8095 | Val: 0.9856/0.7736
        Epoch 7/10 | Train: 0.7779/0.8130 | Val: 0.9152/0.7755
        Epoch 8/10 | Train: 0.7948/0.8231 | Val: 0.8506/0.7751
        Epoch 9/10 | Train: 0.7451/0.8341 | Val: 0.7919/0.7780
        Epoch 10/10 | Train: 0.6905/0.8424 | Val: 0.7388/0.7823
    
      Stage 2: Full fine-tuning (80 epochs, CutMix=True)
        Epoch 11/90 | Train: 0.6994/0.8296 | Val: 0.3429/0.8179 | LR: 1.40e-04 ★
        Epoch 12/90 | Train: 0.6360/0.8561 | Val: 0.3410/0.8179 | LR: 2.30e-04
        Epoch 13/90 | Train: 0.4981/0.8891 | Val: 0.3405/0.8179 | LR: 3.20e-04
        Epoch 14/90 | Train: 0.4291/0.9088 | Val: 0.3399/0.8248 | LR: 4.10e-04 ★
        Epoch 15/90 | Train: 0.4090/0.9157 | Val: 0.3383/0.8316 | LR: 5.00e-04 ★
        Epoch 16/90 | Train: 0.3915/0.9202 | Val: 0.3354/0.8316 | LR: 4.97e-04
        Epoch 17/90 | Train: 0.3154/0.9518 | Val: 0.3327/0.8382 | LR: 4.88e-04 ★
        Epoch 18/90 | Train: 0.3094/0.9482 | Val: 0.3302/0.8382 | LR: 4.73e-04
        Epoch 19/90 | Train: 0.2896/0.9566 | Val: 0.3272/0.8382 | LR: 4.52e-04
        Epoch 20/90 | Train: 0.2787/0.9634 | Val: 0.3228/0.8382 | LR: 4.27e-04
        Epoch 21/90 | Train: 0.2580/0.9759 | Val: 0.3204/0.8436 | LR: 3.97e-04 ★
        Epoch 22/90 | Train: 0.2457/0.9755 | Val: 0.3182/0.8407 | LR: 3.64e-04
        Epoch 23/90 | Train: 0.2848/0.9712 | Val: 0.3155/0.8490 | LR: 3.27e-04 ★
        Epoch 24/90 | Train: 0.2387/0.9779 | Val: 0.3145/0.8449 | LR: 2.89e-04
        Epoch 25/90 | Train: 0.2362/0.9791 | Val: 0.3139/0.8407 | LR: 2.50e-04
        Epoch 26/90 | Train: 0.2438/0.9790 | Val: 0.3129/0.8474 | LR: 2.11e-04
        Epoch 27/90 | Train: 0.2476/0.9849 | Val: 0.3111/0.8474 | LR: 1.73e-04
        Epoch 28/90 | Train: 0.2177/0.9872 | Val: 0.3092/0.8499 | LR: 1.37e-04 ★
        Epoch 29/90 | Train: 0.2297/0.9863 | Val: 0.3076/0.8499 | LR: 1.03e-04
        Epoch 30/90 | Train: 0.2216/0.9826 | Val: 0.3066/0.8591 | LR: 7.33e-05 ★
        Epoch 31/90 | Train: 0.2129/0.9920 | Val: 0.3055/0.8591 | LR: 4.78e-05
        Epoch 32/90 | Train: 0.2095/0.9885 | Val: 0.3048/0.8591 | LR: 2.73e-05
        Epoch 33/90 | Train: 0.2005/0.9864 | Val: 0.3039/0.8615 | LR: 1.23e-05 ★
        Epoch 34/90 | Train: 0.2023/0.9901 | Val: 0.3028/0.8593 | LR: 3.18e-06
        Epoch 35/90 | Train: 0.2049/0.9922 | Val: 0.3019/0.8593 | LR: 5.00e-04
        Epoch 36/90 | Train: 0.2083/0.9889 | Val: 0.3012/0.8593 | LR: 4.99e-04
        Epoch 37/90 | Train: 0.2441/0.9787 | Val: 0.2986/0.8661 | LR: 4.97e-04 ★
        Epoch 38/90 | Train: 0.2537/0.9794 | Val: 0.2957/0.8707 | LR: 4.93e-04 ★
        Epoch 39/90 | Train: 0.2243/0.9876 | Val: 0.2940/0.8707 | LR: 4.88e-04
        Epoch 40/90 | Train: 0.2458/0.9835 | Val: 0.2922/0.8770 | LR: 4.81e-04 ★
        Epoch 41/90 | Train: 0.2475/0.9800 | Val: 0.2912/0.8770 | LR: 4.73e-04
        Epoch 42/90 | Train: 0.2311/0.9891 | Val: 0.2899/0.8770 | LR: 4.63e-04
        Epoch 43/90 | Train: 0.2226/0.9883 | Val: 0.2868/0.8770 | LR: 4.52e-04
        Epoch 44/90 | Train: 0.2308/0.9882 | Val: 0.2841/0.8770 | LR: 4.40e-04
        Epoch 45/90 | Train: 0.2392/0.9815 | Val: 0.2827/0.8770 | LR: 4.27e-04
        Epoch 46/90 | Train: 0.2156/0.9859 | Val: 0.2819/0.8770 | LR: 4.12e-04
        Epoch 47/90 | Train: 0.2083/0.9914 | Val: 0.2806/0.8770 | LR: 3.97e-04
        Epoch 48/90 | Train: 0.2015/0.9950 | Val: 0.2799/0.8795 | LR: 3.81e-04 ★
        Epoch 49/90 | Train: 0.2027/0.9921 | Val: 0.2796/0.8795 | LR: 3.64e-04
        Epoch 50/90 | Train: 0.2037/0.9904 | Val: 0.2797/0.8795 | LR: 3.46e-04
        Epoch 51/90 | Train: 0.2006/0.9907 | Val: 0.2785/0.8836 | LR: 3.27e-04 ★
        Epoch 52/90 | Train: 0.2158/0.9902 | Val: 0.2779/0.8836 | LR: 3.08e-04
        Epoch 53/90 | Train: 0.2082/0.9914 | Val: 0.2780/0.8836 | LR: 2.89e-04
        Epoch 54/90 | Train: 0.2069/0.9908 | Val: 0.2774/0.8836 | LR: 2.70e-04
        Epoch 55/90 | Train: 0.2309/0.9901 | Val: 0.2766/0.8836 | LR: 2.50e-04
        Epoch 56/90 | Train: 0.2012/0.9960 | Val: 0.2762/0.8836 | LR: 2.30e-04
        Epoch 57/90 | Train: 0.1854/0.9972 | Val: 0.2754/0.8861 | LR: 2.11e-04 ★
        Epoch 58/90 | Train: 0.1876/0.9943 | Val: 0.2744/0.8901 | LR: 1.92e-04 ★
        Epoch 59/90 | Train: 0.1883/0.9955 | Val: 0.2744/0.8940 | LR: 1.73e-04 ★
        Epoch 60/90 | Train: 0.1984/0.9929 | Val: 0.2746/0.8940 | LR: 1.54e-04
        Epoch 61/90 | Train: 0.1907/0.9952 | Val: 0.2745/0.8911 | LR: 1.37e-04
        Epoch 62/90 | Train: 0.1865/0.9967 | Val: 0.2746/0.8911 | LR: 1.19e-04
        Epoch 63/90 | Train: 0.1863/0.9953 | Val: 0.2743/0.8911 | LR: 1.03e-04
        Epoch 64/90 | Train: 0.1877/0.9947 | Val: 0.2740/0.8911 | LR: 8.77e-05
        Epoch 65/90 | Train: 0.1828/0.9984 | Val: 0.2736/0.8921 | LR: 7.33e-05
        Epoch 66/90 | Train: 0.1842/0.9955 | Val: 0.2732/0.8921 | LR: 6.00e-05
        Epoch 67/90 | Train: 0.1825/0.9970 | Val: 0.2727/0.8921 | LR: 4.78e-05
        Epoch 68/90 | Train: 0.1820/0.9970 | Val: 0.2721/0.8921 | LR: 3.69e-05
        Epoch 69/90 | Train: 0.1846/0.9962 | Val: 0.2715/0.8921 | LR: 2.73e-05
        Epoch 70/90 | Train: 0.1802/0.9985 | Val: 0.2708/0.8921 | LR: 1.91e-05
        Epoch 71/90 | Train: 0.1805/0.9960 | Val: 0.2699/0.8960 | LR: 1.23e-05 ★
        Epoch 72/90 | Train: 0.1801/0.9981 | Val: 0.2693/0.8960 | LR: 7.01e-06
        Epoch 73/90 | Train: 0.1804/0.9954 | Val: 0.2687/0.9025 | LR: 3.18e-06 ★
        Epoch 74/90 | Train: 0.1784/0.9967 | Val: 0.2681/0.9025 | LR: 8.71e-07
        Epoch 75/90 | Train: 0.1822/0.9957 | Val: 0.2675/0.9025 | LR: 5.00e-04
        Epoch 76/90 | Train: 0.1913/0.9938 | Val: 0.2675/0.9025 | LR: 5.00e-04
        Epoch 77/90 | Train: 0.2180/0.9926 | Val: 0.2677/0.9025 | LR: 4.99e-04
        Epoch 78/90 | Train: 0.2036/0.9895 | Val: 0.2678/0.9062 | LR: 4.98e-04 ★
        Epoch 79/90 | Train: 0.1966/0.9948 | Val: 0.2677/0.9062 | LR: 4.97e-04
        Epoch 80/90 | Train: 0.2019/0.9920 | Val: 0.2678/0.9028 | LR: 4.95e-04
        Epoch 81/90 | Train: 0.2000/0.9903 | Val: 0.2670/0.9028 | LR: 4.93e-04
        Epoch 82/90 | Train: 0.1933/0.9919 | Val: 0.2663/0.9028 | LR: 4.91e-04
        Epoch 83/90 | Train: 0.1939/0.9954 | Val: 0.2654/0.9062 | LR: 4.88e-04
        Epoch 84/90 | Train: 0.1924/0.9975 | Val: 0.2634/0.9088 | LR: 4.85e-04 ★
        Epoch 85/90 | Train: 0.1985/0.9901 | Val: 0.2608/0.9088 | LR: 4.81e-04
        Epoch 86/90 | Train: 0.1905/0.9988 | Val: 0.2597/0.9186 | LR: 4.77e-04 ★
        Epoch 87/90 | Train: 0.1946/0.9941 | Val: 0.2586/0.9160 | LR: 4.73e-04
        Epoch 88/90 | Train: 0.1880/0.9968 | Val: 0.2580/0.9160 | LR: 4.68e-04
        Epoch 89/90 | Train: 0.1804/0.9955 | Val: 0.2581/0.9160 | LR: 4.63e-04
        Epoch 90/90 | Train: 0.1841/0.9950 | Val: 0.2587/0.9160 | LR: 4.58e-04
    
      Best: Epoch 86, Val F1: 0.9186
      EVA-02 fold 1: val F1 = 0.9186
    
    ============================================================
    FOLD 2 — eva02 (eva02_base_patch14_224.mim_in22k)
    Epochs: 90 | Freeze: 10 | Warmup: 5 | CutMix: True
    ============================================================
    
      Total:    85.8M params
      Head:     4.6K params (lr=5e-04)
      Backbone: 85.8M params (lr=2e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (10 epochs, no CutMix)
        Epoch 1/10 | Train: 2.1771/0.5435 | Val: 1.3685/0.6721
        Epoch 2/10 | Train: 1.2636/0.7044 | Val: 1.3085/0.7582
        Epoch 3/10 | Train: 1.0668/0.7450 | Val: 1.2370/0.7811
        Epoch 4/10 | Train: 0.9997/0.7666 | Val: 1.1629/0.7783
        Epoch 5/10 | Train: 0.8916/0.7940 | Val: 1.0891/0.7920
        Epoch 6/10 | Train: 0.8500/0.8053 | Val: 1.0173/0.7967
        Epoch 7/10 | Train: 0.7876/0.8159 | Val: 0.9493/0.8034
        Epoch 8/10 | Train: 0.8029/0.8005 | Val: 0.8862/0.8117
        Epoch 9/10 | Train: 0.7246/0.8251 | Val: 0.8282/0.8097
        Epoch 10/10 | Train: 0.7447/0.8319 | Val: 0.7748/0.8123
    
      Stage 2: Full fine-tuning (80 epochs, CutMix=True)
        Epoch 11/90 | Train: 0.7046/0.8236 | Val: 0.2662/0.8549 | LR: 1.40e-04 ★
        Epoch 12/90 | Train: 0.5784/0.8737 | Val: 0.2646/0.8579 | LR: 2.30e-04 ★
        Epoch 13/90 | Train: 0.5160/0.8872 | Val: 0.2626/0.8617 | LR: 3.20e-04 ★
        Epoch 14/90 | Train: 0.4759/0.8903 | Val: 0.2595/0.8617 | LR: 4.10e-04
        Epoch 15/90 | Train: 0.4277/0.9058 | Val: 0.2548/0.8617 | LR: 5.00e-04
        Epoch 16/90 | Train: 0.3604/0.9352 | Val: 0.2487/0.8710 | LR: 4.97e-04 ★
        Epoch 17/90 | Train: 0.3152/0.9452 | Val: 0.2421/0.8685 | LR: 4.88e-04
        Epoch 18/90 | Train: 0.3462/0.9433 | Val: 0.2357/0.8685 | LR: 4.73e-04
        Epoch 19/90 | Train: 0.3090/0.9554 | Val: 0.2293/0.8729 | LR: 4.52e-04 ★
        Epoch 20/90 | Train: 0.2800/0.9650 | Val: 0.2215/0.8765 | LR: 4.27e-04 ★
        Epoch 21/90 | Train: 0.2833/0.9641 | Val: 0.2138/0.8870 | LR: 3.97e-04 ★
        Epoch 22/90 | Train: 0.2811/0.9689 | Val: 0.2065/0.8926 | LR: 3.64e-04 ★
        Epoch 23/90 | Train: 0.2858/0.9683 | Val: 0.1993/0.8901 | LR: 3.27e-04
        Epoch 24/90 | Train: 0.2404/0.9729 | Val: 0.1934/0.9024 | LR: 2.89e-04 ★
        Epoch 25/90 | Train: 0.2584/0.9726 | Val: 0.1886/0.9055 | LR: 2.50e-04 ★
        Epoch 26/90 | Train: 0.2408/0.9744 | Val: 0.1836/0.9139 | LR: 2.11e-04 ★
        Epoch 27/90 | Train: 0.2424/0.9847 | Val: 0.1798/0.9174 | LR: 1.73e-04 ★
        Epoch 28/90 | Train: 0.2261/0.9861 | Val: 0.1770/0.9174 | LR: 1.37e-04
        Epoch 29/90 | Train: 0.2036/0.9916 | Val: 0.1740/0.9174 | LR: 1.03e-04
        Epoch 30/90 | Train: 0.2215/0.9873 | Val: 0.1715/0.9174 | LR: 7.33e-05
        Epoch 31/90 | Train: 0.2212/0.9875 | Val: 0.1689/0.9231 | LR: 4.78e-05 ★
        Epoch 32/90 | Train: 0.2048/0.9918 | Val: 0.1663/0.9231 | LR: 2.73e-05
        Epoch 33/90 | Train: 0.2098/0.9901 | Val: 0.1639/0.9231 | LR: 1.23e-05
        Epoch 34/90 | Train: 0.2044/0.9896 | Val: 0.1620/0.9256 | LR: 3.18e-06 ★
        Epoch 35/90 | Train: 0.2084/0.9885 | Val: 0.1604/0.9256 | LR: 5.00e-04
        Epoch 36/90 | Train: 0.2458/0.9809 | Val: 0.1586/0.9329 | LR: 4.99e-04 ★
        Epoch 37/90 | Train: 0.2546/0.9823 | Val: 0.1575/0.9329 | LR: 4.97e-04
        Epoch 38/90 | Train: 0.2268/0.9860 | Val: 0.1559/0.9329 | LR: 4.93e-04
        Epoch 39/90 | Train: 0.2427/0.9819 | Val: 0.1535/0.9304 | LR: 4.88e-04
        Epoch 40/90 | Train: 0.2277/0.9838 | Val: 0.1529/0.9304 | LR: 4.81e-04
        Epoch 41/90 | Train: 0.2319/0.9828 | Val: 0.1519/0.9329 | LR: 4.73e-04
        Epoch 42/90 | Train: 0.2332/0.9824 | Val: 0.1513/0.9329 | LR: 4.63e-04
        Epoch 43/90 | Train: 0.2244/0.9861 | Val: 0.1498/0.9355 | LR: 4.52e-04 ★
        Epoch 44/90 | Train: 0.2102/0.9939 | Val: 0.1494/0.9355 | LR: 4.40e-04
        Epoch 45/90 | Train: 0.2267/0.9849 | Val: 0.1481/0.9390 | LR: 4.27e-04 ★
        Epoch 46/90 | Train: 0.2131/0.9901 | Val: 0.1470/0.9390 | LR: 4.12e-04
        Epoch 47/90 | Train: 0.2037/0.9895 | Val: 0.1462/0.9390 | LR: 3.97e-04
        Epoch 48/90 | Train: 0.2116/0.9909 | Val: 0.1456/0.9390 | LR: 3.81e-04
        Epoch 49/90 | Train: 0.1998/0.9918 | Val: 0.1450/0.9301 | LR: 3.64e-04
        Epoch 50/90 | Train: 0.1920/0.9963 | Val: 0.1447/0.9290 | LR: 3.46e-04
        Epoch 51/90 | Train: 0.2027/0.9959 | Val: 0.1446/0.9290 | LR: 3.27e-04
        Epoch 52/90 | Train: 0.1950/0.9933 | Val: 0.1444/0.9200 | LR: 3.08e-04
        Epoch 53/90 | Train: 0.1980/0.9952 | Val: 0.1449/0.9200 | LR: 2.89e-04
        Epoch 54/90 | Train: 0.1980/0.9933 | Val: 0.1445/0.9200 | LR: 2.70e-04
        Epoch 55/90 | Train: 0.1937/0.9889 | Val: 0.1447/0.9200 | LR: 2.50e-04
        Epoch 56/90 | Train: 0.1886/0.9941 | Val: 0.1446/0.9200 | LR: 2.30e-04
        Epoch 57/90 | Train: 0.1962/0.9948 | Val: 0.1440/0.9200 | LR: 2.11e-04
        Epoch 58/90 | Train: 0.1941/0.9937 | Val: 0.1432/0.9200 | LR: 1.92e-04
        Epoch 59/90 | Train: 0.1848/0.9953 | Val: 0.1427/0.9273 | LR: 1.73e-04
        Epoch 60/90 | Train: 0.1886/0.9944 | Val: 0.1430/0.9273 | LR: 1.54e-04
        Early stopping at epoch 60 (patience=15)
    
      Best: Epoch 45, Val F1: 0.9390
      EVA-02 fold 2: val F1 = 0.9390
    
    ============================================================
    FOLD 3 — eva02 (eva02_base_patch14_224.mim_in22k)
    Epochs: 90 | Freeze: 10 | Warmup: 5 | CutMix: True
    ============================================================
    
      Total:    85.8M params
      Head:     4.6K params (lr=5e-04)
      Backbone: 85.8M params (lr=2e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (10 epochs, no CutMix)
        Epoch 1/10 | Train: 2.1277/0.6084 | Val: 1.3708/0.6883
        Epoch 2/10 | Train: 1.2317/0.7360 | Val: 1.3077/0.7482
        Epoch 3/10 | Train: 1.0562/0.7738 | Val: 1.2337/0.7577
        Epoch 4/10 | Train: 0.9560/0.7945 | Val: 1.1580/0.7745
        Epoch 5/10 | Train: 0.8739/0.7796 | Val: 1.0830/0.7998
        Epoch 6/10 | Train: 0.8402/0.7944 | Val: 1.0104/0.8029
        Epoch 7/10 | Train: 0.7631/0.8174 | Val: 0.9407/0.8193
        Epoch 8/10 | Train: 0.7255/0.8243 | Val: 0.8757/0.8193
        Epoch 9/10 | Train: 0.7404/0.8236 | Val: 0.8173/0.8193
        Epoch 10/10 | Train: 0.7307/0.8336 | Val: 0.7636/0.8193
    
      Stage 2: Full fine-tuning (80 epochs, CutMix=True)
        Epoch 11/90 | Train: 0.6330/0.8537 | Val: 0.3668/0.8323 | LR: 1.40e-04 ★
        Epoch 12/90 | Train: 0.5779/0.8715 | Val: 0.3663/0.8323 | LR: 2.30e-04
        Epoch 13/90 | Train: 0.5276/0.8888 | Val: 0.3648/0.8323 | LR: 3.20e-04
        Epoch 14/90 | Train: 0.4513/0.9168 | Val: 0.3621/0.8323 | LR: 4.10e-04
        Epoch 15/90 | Train: 0.4054/0.9124 | Val: 0.3581/0.8350 | LR: 5.00e-04 ★
        Epoch 16/90 | Train: 0.3933/0.9225 | Val: 0.3539/0.8383 | LR: 4.97e-04 ★
        Epoch 17/90 | Train: 0.3446/0.9448 | Val: 0.3498/0.8383 | LR: 4.88e-04
        Epoch 18/90 | Train: 0.3589/0.9466 | Val: 0.3462/0.8458 | LR: 4.73e-04 ★
        Epoch 19/90 | Train: 0.3108/0.9491 | Val: 0.3424/0.8458 | LR: 4.52e-04
        Epoch 20/90 | Train: 0.3035/0.9460 | Val: 0.3383/0.8458 | LR: 4.27e-04
        Epoch 21/90 | Train: 0.3029/0.9652 | Val: 0.3345/0.8464 | LR: 3.97e-04 ★
        Epoch 22/90 | Train: 0.2450/0.9803 | Val: 0.3311/0.8490 | LR: 3.64e-04 ★
        Epoch 23/90 | Train: 0.2495/0.9781 | Val: 0.3274/0.8538 | LR: 3.27e-04 ★
        Epoch 24/90 | Train: 0.2485/0.9799 | Val: 0.3244/0.8538 | LR: 2.89e-04
        Epoch 25/90 | Train: 0.2427/0.9778 | Val: 0.3215/0.8572 | LR: 2.50e-04 ★
        Epoch 26/90 | Train: 0.2504/0.9818 | Val: 0.3187/0.8682 | LR: 2.11e-04 ★
        Epoch 27/90 | Train: 0.2248/0.9859 | Val: 0.3158/0.8682 | LR: 1.73e-04
        Epoch 28/90 | Train: 0.2219/0.9890 | Val: 0.3129/0.8682 | LR: 1.37e-04
        Epoch 29/90 | Train: 0.2213/0.9860 | Val: 0.3101/0.8682 | LR: 1.03e-04
        Epoch 30/90 | Train: 0.2028/0.9907 | Val: 0.3075/0.8682 | LR: 7.33e-05
        Epoch 31/90 | Train: 0.2037/0.9933 | Val: 0.3055/0.8682 | LR: 4.78e-05
        Epoch 32/90 | Train: 0.2054/0.9876 | Val: 0.3033/0.8706 | LR: 2.73e-05 ★
        Epoch 33/90 | Train: 0.2021/0.9972 | Val: 0.3014/0.8706 | LR: 1.23e-05
        Epoch 34/90 | Train: 0.2020/0.9915 | Val: 0.2997/0.8706 | LR: 3.18e-06
        Epoch 35/90 | Train: 0.2047/0.9927 | Val: 0.2977/0.8706 | LR: 5.00e-04
        Epoch 36/90 | Train: 0.2297/0.9826 | Val: 0.2965/0.8737 | LR: 4.99e-04 ★
        Epoch 37/90 | Train: 0.2405/0.9758 | Val: 0.2950/0.8737 | LR: 4.97e-04
        Epoch 38/90 | Train: 0.2265/0.9865 | Val: 0.2936/0.8737 | LR: 4.93e-04
        Epoch 39/90 | Train: 0.2277/0.9819 | Val: 0.2914/0.8772 | LR: 4.88e-04 ★
        Epoch 40/90 | Train: 0.2302/0.9816 | Val: 0.2905/0.8772 | LR: 4.81e-04
        Epoch 41/90 | Train: 0.2249/0.9877 | Val: 0.2893/0.8772 | LR: 4.73e-04
        Epoch 42/90 | Train: 0.2210/0.9872 | Val: 0.2882/0.8772 | LR: 4.63e-04
        Epoch 43/90 | Train: 0.2047/0.9904 | Val: 0.2869/0.8753 | LR: 4.52e-04
        Epoch 44/90 | Train: 0.2129/0.9875 | Val: 0.2847/0.8781 | LR: 4.40e-04 ★
        Epoch 45/90 | Train: 0.2236/0.9891 | Val: 0.2826/0.8781 | LR: 4.27e-04
        Epoch 46/90 | Train: 0.2242/0.9860 | Val: 0.2809/0.8789 | LR: 4.12e-04 ★
        Epoch 47/90 | Train: 0.2140/0.9919 | Val: 0.2789/0.8789 | LR: 3.97e-04
        Epoch 48/90 | Train: 0.2030/0.9911 | Val: 0.2766/0.8789 | LR: 3.81e-04
        Epoch 49/90 | Train: 0.2115/0.9886 | Val: 0.2759/0.8804 | LR: 3.64e-04 ★
        Epoch 50/90 | Train: 0.2061/0.9906 | Val: 0.2745/0.8804 | LR: 3.46e-04
        Epoch 51/90 | Train: 0.2236/0.9851 | Val: 0.2731/0.8804 | LR: 3.27e-04
        Epoch 52/90 | Train: 0.1966/0.9959 | Val: 0.2715/0.8799 | LR: 3.08e-04
        Epoch 53/90 | Train: 0.2068/0.9899 | Val: 0.2701/0.8799 | LR: 2.89e-04
        Epoch 54/90 | Train: 0.2011/0.9936 | Val: 0.2685/0.8838 | LR: 2.70e-04 ★
        Epoch 55/90 | Train: 0.2010/0.9903 | Val: 0.2663/0.8838 | LR: 2.50e-04
        Epoch 56/90 | Train: 0.1984/0.9967 | Val: 0.2643/0.8838 | LR: 2.30e-04
        Epoch 57/90 | Train: 0.1918/0.9923 | Val: 0.2619/0.8877 | LR: 2.11e-04 ★
        Epoch 58/90 | Train: 0.1886/0.9950 | Val: 0.2602/0.8877 | LR: 1.92e-04
        Epoch 59/90 | Train: 0.1904/0.9967 | Val: 0.2592/0.8877 | LR: 1.73e-04
        Epoch 60/90 | Train: 0.1862/0.9936 | Val: 0.2580/0.8877 | LR: 1.54e-04
        Epoch 61/90 | Train: 0.1886/0.9958 | Val: 0.2569/0.8877 | LR: 1.37e-04
        Epoch 62/90 | Train: 0.1803/0.9970 | Val: 0.2558/0.8877 | LR: 1.19e-04
        Epoch 63/90 | Train: 0.1868/0.9945 | Val: 0.2552/0.8850 | LR: 1.03e-04
        Epoch 64/90 | Train: 0.1861/0.9936 | Val: 0.2544/0.8850 | LR: 8.77e-05
        Epoch 65/90 | Train: 0.1979/0.9973 | Val: 0.2537/0.8850 | LR: 7.33e-05
        Epoch 66/90 | Train: 0.1810/0.9974 | Val: 0.2527/0.8850 | LR: 6.00e-05
        Epoch 67/90 | Train: 0.1753/0.9982 | Val: 0.2520/0.8912 | LR: 4.78e-05 ★
        Epoch 68/90 | Train: 0.1796/0.9968 | Val: 0.2511/0.8879 | LR: 3.69e-05
        Epoch 69/90 | Train: 0.1801/0.9984 | Val: 0.2503/0.8879 | LR: 2.73e-05
        Epoch 70/90 | Train: 0.1814/0.9951 | Val: 0.2497/0.8879 | LR: 1.91e-05
        Epoch 71/90 | Train: 0.1868/0.9944 | Val: 0.2487/0.8879 | LR: 1.23e-05
        Epoch 72/90 | Train: 0.1835/0.9967 | Val: 0.2479/0.8852 | LR: 7.01e-06
        Epoch 73/90 | Train: 0.1764/0.9985 | Val: 0.2473/0.8852 | LR: 3.18e-06
        Epoch 74/90 | Train: 0.1852/0.9961 | Val: 0.2465/0.8932 | LR: 8.71e-07 ★
        Epoch 75/90 | Train: 0.1771/0.9978 | Val: 0.2458/0.8932 | LR: 5.00e-04
        Epoch 76/90 | Train: 0.1880/0.9954 | Val: 0.2454/0.8932 | LR: 5.00e-04
        Epoch 77/90 | Train: 0.2027/0.9910 | Val: 0.2456/0.8932 | LR: 4.99e-04
        Epoch 78/90 | Train: 0.1990/0.9933 | Val: 0.2456/0.8932 | LR: 4.98e-04
        Epoch 79/90 | Train: 0.2098/0.9923 | Val: 0.2451/0.8932 | LR: 4.97e-04
        Epoch 80/90 | Train: 0.2044/0.9926 | Val: 0.2442/0.8932 | LR: 4.95e-04
        Epoch 81/90 | Train: 0.2079/0.9922 | Val: 0.2437/0.8932 | LR: 4.93e-04
        Epoch 82/90 | Train: 0.1969/0.9934 | Val: 0.2425/0.8959 | LR: 4.91e-04 ★
        Epoch 83/90 | Train: 0.2082/0.9920 | Val: 0.2416/0.8959 | LR: 4.88e-04
        Epoch 84/90 | Train: 0.1945/0.9946 | Val: 0.2410/0.8996 | LR: 4.85e-04 ★
        Epoch 85/90 | Train: 0.1919/0.9922 | Val: 0.2402/0.8996 | LR: 4.81e-04
        Epoch 86/90 | Train: 0.2179/0.9902 | Val: 0.2393/0.9023 | LR: 4.77e-04 ★
        Epoch 87/90 | Train: 0.1934/0.9957 | Val: 0.2379/0.9023 | LR: 4.73e-04
        Epoch 88/90 | Train: 0.1880/0.9939 | Val: 0.2372/0.9023 | LR: 4.68e-04
        Epoch 89/90 | Train: 0.1913/0.9925 | Val: 0.2361/0.9023 | LR: 4.63e-04
        Epoch 90/90 | Train: 0.2150/0.9896 | Val: 0.2351/0.9023 | LR: 4.58e-04
    
      Best: Epoch 86, Val F1: 0.9023
      EVA-02 fold 3: val F1 = 0.9023
    
    ============================================================
    FOLD 4 — eva02 (eva02_base_patch14_224.mim_in22k)
    Epochs: 90 | Freeze: 10 | Warmup: 5 | CutMix: True
    ============================================================
    
      Total:    85.8M params
      Head:     4.6K params (lr=5e-04)
      Backbone: 85.8M params (lr=2e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (10 epochs, no CutMix)
        Epoch 1/10 | Train: 2.1796/0.5570 | Val: 1.3667/0.7050
        Epoch 2/10 | Train: 1.2500/0.7253 | Val: 1.3008/0.7586
        Epoch 3/10 | Train: 1.0543/0.7623 | Val: 1.2218/0.8077
        Epoch 4/10 | Train: 0.9388/0.7801 | Val: 1.1403/0.8095
        Epoch 5/10 | Train: 0.9122/0.7859 | Val: 1.0589/0.8221
        Epoch 6/10 | Train: 0.8342/0.7973 | Val: 0.9806/0.8221
        Epoch 7/10 | Train: 0.8207/0.8151 | Val: 0.9074/0.8334
        Epoch 8/10 | Train: 0.7828/0.8114 | Val: 0.8387/0.8399
        Epoch 9/10 | Train: 0.7304/0.8243 | Val: 0.7753/0.8437
        Epoch 10/10 | Train: 0.7505/0.8341 | Val: 0.7181/0.8531
    
      Stage 2: Full fine-tuning (80 epochs, CutMix=True)
        Epoch 11/90 | Train: 0.7107/0.8435 | Val: 0.2706/0.8630 | LR: 1.40e-04 ★
        Epoch 12/90 | Train: 0.6350/0.8526 | Val: 0.2696/0.8661 | LR: 2.30e-04 ★
        Epoch 13/90 | Train: 0.5458/0.8748 | Val: 0.2678/0.8661 | LR: 3.20e-04
        Epoch 14/90 | Train: 0.4881/0.8918 | Val: 0.2664/0.8648 | LR: 4.10e-04
        Epoch 15/90 | Train: 0.4343/0.9044 | Val: 0.2630/0.8648 | LR: 5.00e-04
        Epoch 16/90 | Train: 0.3716/0.9177 | Val: 0.2588/0.8648 | LR: 4.97e-04
        Epoch 17/90 | Train: 0.3443/0.9373 | Val: 0.2531/0.8711 | LR: 4.88e-04 ★
        Epoch 18/90 | Train: 0.3422/0.9517 | Val: 0.2481/0.8737 | LR: 4.73e-04 ★
        Epoch 19/90 | Train: 0.2892/0.9605 | Val: 0.2415/0.8737 | LR: 4.52e-04
        Epoch 20/90 | Train: 0.2976/0.9599 | Val: 0.2355/0.8737 | LR: 4.27e-04
        Epoch 21/90 | Train: 0.2935/0.9657 | Val: 0.2290/0.8737 | LR: 3.97e-04
        Epoch 22/90 | Train: 0.2618/0.9689 | Val: 0.2229/0.8762 | LR: 3.64e-04 ★
        Epoch 23/90 | Train: 0.2463/0.9678 | Val: 0.2172/0.8762 | LR: 3.27e-04
        Epoch 24/90 | Train: 0.2428/0.9811 | Val: 0.2116/0.8799 | LR: 2.89e-04 ★
        Epoch 25/90 | Train: 0.2292/0.9772 | Val: 0.2059/0.8775 | LR: 2.50e-04
        Epoch 26/90 | Train: 0.2422/0.9844 | Val: 0.2001/0.8790 | LR: 2.11e-04
        Epoch 27/90 | Train: 0.2145/0.9856 | Val: 0.1945/0.8827 | LR: 1.73e-04 ★
        Epoch 28/90 | Train: 0.2176/0.9861 | Val: 0.1894/0.8877 | LR: 1.37e-04 ★
        Epoch 29/90 | Train: 0.2214/0.9887 | Val: 0.1844/0.8890 | LR: 1.03e-04 ★
        Epoch 30/90 | Train: 0.2222/0.9880 | Val: 0.1800/0.8958 | LR: 7.33e-05 ★
        Epoch 31/90 | Train: 0.2159/0.9886 | Val: 0.1756/0.8968 | LR: 4.78e-05 ★
        Epoch 32/90 | Train: 0.2105/0.9869 | Val: 0.1716/0.9038 | LR: 2.73e-05 ★
        Epoch 33/90 | Train: 0.2051/0.9913 | Val: 0.1674/0.9038 | LR: 1.23e-05
        Epoch 34/90 | Train: 0.2002/0.9942 | Val: 0.1636/0.9038 | LR: 3.18e-06
        Epoch 35/90 | Train: 0.2098/0.9886 | Val: 0.1600/0.9095 | LR: 5.00e-04 ★
        Epoch 36/90 | Train: 0.2463/0.9762 | Val: 0.1560/0.9129 | LR: 4.99e-04 ★
        Epoch 37/90 | Train: 0.2602/0.9788 | Val: 0.1526/0.9129 | LR: 4.97e-04
        Epoch 38/90 | Train: 0.2573/0.9811 | Val: 0.1489/0.9260 | LR: 4.93e-04 ★
        Epoch 39/90 | Train: 0.2148/0.9903 | Val: 0.1458/0.9285 | LR: 4.88e-04 ★
        Epoch 40/90 | Train: 0.2290/0.9867 | Val: 0.1429/0.9339 | LR: 4.81e-04 ★
        Epoch 41/90 | Train: 0.2341/0.9846 | Val: 0.1403/0.9339 | LR: 4.73e-04
        Epoch 42/90 | Train: 0.2180/0.9901 | Val: 0.1380/0.9339 | LR: 4.63e-04
        Epoch 43/90 | Train: 0.2060/0.9888 | Val: 0.1356/0.9435 | LR: 4.52e-04 ★
        Epoch 44/90 | Train: 0.2157/0.9900 | Val: 0.1329/0.9460 | LR: 4.40e-04 ★
        Epoch 45/90 | Train: 0.2210/0.9821 | Val: 0.1306/0.9512 | LR: 4.27e-04 ★
        Epoch 46/90 | Train: 0.2133/0.9867 | Val: 0.1285/0.9512 | LR: 4.12e-04
        Epoch 47/90 | Train: 0.2052/0.9917 | Val: 0.1263/0.9512 | LR: 3.97e-04
        Epoch 48/90 | Train: 0.2106/0.9878 | Val: 0.1247/0.9512 | LR: 3.81e-04
        Epoch 49/90 | Train: 0.2173/0.9892 | Val: 0.1228/0.9449 | LR: 3.64e-04
        Epoch 50/90 | Train: 0.2047/0.9908 | Val: 0.1208/0.9449 | LR: 3.46e-04
        Epoch 51/90 | Train: 0.2039/0.9912 | Val: 0.1192/0.9449 | LR: 3.27e-04
        Epoch 52/90 | Train: 0.2047/0.9904 | Val: 0.1177/0.9475 | LR: 3.08e-04
        Epoch 53/90 | Train: 0.1988/0.9951 | Val: 0.1159/0.9475 | LR: 2.89e-04
        Epoch 54/90 | Train: 0.1895/0.9961 | Val: 0.1144/0.9423 | LR: 2.70e-04
        Epoch 55/90 | Train: 0.1920/0.9937 | Val: 0.1133/0.9423 | LR: 2.50e-04
        Epoch 56/90 | Train: 0.1932/0.9934 | Val: 0.1119/0.9481 | LR: 2.30e-04
        Epoch 57/90 | Train: 0.1915/0.9959 | Val: 0.1103/0.9481 | LR: 2.11e-04
        Epoch 58/90 | Train: 0.1883/0.9924 | Val: 0.1088/0.9481 | LR: 1.92e-04
        Epoch 59/90 | Train: 0.1814/0.9973 | Val: 0.1074/0.9481 | LR: 1.73e-04
        Epoch 60/90 | Train: 0.1865/0.9987 | Val: 0.1060/0.9535 | LR: 1.54e-04 ★
        Epoch 61/90 | Train: 0.1824/0.9979 | Val: 0.1046/0.9535 | LR: 1.37e-04
        Epoch 62/90 | Train: 0.1863/0.9967 | Val: 0.1032/0.9475 | LR: 1.19e-04
        Epoch 63/90 | Train: 0.1876/0.9944 | Val: 0.1023/0.9475 | LR: 1.03e-04
        Epoch 64/90 | Train: 0.1765/0.9960 | Val: 0.1015/0.9500 | LR: 8.77e-05
        Epoch 65/90 | Train: 0.1831/0.9950 | Val: 0.1003/0.9500 | LR: 7.33e-05
        Epoch 66/90 | Train: 0.1814/0.9967 | Val: 0.0996/0.9500 | LR: 6.00e-05
        Epoch 67/90 | Train: 0.1805/0.9993 | Val: 0.0986/0.9500 | LR: 4.78e-05
        Epoch 68/90 | Train: 0.1784/0.9951 | Val: 0.0978/0.9500 | LR: 3.69e-05
        Epoch 69/90 | Train: 0.1803/0.9976 | Val: 0.0972/0.9500 | LR: 2.73e-05
        Epoch 70/90 | Train: 0.1804/0.9951 | Val: 0.0965/0.9500 | LR: 1.91e-05
        Epoch 71/90 | Train: 0.1742/0.9994 | Val: 0.0959/0.9500 | LR: 1.23e-05
        Epoch 72/90 | Train: 0.1760/0.9955 | Val: 0.0953/0.9526 | LR: 7.01e-06
        Epoch 73/90 | Train: 0.1815/0.9958 | Val: 0.0947/0.9526 | LR: 3.18e-06
        Epoch 74/90 | Train: 0.1843/0.9947 | Val: 0.0942/0.9526 | LR: 8.71e-07
        Epoch 75/90 | Train: 0.1833/0.9971 | Val: 0.0937/0.9526 | LR: 5.00e-04
        Early stopping at epoch 75 (patience=15)
    
      Best: Epoch 60, Val F1: 0.9535
      EVA-02 fold 4: val F1 = 0.9535
    
    EVA-02 summary: mean=0.9293 ± 0.0175
      exp06 baseline: mean=0.9267 ± 0.0239


## 8 · Train DINOv2-Base  (70 epochs)


```python
dinov2_results = {}

for fold in range(N_FOLDS):
    result = train_fold(
        fold         = fold,
        model_key    = 'dinov2',
        train_df     = train_df,
        class_weights= class_weights,
        use_cutmix   = True,
        use_sampler  = False,
        exp_id       = EXP_ID,
    )
    dinov2_results[fold] = result
    print(f'  DINOv2 fold {fold}: val F1 = {result["best_f1"]:.4f}')

scores = [r['best_f1'] for r in dinov2_results.values()]
print(f'\nDINOv2 summary: mean={np.mean(scores):.4f} ± {np.std(scores):.4f}')
print(f'  exp06 baseline: mean=0.9321 ± 0.0146')
```

    
    ============================================================
    FOLD 0 — dinov2 (vit_base_patch14_reg4_dinov2.lvd142m)
    Epochs: 90 | Freeze: 10 | Warmup: 5 | CutMix: True
    ============================================================
    
      Loaded linear probe head → dinov2_probe_fold0.pth
    
      Total:    85.7M params
      Head:     4.6K params (lr=1e-03)
      Backbone: 85.7M params (lr=1e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (10 epochs, no CutMix)
        Epoch 1/10 | Train: 0.9523/0.8205 | Val: 0.3549/0.9269
        Epoch 2/10 | Train: 0.7093/0.8359 | Val: 0.3451/0.9269
        Epoch 3/10 | Train: 0.6202/0.8619 | Val: 0.3342/0.9269
        Epoch 4/10 | Train: 0.5629/0.8732 | Val: 0.3229/0.9269
        Epoch 5/10 | Train: 0.4957/0.8955 | Val: 0.3121/0.9244
        Epoch 6/10 | Train: 0.5070/0.8932 | Val: 0.3021/0.9244
        Epoch 7/10 | Train: 0.5085/0.8809 | Val: 0.2931/0.9247
        Epoch 8/10 | Train: 0.4985/0.9097 | Val: 0.2849/0.9247
        Epoch 9/10 | Train: 0.5176/0.8917 | Val: 0.2781/0.9247
        Epoch 10/10 | Train: 0.4618/0.8961 | Val: 0.2714/0.9334
    
      Stage 2: Full fine-tuning (80 epochs, CutMix=True)
        Epoch 11/90 | Train: 0.4519/0.9070 | Val: 0.2860/0.8891 | LR: 2.80e-04 ★
        Epoch 12/90 | Train: 0.4256/0.8991 | Val: 0.2855/0.8874 | LR: 4.60e-04
        Epoch 13/90 | Train: 0.3963/0.9101 | Val: 0.2857/0.8959 | LR: 6.40e-04 ★
        Epoch 14/90 | Train: 0.3427/0.9323 | Val: 0.2845/0.9000 | LR: 8.20e-04 ★
        Epoch 15/90 | Train: 0.3661/0.9314 | Val: 0.2836/0.9000 | LR: 1.00e-03
        Epoch 16/90 | Train: 0.3390/0.9411 | Val: 0.2819/0.9000 | LR: 9.94e-04
        Epoch 17/90 | Train: 0.3183/0.9556 | Val: 0.2803/0.9000 | LR: 9.76e-04
        Epoch 18/90 | Train: 0.3132/0.9572 | Val: 0.2785/0.9000 | LR: 9.46e-04
        Epoch 19/90 | Train: 0.2837/0.9672 | Val: 0.2763/0.9000 | LR: 9.05e-04
        Epoch 20/90 | Train: 0.2904/0.9622 | Val: 0.2746/0.9000 | LR: 8.54e-04
        Epoch 21/90 | Train: 0.2584/0.9697 | Val: 0.2727/0.9028 | LR: 7.94e-04 ★
        Epoch 22/90 | Train: 0.2398/0.9778 | Val: 0.2709/0.9028 | LR: 7.27e-04
        Epoch 23/90 | Train: 0.2606/0.9740 | Val: 0.2694/0.9053 | LR: 6.55e-04 ★
        Epoch 24/90 | Train: 0.2420/0.9795 | Val: 0.2679/0.9053 | LR: 5.78e-04
        Epoch 25/90 | Train: 0.2293/0.9857 | Val: 0.2663/0.9053 | LR: 5.00e-04
        Epoch 26/90 | Train: 0.2302/0.9866 | Val: 0.2649/0.9119 | LR: 4.22e-04 ★
        Epoch 27/90 | Train: 0.2297/0.9868 | Val: 0.2631/0.9119 | LR: 3.46e-04
        Epoch 28/90 | Train: 0.2148/0.9861 | Val: 0.2616/0.9145 | LR: 2.73e-04 ★
        Epoch 29/90 | Train: 0.2118/0.9850 | Val: 0.2600/0.9145 | LR: 2.06e-04
        Epoch 30/90 | Train: 0.2023/0.9890 | Val: 0.2586/0.9145 | LR: 1.47e-04
        Epoch 31/90 | Train: 0.2066/0.9934 | Val: 0.2573/0.9214 | LR: 9.56e-05 ★
        Epoch 32/90 | Train: 0.2151/0.9870 | Val: 0.2556/0.9250 | LR: 5.46e-05 ★
        Epoch 33/90 | Train: 0.2217/0.9860 | Val: 0.2543/0.9250 | LR: 2.46e-05
        Epoch 34/90 | Train: 0.2069/0.9926 | Val: 0.2531/0.9250 | LR: 6.26e-06
        Epoch 35/90 | Train: 0.2057/0.9918 | Val: 0.2520/0.9250 | LR: 1.00e-03
        Epoch 36/90 | Train: 0.2369/0.9850 | Val: 0.2514/0.9225 | LR: 9.98e-04
        Epoch 37/90 | Train: 0.2209/0.9829 | Val: 0.2503/0.9225 | LR: 9.94e-04
        Epoch 38/90 | Train: 0.2233/0.9855 | Val: 0.2485/0.9225 | LR: 9.86e-04
        Epoch 39/90 | Train: 0.2246/0.9849 | Val: 0.2467/0.9225 | LR: 9.76e-04
        Epoch 40/90 | Train: 0.2179/0.9878 | Val: 0.2447/0.9225 | LR: 9.62e-04
        Epoch 41/90 | Train: 0.2205/0.9856 | Val: 0.2424/0.9286 | LR: 9.46e-04 ★
        Epoch 42/90 | Train: 0.2283/0.9823 | Val: 0.2400/0.9286 | LR: 9.26e-04
        Epoch 43/90 | Train: 0.2321/0.9834 | Val: 0.2383/0.9286 | LR: 9.05e-04
        Epoch 44/90 | Train: 0.2272/0.9867 | Val: 0.2365/0.9286 | LR: 8.80e-04
        Epoch 45/90 | Train: 0.2144/0.9914 | Val: 0.2348/0.9391 | LR: 8.54e-04 ★
        Epoch 46/90 | Train: 0.2121/0.9927 | Val: 0.2337/0.9391 | LR: 8.25e-04
        Epoch 47/90 | Train: 0.2149/0.9916 | Val: 0.2321/0.9391 | LR: 7.94e-04
        Epoch 48/90 | Train: 0.2029/0.9957 | Val: 0.2307/0.9391 | LR: 7.61e-04
        Epoch 49/90 | Train: 0.2030/0.9924 | Val: 0.2299/0.9391 | LR: 7.27e-04
        Epoch 50/90 | Train: 0.2060/0.9922 | Val: 0.2288/0.9391 | LR: 6.91e-04
        Epoch 51/90 | Train: 0.1915/0.9957 | Val: 0.2273/0.9391 | LR: 6.55e-04
        Epoch 52/90 | Train: 0.1943/0.9961 | Val: 0.2260/0.9391 | LR: 6.17e-04
        Epoch 53/90 | Train: 0.2034/0.9932 | Val: 0.2247/0.9391 | LR: 5.78e-04
        Epoch 54/90 | Train: 0.1938/0.9955 | Val: 0.2235/0.9391 | LR: 5.39e-04
        Epoch 55/90 | Train: 0.1949/0.9927 | Val: 0.2226/0.9391 | LR: 5.00e-04
        Epoch 56/90 | Train: 0.1996/0.9933 | Val: 0.2217/0.9391 | LR: 4.61e-04
        Epoch 57/90 | Train: 0.1910/0.9973 | Val: 0.2207/0.9426 | LR: 4.22e-04 ★
        Epoch 58/90 | Train: 0.1952/0.9935 | Val: 0.2195/0.9426 | LR: 3.83e-04
        Epoch 59/90 | Train: 0.1944/0.9932 | Val: 0.2184/0.9426 | LR: 3.46e-04
        Epoch 60/90 | Train: 0.1907/0.9970 | Val: 0.2176/0.9426 | LR: 3.09e-04
        Epoch 61/90 | Train: 0.1847/0.9984 | Val: 0.2165/0.9426 | LR: 2.73e-04
        Epoch 62/90 | Train: 0.1910/0.9968 | Val: 0.2160/0.9426 | LR: 2.39e-04
        Epoch 63/90 | Train: 0.1975/0.9964 | Val: 0.2151/0.9426 | LR: 2.06e-04
        Epoch 64/90 | Train: 0.1875/0.9952 | Val: 0.2140/0.9451 | LR: 1.75e-04 ★
        Epoch 65/90 | Train: 0.1862/0.9960 | Val: 0.2130/0.9451 | LR: 1.47e-04
        Epoch 66/90 | Train: 0.1868/0.9948 | Val: 0.2121/0.9451 | LR: 1.20e-04
        Epoch 67/90 | Train: 0.1835/0.9980 | Val: 0.2111/0.9451 | LR: 9.56e-05
        Epoch 68/90 | Train: 0.1805/0.9976 | Val: 0.2103/0.9451 | LR: 7.38e-05
        Epoch 69/90 | Train: 0.1814/0.9993 | Val: 0.2096/0.9451 | LR: 5.46e-05
        Epoch 70/90 | Train: 0.1869/0.9966 | Val: 0.2089/0.9451 | LR: 3.82e-05
        Epoch 71/90 | Train: 0.1773/0.9990 | Val: 0.2082/0.9477 | LR: 2.46e-05 ★
        Epoch 72/90 | Train: 0.1855/0.9987 | Val: 0.2074/0.9477 | LR: 1.39e-05
        Epoch 73/90 | Train: 0.1850/0.9991 | Val: 0.2066/0.9477 | LR: 6.26e-06
        Epoch 74/90 | Train: 0.1820/0.9967 | Val: 0.2059/0.9477 | LR: 1.64e-06
        Epoch 75/90 | Train: 0.1811/0.9963 | Val: 0.2053/0.9477 | LR: 1.00e-03
        Epoch 76/90 | Train: 0.1973/0.9920 | Val: 0.2043/0.9477 | LR: 1.00e-03
        Epoch 77/90 | Train: 0.1974/0.9929 | Val: 0.2039/0.9477 | LR: 9.98e-04
        Epoch 78/90 | Train: 0.2026/0.9980 | Val: 0.2034/0.9477 | LR: 9.97e-04
        Epoch 79/90 | Train: 0.1927/0.9975 | Val: 0.2027/0.9477 | LR: 9.94e-04
        Epoch 80/90 | Train: 0.1977/0.9923 | Val: 0.2019/0.9477 | LR: 9.90e-04
        Epoch 81/90 | Train: 0.1975/0.9939 | Val: 0.2009/0.9477 | LR: 9.86e-04
        Epoch 82/90 | Train: 0.1963/0.9930 | Val: 0.2001/0.9446 | LR: 9.81e-04
        Epoch 83/90 | Train: 0.2031/0.9928 | Val: 0.1995/0.9446 | LR: 9.76e-04
        Epoch 84/90 | Train: 0.1894/0.9986 | Val: 0.1991/0.9446 | LR: 9.69e-04
        Epoch 85/90 | Train: 0.1995/0.9972 | Val: 0.1990/0.9446 | LR: 9.62e-04
        Epoch 86/90 | Train: 0.2093/0.9941 | Val: 0.1981/0.9446 | LR: 9.54e-04
        Early stopping at epoch 86 (patience=15)
    
      Best: Epoch 71, Val F1: 0.9477
      DINOv2 fold 0: val F1 = 0.9477
    
    ============================================================
    FOLD 1 — dinov2 (vit_base_patch14_reg4_dinov2.lvd142m)
    Epochs: 90 | Freeze: 10 | Warmup: 5 | CutMix: True
    ============================================================
    
      Loaded linear probe head → dinov2_probe_fold1.pth
    
      Total:    85.7M params
      Head:     4.6K params (lr=1e-03)
      Backbone: 85.7M params (lr=1e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (10 epochs, no CutMix)
        Epoch 1/10 | Train: 0.8075/0.8617 | Val: 0.3987/0.8468
        Epoch 2/10 | Train: 0.6325/0.8642 | Val: 0.3894/0.8562
        Epoch 3/10 | Train: 0.5431/0.8792 | Val: 0.3787/0.8549
        Epoch 4/10 | Train: 0.5629/0.8859 | Val: 0.3680/0.8590
        Epoch 5/10 | Train: 0.4947/0.8932 | Val: 0.3576/0.8590
        Epoch 6/10 | Train: 0.5108/0.8956 | Val: 0.3472/0.8614
        Epoch 7/10 | Train: 0.4951/0.8978 | Val: 0.3376/0.8752
        Epoch 8/10 | Train: 0.4693/0.9064 | Val: 0.3288/0.8778
        Epoch 9/10 | Train: 0.4746/0.9054 | Val: 0.3207/0.8804
        Epoch 10/10 | Train: 0.4512/0.9127 | Val: 0.3134/0.8804
    
      Stage 2: Full fine-tuning (80 epochs, CutMix=True)
        Epoch 11/90 | Train: 0.4036/0.9203 | Val: 0.2323/0.8867 | LR: 2.80e-04 ★
        Epoch 12/90 | Train: 0.4175/0.9237 | Val: 0.2321/0.8867 | LR: 4.60e-04
        Epoch 13/90 | Train: 0.3852/0.9288 | Val: 0.2318/0.8867 | LR: 6.40e-04
        Epoch 14/90 | Train: 0.3664/0.9229 | Val: 0.2315/0.8867 | LR: 8.20e-04
        Epoch 15/90 | Train: 0.3800/0.9289 | Val: 0.2306/0.8867 | LR: 1.00e-03
        Epoch 16/90 | Train: 0.3030/0.9536 | Val: 0.2296/0.8867 | LR: 9.94e-04
        Epoch 17/90 | Train: 0.2886/0.9659 | Val: 0.2292/0.8900 | LR: 9.76e-04 ★
        Epoch 18/90 | Train: 0.2914/0.9689 | Val: 0.2285/0.8900 | LR: 9.46e-04
        Epoch 19/90 | Train: 0.2818/0.9685 | Val: 0.2278/0.8900 | LR: 9.05e-04
        Epoch 20/90 | Train: 0.2706/0.9709 | Val: 0.2270/0.8900 | LR: 8.54e-04
        Epoch 21/90 | Train: 0.2526/0.9786 | Val: 0.2262/0.8900 | LR: 7.94e-04
        Epoch 22/90 | Train: 0.2587/0.9760 | Val: 0.2254/0.8900 | LR: 7.27e-04
        Epoch 23/90 | Train: 0.2325/0.9838 | Val: 0.2246/0.8900 | LR: 6.55e-04
        Epoch 24/90 | Train: 0.2151/0.9863 | Val: 0.2236/0.8938 | LR: 5.78e-04 ★
        Epoch 25/90 | Train: 0.2407/0.9814 | Val: 0.2234/0.8938 | LR: 5.00e-04
        Epoch 26/90 | Train: 0.2349/0.9787 | Val: 0.2231/0.8969 | LR: 4.22e-04 ★
        Epoch 27/90 | Train: 0.2358/0.9829 | Val: 0.2224/0.8969 | LR: 3.46e-04
        Epoch 28/90 | Train: 0.2112/0.9949 | Val: 0.2216/0.8969 | LR: 2.73e-04
        Epoch 29/90 | Train: 0.2090/0.9912 | Val: 0.2212/0.8938 | LR: 2.06e-04
        Epoch 30/90 | Train: 0.2105/0.9931 | Val: 0.2205/0.8938 | LR: 1.47e-04
        Epoch 31/90 | Train: 0.1980/0.9946 | Val: 0.2200/0.8964 | LR: 9.56e-05
        Epoch 32/90 | Train: 0.2112/0.9900 | Val: 0.2197/0.8990 | LR: 5.46e-05 ★
        Epoch 33/90 | Train: 0.2021/0.9927 | Val: 0.2193/0.8990 | LR: 2.46e-05
        Epoch 34/90 | Train: 0.2126/0.9926 | Val: 0.2189/0.8990 | LR: 6.26e-06
        Epoch 35/90 | Train: 0.2041/0.9964 | Val: 0.2185/0.8990 | LR: 1.00e-03
        Epoch 36/90 | Train: 0.2370/0.9850 | Val: 0.2185/0.9016 | LR: 9.98e-04 ★
        Epoch 37/90 | Train: 0.2369/0.9831 | Val: 0.2182/0.9016 | LR: 9.94e-04
        Epoch 38/90 | Train: 0.2290/0.9847 | Val: 0.2173/0.9016 | LR: 9.86e-04
        Epoch 39/90 | Train: 0.2174/0.9879 | Val: 0.2165/0.9016 | LR: 9.76e-04
        Epoch 40/90 | Train: 0.2264/0.9853 | Val: 0.2158/0.9016 | LR: 9.62e-04
        Epoch 41/90 | Train: 0.2032/0.9907 | Val: 0.2152/0.9016 | LR: 9.46e-04
        Epoch 42/90 | Train: 0.2216/0.9889 | Val: 0.2145/0.9041 | LR: 9.26e-04 ★
        Epoch 43/90 | Train: 0.2074/0.9891 | Val: 0.2136/0.9041 | LR: 9.05e-04
        Epoch 44/90 | Train: 0.2188/0.9903 | Val: 0.2128/0.9041 | LR: 8.80e-04
        Epoch 45/90 | Train: 0.2320/0.9919 | Val: 0.2121/0.9041 | LR: 8.54e-04
        Epoch 46/90 | Train: 0.1990/0.9961 | Val: 0.2114/0.9041 | LR: 8.25e-04
        Epoch 47/90 | Train: 0.2030/0.9923 | Val: 0.2107/0.9041 | LR: 7.94e-04
        Epoch 48/90 | Train: 0.1971/0.9940 | Val: 0.2094/0.9067 | LR: 7.61e-04 ★
        Epoch 49/90 | Train: 0.2168/0.9937 | Val: 0.2081/0.9105 | LR: 7.27e-04 ★
        Epoch 50/90 | Train: 0.2099/0.9938 | Val: 0.2068/0.9105 | LR: 6.91e-04
        Epoch 51/90 | Train: 0.2014/0.9955 | Val: 0.2059/0.9105 | LR: 6.55e-04
        Epoch 52/90 | Train: 0.2001/0.9960 | Val: 0.2048/0.9105 | LR: 6.17e-04
        Epoch 53/90 | Train: 0.1921/0.9952 | Val: 0.2036/0.9105 | LR: 5.78e-04
        Epoch 54/90 | Train: 0.1875/0.9981 | Val: 0.2026/0.9105 | LR: 5.39e-04
        Epoch 55/90 | Train: 0.1952/0.9966 | Val: 0.2013/0.9105 | LR: 5.00e-04
        Epoch 56/90 | Train: 0.1975/0.9946 | Val: 0.2008/0.9105 | LR: 4.61e-04
        Epoch 57/90 | Train: 0.1958/0.9927 | Val: 0.2000/0.9105 | LR: 4.22e-04
        Epoch 58/90 | Train: 0.1925/0.9973 | Val: 0.1992/0.9105 | LR: 3.83e-04
        Epoch 59/90 | Train: 0.1908/0.9975 | Val: 0.1983/0.9105 | LR: 3.46e-04
        Epoch 60/90 | Train: 0.1910/0.9938 | Val: 0.1976/0.9105 | LR: 3.09e-04
        Epoch 61/90 | Train: 0.1910/0.9954 | Val: 0.1969/0.9136 | LR: 2.73e-04 ★
        Epoch 62/90 | Train: 0.1906/0.9973 | Val: 0.1965/0.9136 | LR: 2.39e-04
        Epoch 63/90 | Train: 0.1855/0.9980 | Val: 0.1961/0.9136 | LR: 2.06e-04
        Epoch 64/90 | Train: 0.1859/0.9968 | Val: 0.1957/0.9136 | LR: 1.75e-04
        Epoch 65/90 | Train: 0.1848/0.9951 | Val: 0.1951/0.9136 | LR: 1.47e-04
        Epoch 66/90 | Train: 0.1753/0.9982 | Val: 0.1943/0.9136 | LR: 1.20e-04
        Epoch 67/90 | Train: 0.2011/0.9970 | Val: 0.1937/0.9136 | LR: 9.56e-05
        Epoch 68/90 | Train: 0.1830/0.9974 | Val: 0.1931/0.9136 | LR: 7.38e-05
        Epoch 69/90 | Train: 0.1817/0.9993 | Val: 0.1926/0.9136 | LR: 5.46e-05
        Epoch 70/90 | Train: 0.1821/0.9952 | Val: 0.1923/0.9136 | LR: 3.82e-05
        Epoch 71/90 | Train: 0.1815/0.9988 | Val: 0.1918/0.9136 | LR: 2.46e-05
        Epoch 72/90 | Train: 0.1841/0.9969 | Val: 0.1914/0.9136 | LR: 1.39e-05
        Epoch 73/90 | Train: 0.1819/0.9952 | Val: 0.1908/0.9136 | LR: 6.26e-06
        Epoch 74/90 | Train: 0.1771/0.9991 | Val: 0.1904/0.9136 | LR: 1.64e-06
        Epoch 75/90 | Train: 0.1814/0.9985 | Val: 0.1901/0.9136 | LR: 1.00e-03
        Epoch 76/90 | Train: 0.1785/0.9978 | Val: 0.1895/0.9136 | LR: 1.00e-03
        Early stopping at epoch 76 (patience=15)
    
      Best: Epoch 61, Val F1: 0.9136
      DINOv2 fold 1: val F1 = 0.9136
    
    ============================================================
    FOLD 2 — dinov2 (vit_base_patch14_reg4_dinov2.lvd142m)
    Epochs: 90 | Freeze: 10 | Warmup: 5 | CutMix: True
    ============================================================
    
      Loaded linear probe head → dinov2_probe_fold2.pth
    
      Total:    85.7M params
      Head:     4.6K params (lr=1e-03)
      Backbone: 85.7M params (lr=1e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (10 epochs, no CutMix)
        Epoch 1/10 | Train: 0.8276/0.8464 | Val: 0.3645/0.8743
        Epoch 2/10 | Train: 0.6401/0.8567 | Val: 0.3561/0.8810
        Epoch 3/10 | Train: 0.5690/0.8735 | Val: 0.3460/0.8888
        Epoch 4/10 | Train: 0.5401/0.8736 | Val: 0.3353/0.8915
        Epoch 5/10 | Train: 0.5070/0.8914 | Val: 0.3249/0.8915
        Epoch 6/10 | Train: 0.5056/0.8955 | Val: 0.3148/0.8915
        Epoch 7/10 | Train: 0.4744/0.9059 | Val: 0.3052/0.8915
        Epoch 8/10 | Train: 0.4945/0.8866 | Val: 0.2966/0.8915
        Epoch 9/10 | Train: 0.4667/0.9112 | Val: 0.2886/0.8915
        Epoch 10/10 | Train: 0.4536/0.9126 | Val: 0.2816/0.8915
    
      Stage 2: Full fine-tuning (80 epochs, CutMix=True)
        Epoch 11/90 | Train: 0.4563/0.9074 | Val: 0.2263/0.8946 | LR: 2.80e-04 ★
        Epoch 12/90 | Train: 0.3886/0.9234 | Val: 0.2264/0.8946 | LR: 4.60e-04
        Epoch 13/90 | Train: 0.3690/0.9306 | Val: 0.2259/0.8946 | LR: 6.40e-04
        Epoch 14/90 | Train: 0.3581/0.9405 | Val: 0.2248/0.8912 | LR: 8.20e-04
        Epoch 15/90 | Train: 0.3231/0.9477 | Val: 0.2240/0.8955 | LR: 1.00e-03 ★
        Epoch 16/90 | Train: 0.3170/0.9490 | Val: 0.2227/0.8955 | LR: 9.94e-04
        Epoch 17/90 | Train: 0.3240/0.9488 | Val: 0.2214/0.8955 | LR: 9.76e-04
        Epoch 18/90 | Train: 0.2900/0.9647 | Val: 0.2200/0.8955 | LR: 9.46e-04
        Epoch 19/90 | Train: 0.2691/0.9731 | Val: 0.2191/0.8955 | LR: 9.05e-04
        Epoch 20/90 | Train: 0.2648/0.9688 | Val: 0.2182/0.8955 | LR: 8.54e-04
        Epoch 21/90 | Train: 0.2736/0.9672 | Val: 0.2172/0.8955 | LR: 7.94e-04
        Epoch 22/90 | Train: 0.2531/0.9828 | Val: 0.2164/0.8955 | LR: 7.27e-04
        Epoch 23/90 | Train: 0.2512/0.9776 | Val: 0.2157/0.8993 | LR: 6.55e-04 ★
        Epoch 24/90 | Train: 0.2280/0.9860 | Val: 0.2148/0.9019 | LR: 5.78e-04 ★
        Epoch 25/90 | Train: 0.2262/0.9824 | Val: 0.2136/0.9019 | LR: 5.00e-04
        Epoch 26/90 | Train: 0.2208/0.9872 | Val: 0.2123/0.9074 | LR: 4.22e-04 ★
        Epoch 27/90 | Train: 0.2190/0.9874 | Val: 0.2115/0.9111 | LR: 3.46e-04 ★
        Epoch 28/90 | Train: 0.2241/0.9838 | Val: 0.2107/0.9015 | LR: 2.73e-04
        Epoch 29/90 | Train: 0.2090/0.9892 | Val: 0.2098/0.9041 | LR: 2.06e-04
        Epoch 30/90 | Train: 0.2045/0.9891 | Val: 0.2089/0.9041 | LR: 1.47e-04
        Epoch 31/90 | Train: 0.2091/0.9890 | Val: 0.2083/0.9041 | LR: 9.56e-05
        Epoch 32/90 | Train: 0.2257/0.9914 | Val: 0.2077/0.9041 | LR: 5.46e-05
        Epoch 33/90 | Train: 0.2004/0.9902 | Val: 0.2070/0.9089 | LR: 2.46e-05
        Epoch 34/90 | Train: 0.2047/0.9924 | Val: 0.2064/0.9089 | LR: 6.26e-06
        Epoch 35/90 | Train: 0.1947/0.9934 | Val: 0.2060/0.9114 | LR: 1.00e-03 ★
        Epoch 36/90 | Train: 0.2141/0.9895 | Val: 0.2055/0.9114 | LR: 9.98e-04
        Epoch 37/90 | Train: 0.2201/0.9911 | Val: 0.2052/0.9114 | LR: 9.94e-04
        Epoch 38/90 | Train: 0.2145/0.9896 | Val: 0.2050/0.9114 | LR: 9.86e-04
        Epoch 39/90 | Train: 0.2313/0.9890 | Val: 0.2041/0.9149 | LR: 9.76e-04 ★
        Epoch 40/90 | Train: 0.2216/0.9874 | Val: 0.2034/0.9087 | LR: 9.62e-04
        Epoch 41/90 | Train: 0.2104/0.9938 | Val: 0.2036/0.9087 | LR: 9.46e-04
        Epoch 42/90 | Train: 0.2170/0.9875 | Val: 0.2036/0.9087 | LR: 9.26e-04
        Epoch 43/90 | Train: 0.2340/0.9846 | Val: 0.2037/0.9087 | LR: 9.05e-04
        Epoch 44/90 | Train: 0.2130/0.9943 | Val: 0.2031/0.9087 | LR: 8.80e-04
        Epoch 45/90 | Train: 0.2058/0.9882 | Val: 0.2026/0.9113 | LR: 8.54e-04
        Epoch 46/90 | Train: 0.2003/0.9937 | Val: 0.2021/0.9113 | LR: 8.25e-04
        Epoch 47/90 | Train: 0.2079/0.9925 | Val: 0.2012/0.9139 | LR: 7.94e-04
        Epoch 48/90 | Train: 0.2174/0.9891 | Val: 0.2005/0.9139 | LR: 7.61e-04
        Epoch 49/90 | Train: 0.2070/0.9932 | Val: 0.1995/0.9139 | LR: 7.27e-04
        Epoch 50/90 | Train: 0.1947/0.9941 | Val: 0.1987/0.9139 | LR: 6.91e-04
        Epoch 51/90 | Train: 0.1978/0.9942 | Val: 0.1987/0.9139 | LR: 6.55e-04
        Epoch 52/90 | Train: 0.1906/0.9956 | Val: 0.1981/0.9139 | LR: 6.17e-04
        Epoch 53/90 | Train: 0.2078/0.9915 | Val: 0.1978/0.9137 | LR: 5.78e-04
        Epoch 54/90 | Train: 0.1896/0.9963 | Val: 0.1979/0.9137 | LR: 5.39e-04
        Early stopping at epoch 54 (patience=15)
    
      Best: Epoch 39, Val F1: 0.9149
      DINOv2 fold 2: val F1 = 0.9149
    
    ============================================================
    FOLD 3 — dinov2 (vit_base_patch14_reg4_dinov2.lvd142m)
    Epochs: 90 | Freeze: 10 | Warmup: 5 | CutMix: True
    ============================================================
    
      Loaded linear probe head → dinov2_probe_fold3.pth
    
      Total:    85.7M params
      Head:     4.6K params (lr=1e-03)
      Backbone: 85.7M params (lr=1e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (10 epochs, no CutMix)
        Epoch 1/10 | Train: 0.8293/0.8595 | Val: 0.3388/0.8419
        Epoch 2/10 | Train: 0.6430/0.8621 | Val: 0.3330/0.8474
        Epoch 3/10 | Train: 0.5777/0.8785 | Val: 0.3259/0.8506
        Epoch 4/10 | Train: 0.5148/0.8756 | Val: 0.3183/0.8533
        Epoch 5/10 | Train: 0.5245/0.8780 | Val: 0.3106/0.8533
        Epoch 6/10 | Train: 0.4881/0.8997 | Val: 0.3031/0.8533
        Epoch 7/10 | Train: 0.4966/0.8943 | Val: 0.2960/0.8533
        Epoch 8/10 | Train: 0.4619/0.8908 | Val: 0.2891/0.8533
        Epoch 9/10 | Train: 0.4636/0.9112 | Val: 0.2823/0.8533
        Epoch 10/10 | Train: 0.4393/0.9138 | Val: 0.2761/0.8653
    
      Stage 2: Full fine-tuning (80 epochs, CutMix=True)
        Epoch 11/90 | Train: 0.4668/0.9012 | Val: 0.2103/0.8870 | LR: 2.80e-04 ★
        Epoch 12/90 | Train: 0.4022/0.9137 | Val: 0.2102/0.8870 | LR: 4.60e-04
        Epoch 13/90 | Train: 0.3700/0.9271 | Val: 0.2101/0.8870 | LR: 6.40e-04
        Epoch 14/90 | Train: 0.3750/0.9373 | Val: 0.2099/0.8870 | LR: 8.20e-04
        Epoch 15/90 | Train: 0.3499/0.9401 | Val: 0.2098/0.8835 | LR: 1.00e-03
        Epoch 16/90 | Train: 0.2995/0.9547 | Val: 0.2098/0.8798 | LR: 9.94e-04
        Epoch 17/90 | Train: 0.3232/0.9444 | Val: 0.2096/0.8857 | LR: 9.76e-04
        Epoch 18/90 | Train: 0.2858/0.9699 | Val: 0.2094/0.8857 | LR: 9.46e-04
        Epoch 19/90 | Train: 0.2757/0.9698 | Val: 0.2093/0.8857 | LR: 9.05e-04
        Epoch 20/90 | Train: 0.2662/0.9705 | Val: 0.2093/0.8823 | LR: 8.54e-04
        Epoch 21/90 | Train: 0.2817/0.9711 | Val: 0.2088/0.8823 | LR: 7.94e-04
        Epoch 22/90 | Train: 0.2548/0.9753 | Val: 0.2087/0.8832 | LR: 7.27e-04
        Epoch 23/90 | Train: 0.2519/0.9802 | Val: 0.2089/0.8890 | LR: 6.55e-04 ★
        Epoch 24/90 | Train: 0.2289/0.9837 | Val: 0.2093/0.8890 | LR: 5.78e-04
        Epoch 25/90 | Train: 0.2279/0.9790 | Val: 0.2098/0.8890 | LR: 5.00e-04
        Epoch 26/90 | Train: 0.2347/0.9849 | Val: 0.2099/0.8890 | LR: 4.22e-04
        Epoch 27/90 | Train: 0.2328/0.9808 | Val: 0.2101/0.8936 | LR: 3.46e-04 ★
        Epoch 28/90 | Train: 0.2090/0.9894 | Val: 0.2102/0.8936 | LR: 2.73e-04
        Epoch 29/90 | Train: 0.2191/0.9914 | Val: 0.2105/0.8936 | LR: 2.06e-04
        Epoch 30/90 | Train: 0.2094/0.9861 | Val: 0.2107/0.8936 | LR: 1.47e-04
        Epoch 31/90 | Train: 0.2088/0.9906 | Val: 0.2107/0.8936 | LR: 9.56e-05
        Epoch 32/90 | Train: 0.2087/0.9900 | Val: 0.2109/0.8936 | LR: 5.46e-05
        Epoch 33/90 | Train: 0.2144/0.9894 | Val: 0.2113/0.8966 | LR: 2.46e-05 ★
        Epoch 34/90 | Train: 0.2167/0.9880 | Val: 0.2115/0.9052 | LR: 6.26e-06 ★
        Epoch 35/90 | Train: 0.1980/0.9949 | Val: 0.2118/0.9052 | LR: 1.00e-03
        Epoch 36/90 | Train: 0.2218/0.9883 | Val: 0.2117/0.9052 | LR: 9.98e-04
        Epoch 37/90 | Train: 0.2466/0.9868 | Val: 0.2123/0.9012 | LR: 9.94e-04
        Epoch 38/90 | Train: 0.2195/0.9882 | Val: 0.2122/0.9012 | LR: 9.86e-04
        Epoch 39/90 | Train: 0.2260/0.9883 | Val: 0.2120/0.9012 | LR: 9.76e-04
        Epoch 40/90 | Train: 0.2154/0.9833 | Val: 0.2118/0.9012 | LR: 9.62e-04
        Epoch 41/90 | Train: 0.2313/0.9853 | Val: 0.2120/0.9012 | LR: 9.46e-04
        Epoch 42/90 | Train: 0.2058/0.9956 | Val: 0.2118/0.9043 | LR: 9.26e-04
        Epoch 43/90 | Train: 0.2274/0.9862 | Val: 0.2115/0.9043 | LR: 9.05e-04
        Epoch 44/90 | Train: 0.2100/0.9918 | Val: 0.2116/0.9078 | LR: 8.80e-04 ★
        Epoch 45/90 | Train: 0.2044/0.9877 | Val: 0.2113/0.9078 | LR: 8.54e-04
        Epoch 46/90 | Train: 0.1973/0.9944 | Val: 0.2110/0.9078 | LR: 8.25e-04
        Epoch 47/90 | Train: 0.2265/0.9866 | Val: 0.2109/0.9078 | LR: 7.94e-04
        Epoch 48/90 | Train: 0.2032/0.9921 | Val: 0.2101/0.9078 | LR: 7.61e-04
        Epoch 49/90 | Train: 0.2045/0.9922 | Val: 0.2095/0.9078 | LR: 7.27e-04
        Epoch 50/90 | Train: 0.2085/0.9903 | Val: 0.2090/0.9078 | LR: 6.91e-04
        Epoch 51/90 | Train: 0.2171/0.9928 | Val: 0.2087/0.9106 | LR: 6.55e-04 ★
        Epoch 52/90 | Train: 0.1959/0.9944 | Val: 0.2085/0.9133 | LR: 6.17e-04 ★
        Epoch 53/90 | Train: 0.1962/0.9940 | Val: 0.2087/0.9133 | LR: 5.78e-04
        Epoch 54/90 | Train: 0.1924/0.9983 | Val: 0.2090/0.9136 | LR: 5.39e-04 ★
        Epoch 55/90 | Train: 0.1928/0.9932 | Val: 0.2087/0.9136 | LR: 5.00e-04
        Epoch 56/90 | Train: 0.1924/0.9938 | Val: 0.2084/0.9136 | LR: 4.61e-04
        Epoch 57/90 | Train: 0.1943/0.9957 | Val: 0.2080/0.9136 | LR: 4.22e-04
        Epoch 58/90 | Train: 0.1847/0.9987 | Val: 0.2077/0.9136 | LR: 3.83e-04
        Epoch 59/90 | Train: 0.1906/0.9953 | Val: 0.2073/0.9112 | LR: 3.46e-04
        Epoch 60/90 | Train: 0.1921/0.9962 | Val: 0.2070/0.9152 | LR: 3.09e-04 ★
        Epoch 61/90 | Train: 0.1987/0.9937 | Val: 0.2065/0.9152 | LR: 2.73e-04
        Epoch 62/90 | Train: 0.1881/0.9965 | Val: 0.2063/0.9152 | LR: 2.39e-04
        Epoch 63/90 | Train: 0.1939/0.9981 | Val: 0.2060/0.9116 | LR: 2.06e-04
        Epoch 64/90 | Train: 0.1853/0.9978 | Val: 0.2058/0.9082 | LR: 1.75e-04
        Epoch 65/90 | Train: 0.1919/0.9981 | Val: 0.2053/0.9082 | LR: 1.47e-04
        Epoch 66/90 | Train: 0.1794/0.9987 | Val: 0.2048/0.9057 | LR: 1.20e-04
        Epoch 67/90 | Train: 0.1841/0.9980 | Val: 0.2045/0.9057 | LR: 9.56e-05
        Epoch 68/90 | Train: 0.1838/0.9977 | Val: 0.2042/0.9057 | LR: 7.38e-05
        Epoch 69/90 | Train: 0.1855/0.9975 | Val: 0.2039/0.9057 | LR: 5.46e-05
        Epoch 70/90 | Train: 0.1822/0.9982 | Val: 0.2035/0.9057 | LR: 3.82e-05
        Epoch 71/90 | Train: 0.1811/0.9956 | Val: 0.2032/0.9033 | LR: 2.46e-05
        Epoch 72/90 | Train: 0.1818/0.9988 | Val: 0.2028/0.8992 | LR: 1.39e-05
        Epoch 73/90 | Train: 0.1930/0.9959 | Val: 0.2027/0.9004 | LR: 6.26e-06
        Epoch 74/90 | Train: 0.1850/0.9970 | Val: 0.2025/0.9004 | LR: 1.64e-06
        Epoch 75/90 | Train: 0.1822/1.0000 | Val: 0.2023/0.9004 | LR: 1.00e-03
        Early stopping at epoch 75 (patience=15)
    
      Best: Epoch 60, Val F1: 0.9152
      DINOv2 fold 3: val F1 = 0.9152
    
    ============================================================
    FOLD 4 — dinov2 (vit_base_patch14_reg4_dinov2.lvd142m)
    Epochs: 90 | Freeze: 10 | Warmup: 5 | CutMix: True
    ============================================================
    
      Loaded linear probe head → dinov2_probe_fold4.pth
    
      Total:    85.7M params
      Head:     4.6K params (lr=1e-03)
      Backbone: 85.7M params (lr=1e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (10 epochs, no CutMix)
        Epoch 1/10 | Train: 0.8375/0.8611 | Val: 0.2973/0.8817
        Epoch 2/10 | Train: 0.6398/0.8589 | Val: 0.2890/0.8790
        Epoch 3/10 | Train: 0.6076/0.8602 | Val: 0.2791/0.8911
        Epoch 4/10 | Train: 0.5397/0.8857 | Val: 0.2687/0.8939
        Epoch 5/10 | Train: 0.5014/0.8936 | Val: 0.2585/0.8939
        Epoch 6/10 | Train: 0.5015/0.8816 | Val: 0.2489/0.8939
        Epoch 7/10 | Train: 0.4990/0.8786 | Val: 0.2397/0.8939
        Epoch 8/10 | Train: 0.4681/0.8993 | Val: 0.2313/0.8915
        Epoch 9/10 | Train: 0.4967/0.9010 | Val: 0.2236/0.9011
        Epoch 10/10 | Train: 0.4373/0.9072 | Val: 0.2163/0.9039
    
      Stage 2: Full fine-tuning (80 epochs, CutMix=True)
        Epoch 11/90 | Train: 0.4654/0.9107 | Val: 0.1683/0.9173 | LR: 2.80e-04 ★
        Epoch 12/90 | Train: 0.4269/0.9145 | Val: 0.1676/0.9173 | LR: 4.60e-04
        Epoch 13/90 | Train: 0.3800/0.9351 | Val: 0.1665/0.9141 | LR: 6.40e-04
        Epoch 14/90 | Train: 0.3900/0.9236 | Val: 0.1657/0.9141 | LR: 8.20e-04
        Epoch 15/90 | Train: 0.3368/0.9447 | Val: 0.1644/0.9172 | LR: 1.00e-03
        Epoch 16/90 | Train: 0.2963/0.9583 | Val: 0.1630/0.9172 | LR: 9.94e-04
        Epoch 17/90 | Train: 0.2961/0.9608 | Val: 0.1610/0.9146 | LR: 9.76e-04
        Epoch 18/90 | Train: 0.3022/0.9638 | Val: 0.1594/0.9191 | LR: 9.46e-04 ★
        Epoch 19/90 | Train: 0.2897/0.9664 | Val: 0.1578/0.9191 | LR: 9.05e-04
        Epoch 20/90 | Train: 0.2589/0.9747 | Val: 0.1560/0.9191 | LR: 8.54e-04
        Epoch 21/90 | Train: 0.2639/0.9782 | Val: 0.1546/0.9191 | LR: 7.94e-04
        Epoch 22/90 | Train: 0.2587/0.9731 | Val: 0.1534/0.9216 | LR: 7.27e-04 ★
        Epoch 23/90 | Train: 0.2490/0.9734 | Val: 0.1522/0.9216 | LR: 6.55e-04
        Epoch 24/90 | Train: 0.2514/0.9724 | Val: 0.1514/0.9216 | LR: 5.78e-04
        Epoch 25/90 | Train: 0.2380/0.9821 | Val: 0.1498/0.9216 | LR: 5.00e-04
        Epoch 26/90 | Train: 0.2643/0.9795 | Val: 0.1486/0.9242 | LR: 4.22e-04 ★
        Epoch 27/90 | Train: 0.2343/0.9875 | Val: 0.1473/0.9242 | LR: 3.46e-04
        Epoch 28/90 | Train: 0.2117/0.9907 | Val: 0.1459/0.9242 | LR: 2.73e-04
        Epoch 29/90 | Train: 0.2082/0.9894 | Val: 0.1446/0.9242 | LR: 2.06e-04
        Epoch 30/90 | Train: 0.2236/0.9873 | Val: 0.1434/0.9307 | LR: 1.47e-04 ★
        Epoch 31/90 | Train: 0.2159/0.9923 | Val: 0.1423/0.9307 | LR: 9.56e-05
        Epoch 32/90 | Train: 0.2097/0.9890 | Val: 0.1413/0.9307 | LR: 5.46e-05
        Epoch 33/90 | Train: 0.2089/0.9887 | Val: 0.1404/0.9310 | LR: 2.46e-05 ★
        Epoch 34/90 | Train: 0.1998/0.9934 | Val: 0.1393/0.9357 | LR: 6.26e-06 ★
        Epoch 35/90 | Train: 0.2034/0.9912 | Val: 0.1385/0.9357 | LR: 1.00e-03
        Epoch 36/90 | Train: 0.2139/0.9940 | Val: 0.1375/0.9354 | LR: 9.98e-04
        Epoch 37/90 | Train: 0.2360/0.9817 | Val: 0.1368/0.9354 | LR: 9.94e-04
        Epoch 38/90 | Train: 0.2379/0.9856 | Val: 0.1358/0.9354 | LR: 9.86e-04
        Epoch 39/90 | Train: 0.2456/0.9857 | Val: 0.1353/0.9354 | LR: 9.76e-04
        Epoch 40/90 | Train: 0.2304/0.9888 | Val: 0.1347/0.9354 | LR: 9.62e-04
        Epoch 41/90 | Train: 0.2335/0.9912 | Val: 0.1341/0.9354 | LR: 9.46e-04
        Epoch 42/90 | Train: 0.2161/0.9930 | Val: 0.1329/0.9354 | LR: 9.26e-04
        Epoch 43/90 | Train: 0.2307/0.9864 | Val: 0.1322/0.9357 | LR: 9.05e-04
        Epoch 44/90 | Train: 0.2134/0.9867 | Val: 0.1313/0.9357 | LR: 8.80e-04
        Epoch 45/90 | Train: 0.2091/0.9937 | Val: 0.1305/0.9357 | LR: 8.54e-04
        Epoch 46/90 | Train: 0.2062/0.9901 | Val: 0.1298/0.9357 | LR: 8.25e-04
        Epoch 47/90 | Train: 0.2028/0.9913 | Val: 0.1293/0.9324 | LR: 7.94e-04
        Epoch 48/90 | Train: 0.2078/0.9909 | Val: 0.1288/0.9324 | LR: 7.61e-04
        Epoch 49/90 | Train: 0.2114/0.9929 | Val: 0.1282/0.9324 | LR: 7.27e-04
        Early stopping at epoch 49 (patience=15)
    
      Best: Epoch 34, Val F1: 0.9357
      DINOv2 fold 4: val F1 = 0.9357
    
    DINOv2 summary: mean=0.9254 ± 0.0138
      exp06 baseline: mean=0.9321 ± 0.0146


## 9 · Train EfficientNet-B4  (70 epochs)


```python
effnet_results = {}

for fold in range(N_FOLDS):
    result = train_fold(
        fold         = fold,
        model_key    = 'effnet_b4',
        train_df     = train_df,
        class_weights= class_weights,
        use_cutmix   = True,
        use_sampler  = False,
        exp_id       = EXP_ID,
    )
    effnet_results[fold] = result
    print(f'  EfficientNet-B4 fold {fold}: val F1 = {result["best_f1"]:.4f}')

scores = [r['best_f1'] for r in effnet_results.values()]
print(f'\nEfficientNet-B4 summary: mean={np.mean(scores):.4f} ± {np.std(scores):.4f}')
print(f'  exp06 baseline: mean=0.8902 ± 0.0309')
```

    
    ============================================================
    FOLD 0 — effnet_b4 (efficientnet_b4)
    Epochs: 90 | Freeze: 5 | Warmup: 3 | CutMix: True
    ============================================================



    model.safetensors:   0%|          | 0.00/77.9M [00:00<?, ?B/s]


    
      Total:    17.6M params
      Head:     10.8K params (lr=1e-03)
      Backbone: 17.5M params (lr=1e-04)
      EMA:      decay=0.999
    
      Stage 1: Head-only (5 epochs, no CutMix)
        Epoch 1/5 | Train: 3.3872/0.2263 | Val: 1.8610/0.1353
        Epoch 2/5 | Train: 2.4981/0.4461 | Val: 1.7907/0.1467
        Epoch 3/5 | Train: 2.0064/0.5597 | Val: 1.6940/0.1617
        Epoch 4/5 | Train: 1.7320/0.6097 | Val: 1.5863/0.1929
        Epoch 5/5 | Train: 1.5684/0.6402 | Val: 1.4706/0.2303
    
      Stage 2: Full fine-tuning (85 epochs, CutMix=True)
        Epoch 06/90 | Train: 1.3014/0.6792 | Val: 0.5264/0.7628 | LR: 4.00e-05 ★
        Epoch 07/90 | Train: 0.9616/0.7622 | Val: 0.5122/0.7757 | LR: 7.00e-05 ★
        Epoch 08/90 | Train: 0.7045/0.8196 | Val: 0.4926/0.7813 | LR: 1.00e-04 ★
        Epoch 09/90 | Train: 0.5546/0.8684 | Val: 0.4698/0.7813 | LR: 9.94e-05
        Epoch 10/90 | Train: 0.4389/0.9039 | Val: 0.4468/0.7864 | LR: 9.76e-05 ★
        Epoch 11/90 | Train: 0.4049/0.9233 | Val: 0.4241/0.7881 | LR: 9.46e-05 ★
        Epoch 12/90 | Train: 0.3591/0.9354 | Val: 0.4032/0.8088 | LR: 9.05e-05 ★
        Epoch 13/90 | Train: 0.3427/0.9461 | Val: 0.3825/0.8207 | LR: 8.54e-05 ★
        Epoch 14/90 | Train: 0.3295/0.9492 | Val: 0.3628/0.8233 | LR: 7.94e-05 ★
        Epoch 15/90 | Train: 0.2991/0.9598 | Val: 0.3459/0.8256 | LR: 7.27e-05 ★
        Epoch 16/90 | Train: 0.2829/0.9647 | Val: 0.3292/0.8400 | LR: 6.55e-05 ★
        Epoch 17/90 | Train: 0.2361/0.9788 | Val: 0.3146/0.8426 | LR: 5.79e-05 ★
        Epoch 18/90 | Train: 0.2429/0.9818 | Val: 0.3002/0.8625 | LR: 5.01e-05 ★
        Epoch 19/90 | Train: 0.2359/0.9824 | Val: 0.2869/0.8660 | LR: 4.22e-05 ★
        Epoch 20/90 | Train: 0.2261/0.9856 | Val: 0.2771/0.8734 | LR: 3.46e-05 ★
        Epoch 21/90 | Train: 0.2211/0.9853 | Val: 0.2669/0.8824 | LR: 2.74e-05 ★
        Epoch 22/90 | Train: 0.2446/0.9825 | Val: 0.2582/0.8879 | LR: 2.07e-05 ★
        Epoch 23/90 | Train: 0.2459/0.9809 | Val: 0.2497/0.8879 | LR: 1.47e-05
        Epoch 24/90 | Train: 0.2185/0.9917 | Val: 0.2426/0.8951 | LR: 9.64e-06 ★
        Epoch 25/90 | Train: 0.1982/0.9953 | Val: 0.2360/0.9002 | LR: 5.54e-06 ★
        Epoch 26/90 | Train: 0.2188/0.9821 | Val: 0.2293/0.9009 | LR: 2.54e-06 ★
        Epoch 27/90 | Train: 0.2253/0.9887 | Val: 0.2238/0.8961 | LR: 7.15e-07
        Epoch 28/90 | Train: 0.2147/0.9892 | Val: 0.2189/0.8998 | LR: 1.00e-04
        Epoch 29/90 | Train: 0.2260/0.9864 | Val: 0.2156/0.8998 | LR: 9.98e-05
        Epoch 30/90 | Train: 0.2360/0.9854 | Val: 0.2112/0.8998 | LR: 9.94e-05
        Epoch 31/90 | Train: 0.2260/0.9872 | Val: 0.2080/0.9025 | LR: 9.86e-05 ★
        Epoch 32/90 | Train: 0.2422/0.9769 | Val: 0.2043/0.9061 | LR: 9.76e-05 ★
        Epoch 33/90 | Train: 0.2261/0.9849 | Val: 0.2017/0.9061 | LR: 9.62e-05
        Epoch 34/90 | Train: 0.2240/0.9860 | Val: 0.1996/0.9061 | LR: 9.46e-05
        Epoch 35/90 | Train: 0.2443/0.9865 | Val: 0.1975/0.9061 | LR: 9.26e-05
        Epoch 36/90 | Train: 0.2098/0.9892 | Val: 0.1954/0.9065 | LR: 9.05e-05 ★
        Epoch 37/90 | Train: 0.2248/0.9883 | Val: 0.1925/0.9065 | LR: 8.80e-05
        Epoch 38/90 | Train: 0.2223/0.9889 | Val: 0.1901/0.9069 | LR: 8.54e-05 ★
        Epoch 39/90 | Train: 0.2249/0.9893 | Val: 0.1880/0.9069 | LR: 8.25e-05
        Epoch 40/90 | Train: 0.2064/0.9936 | Val: 0.1868/0.9117 | LR: 7.94e-05 ★
        Epoch 41/90 | Train: 0.1946/0.9962 | Val: 0.1855/0.9151 | LR: 7.61e-05 ★
        Epoch 42/90 | Train: 0.1980/0.9896 | Val: 0.1838/0.9117 | LR: 7.27e-05
        Epoch 43/90 | Train: 0.2036/0.9933 | Val: 0.1810/0.9092 | LR: 6.92e-05
        Epoch 44/90 | Train: 0.1935/0.9949 | Val: 0.1785/0.9092 | LR: 6.55e-05
        Epoch 45/90 | Train: 0.1895/0.9979 | Val: 0.1760/0.9092 | LR: 6.17e-05
        Epoch 46/90 | Train: 0.1932/0.9949 | Val: 0.1742/0.9092 | LR: 5.79e-05
        Epoch 47/90 | Train: 0.1896/0.9938 | Val: 0.1727/0.9128 | LR: 5.40e-05
        Epoch 48/90 | Train: 0.2039/0.9943 | Val: 0.1707/0.9163 | LR: 5.01e-05 ★
        Epoch 49/90 | Train: 0.1911/0.9950 | Val: 0.1688/0.9163 | LR: 4.61e-05
        Epoch 50/90 | Train: 0.1792/0.9973 | Val: 0.1675/0.9163 | LR: 4.22e-05
        Epoch 51/90 | Train: 0.1868/0.9926 | Val: 0.1663/0.9216 | LR: 3.84e-05 ★
        Epoch 52/90 | Train: 0.1785/0.9977 | Val: 0.1644/0.9216 | LR: 3.46e-05
        Epoch 53/90 | Train: 0.1888/0.9959 | Val: 0.1630/0.9216 | LR: 3.09e-05
        Epoch 54/90 | Train: 0.1838/0.9980 | Val: 0.1619/0.9237 | LR: 2.74e-05 ★
        Epoch 55/90 | Train: 0.1751/0.9995 | Val: 0.1612/0.9334 | LR: 2.40e-05 ★
        Epoch 56/90 | Train: 0.1760/0.9973 | Val: 0.1612/0.9334 | LR: 2.07e-05
        Epoch 57/90 | Train: 0.1845/0.9977 | Val: 0.1604/0.9316 | LR: 1.76e-05
        Epoch 58/90 | Train: 0.1802/0.9945 | Val: 0.1601/0.9316 | LR: 1.47e-05
        Epoch 59/90 | Train: 0.1895/0.9982 | Val: 0.1597/0.9316 | LR: 1.21e-05
        Epoch 60/90 | Train: 0.1772/0.9977 | Val: 0.1591/0.9316 | LR: 9.64e-06
        Epoch 61/90 | Train: 0.1812/0.9982 | Val: 0.1585/0.9350 | LR: 7.46e-06 ★
        Epoch 62/90 | Train: 0.1790/0.9978 | Val: 0.1577/0.9350 | LR: 5.54e-06
        Epoch 63/90 | Train: 0.1700/1.0000 | Val: 0.1565/0.9350 | LR: 3.90e-06
        Epoch 64/90 | Train: 0.1833/0.9950 | Val: 0.1556/0.9350 | LR: 2.54e-06
        Epoch 65/90 | Train: 0.1704/0.9988 | Val: 0.1553/0.9350 | LR: 1.48e-06
        Epoch 66/90 | Train: 0.1721/0.9993 | Val: 0.1550/0.9350 | LR: 7.15e-07
        Epoch 67/90 | Train: 0.1677/0.9991 | Val: 0.1546/0.9350 | LR: 2.54e-07
        Epoch 68/90 | Train: 0.1717/0.9984 | Val: 0.1545/0.9350 | LR: 1.00e-04
        Epoch 69/90 | Train: 0.1861/0.9968 | Val: 0.1549/0.9350 | LR: 1.00e-04
        Epoch 70/90 | Train: 0.2122/0.9959 | Val: 0.1557/0.9350 | LR: 9.98e-05
        Epoch 71/90 | Train: 0.2053/0.9924 | Val: 0.1564/0.9325 | LR: 9.97e-05
        Epoch 72/90 | Train: 0.1881/0.9940 | Val: 0.1567/0.9375 | LR: 9.94e-05 ★
        Epoch 73/90 | Train: 0.2066/0.9868 | Val: 0.1554/0.9375 | LR: 9.90e-05
        Epoch 74/90 | Train: 0.1955/0.9929 | Val: 0.1547/0.9375 | LR: 9.86e-05
        Epoch 75/90 | Train: 0.1889/0.9944 | Val: 0.1541/0.9375 | LR: 9.81e-05
        Epoch 76/90 | Train: 0.2095/0.9913 | Val: 0.1535/0.9375 | LR: 9.76e-05
        Epoch 77/90 | Train: 0.1946/0.9919 | Val: 0.1534/0.9375 | LR: 9.69e-05
        Epoch 78/90 | Train: 0.1883/0.9959 | Val: 0.1530/0.9375 | LR: 9.62e-05
        Epoch 79/90 | Train: 0.1896/0.9930 | Val: 0.1537/0.9375 | LR: 9.54e-05
        Epoch 80/90 | Train: 0.2190/0.9916 | Val: 0.1536/0.9375 | LR: 9.46e-05
        Epoch 81/90 | Train: 0.1796/0.9952 | Val: 0.1537/0.9375 | LR: 9.36e-05
        Epoch 82/90 | Train: 0.1798/0.9975 | Val: 0.1534/0.9375 | LR: 9.26e-05
        Epoch 83/90 | Train: 0.1775/0.9976 | Val: 0.1539/0.9375 | LR: 9.16e-05
        Epoch 84/90 | Train: 0.1934/0.9935 | Val: 0.1539/0.9375 | LR: 9.05e-05
        Epoch 85/90 | Train: 0.1842/0.9962 | Val: 0.1543/0.9340 | LR: 8.93e-05
        Epoch 86/90 | Train: 0.1747/0.9981 | Val: 0.1542/0.9375 | LR: 8.80e-05
        Epoch 87/90 | Train: 0.1811/0.9968 | Val: 0.1549/0.9375 | LR: 8.67e-05
        Early stopping at epoch 87 (patience=15)
    
      Best: Epoch 72, Val F1: 0.9375
      EfficientNet-B4 fold 0: val F1 = 0.9375
    
    ============================================================
    FOLD 1 — effnet_b4 (efficientnet_b4)
    Epochs: 90 | Freeze: 5 | Warmup: 3 | CutMix: True
    ============================================================
    
      Total:    17.6M params
      Head:     10.8K params (lr=1e-03)
      Backbone: 17.5M params (lr=1e-04)
      EMA:      decay=0.999
    
      Stage 1: Head-only (5 epochs, no CutMix)
        Epoch 1/5 | Train: 3.1513/0.2743 | Val: 1.7550/0.1649
        Epoch 2/5 | Train: 2.3012/0.4569 | Val: 1.6926/0.1774
        Epoch 3/5 | Train: 1.9123/0.5774 | Val: 1.6110/0.1951
        Epoch 4/5 | Train: 1.6524/0.6320 | Val: 1.5185/0.2181
        Epoch 5/5 | Train: 1.4539/0.6684 | Val: 1.4226/0.2483
    
      Stage 2: Full fine-tuning (85 epochs, CutMix=True)
        Epoch 06/90 | Train: 1.2256/0.7105 | Val: 0.6042/0.7256 | LR: 4.00e-05 ★
        Epoch 07/90 | Train: 0.9122/0.7809 | Val: 0.5895/0.7305 | LR: 7.00e-05 ★
        Epoch 08/90 | Train: 0.6462/0.8468 | Val: 0.5692/0.7317 | LR: 1.00e-04 ★
        Epoch 09/90 | Train: 0.5342/0.8814 | Val: 0.5462/0.7504 | LR: 9.94e-05 ★
        Epoch 10/90 | Train: 0.4236/0.9218 | Val: 0.5223/0.7707 | LR: 9.76e-05 ★
        Epoch 11/90 | Train: 0.4343/0.9306 | Val: 0.5003/0.7804 | LR: 9.46e-05 ★
        Epoch 12/90 | Train: 0.3126/0.9498 | Val: 0.4809/0.7937 | LR: 9.05e-05 ★
        Epoch 13/90 | Train: 0.3479/0.9408 | Val: 0.4634/0.7979 | LR: 8.54e-05 ★
        Epoch 14/90 | Train: 0.3231/0.9548 | Val: 0.4487/0.8083 | LR: 7.94e-05 ★
        Epoch 15/90 | Train: 0.2701/0.9674 | Val: 0.4356/0.8136 | LR: 7.27e-05 ★
        Epoch 16/90 | Train: 0.2745/0.9782 | Val: 0.4248/0.8171 | LR: 6.55e-05 ★
        Epoch 17/90 | Train: 0.2588/0.9750 | Val: 0.4154/0.8193 | LR: 5.79e-05 ★
        Epoch 18/90 | Train: 0.2524/0.9746 | Val: 0.4065/0.8303 | LR: 5.01e-05 ★
        Epoch 19/90 | Train: 0.2606/0.9810 | Val: 0.3978/0.8342 | LR: 4.22e-05 ★
        Epoch 20/90 | Train: 0.2341/0.9793 | Val: 0.3904/0.8385 | LR: 3.46e-05 ★
        Epoch 21/90 | Train: 0.2187/0.9877 | Val: 0.3842/0.8362 | LR: 2.74e-05
        Epoch 22/90 | Train: 0.2200/0.9890 | Val: 0.3784/0.8384 | LR: 2.07e-05
        Epoch 23/90 | Train: 0.2092/0.9901 | Val: 0.3740/0.8416 | LR: 1.47e-05 ★
        Epoch 24/90 | Train: 0.2158/0.9902 | Val: 0.3693/0.8416 | LR: 9.64e-06
        Epoch 25/90 | Train: 0.2103/0.9916 | Val: 0.3656/0.8437 | LR: 5.54e-06 ★
        Epoch 26/90 | Train: 0.2193/0.9865 | Val: 0.3621/0.8524 | LR: 2.54e-06 ★
        Epoch 27/90 | Train: 0.2135/0.9927 | Val: 0.3598/0.8524 | LR: 7.15e-07
        Epoch 28/90 | Train: 0.2163/0.9920 | Val: 0.3577/0.8524 | LR: 1.00e-04
        Epoch 29/90 | Train: 0.2368/0.9855 | Val: 0.3558/0.8572 | LR: 9.98e-05 ★
        Epoch 30/90 | Train: 0.2562/0.9767 | Val: 0.3534/0.8528 | LR: 9.94e-05
        Epoch 31/90 | Train: 0.2293/0.9819 | Val: 0.3509/0.8528 | LR: 9.86e-05
        Epoch 32/90 | Train: 0.2161/0.9880 | Val: 0.3498/0.8596 | LR: 9.76e-05 ★
        Epoch 33/90 | Train: 0.2178/0.9904 | Val: 0.3494/0.8623 | LR: 9.62e-05 ★
        Epoch 34/90 | Train: 0.2311/0.9812 | Val: 0.3472/0.8596 | LR: 9.46e-05
        Epoch 35/90 | Train: 0.2172/0.9906 | Val: 0.3443/0.8596 | LR: 9.26e-05
        Epoch 36/90 | Train: 0.2206/0.9912 | Val: 0.3416/0.8543 | LR: 9.05e-05
        Epoch 37/90 | Train: 0.2134/0.9906 | Val: 0.3390/0.8531 | LR: 8.80e-05
        Epoch 38/90 | Train: 0.2086/0.9881 | Val: 0.3367/0.8556 | LR: 8.54e-05
        Epoch 39/90 | Train: 0.2153/0.9902 | Val: 0.3346/0.8661 | LR: 8.25e-05 ★
        Epoch 40/90 | Train: 0.1949/0.9931 | Val: 0.3321/0.8701 | LR: 7.94e-05 ★
        Epoch 41/90 | Train: 0.2066/0.9894 | Val: 0.3296/0.8701 | LR: 7.61e-05
        Epoch 42/90 | Train: 0.1897/0.9953 | Val: 0.3271/0.8701 | LR: 7.27e-05
        Epoch 43/90 | Train: 0.2075/0.9888 | Val: 0.3265/0.8701 | LR: 6.92e-05
        Epoch 44/90 | Train: 0.1894/0.9955 | Val: 0.3252/0.8726 | LR: 6.55e-05 ★
        Epoch 45/90 | Train: 0.1876/0.9979 | Val: 0.3239/0.8752 | LR: 6.17e-05 ★
        Epoch 46/90 | Train: 0.1900/0.9953 | Val: 0.3231/0.8752 | LR: 5.79e-05
        Epoch 47/90 | Train: 0.1833/0.9965 | Val: 0.3209/0.8752 | LR: 5.40e-05
        Epoch 48/90 | Train: 0.1889/0.9960 | Val: 0.3208/0.8689 | LR: 5.01e-05
        Epoch 49/90 | Train: 0.1923/0.9959 | Val: 0.3196/0.8689 | LR: 4.61e-05
        Epoch 50/90 | Train: 0.1863/0.9982 | Val: 0.3188/0.8726 | LR: 4.22e-05
        Epoch 51/90 | Train: 0.1982/0.9941 | Val: 0.3182/0.8726 | LR: 3.84e-05
        Epoch 52/90 | Train: 0.1722/1.0000 | Val: 0.3176/0.8726 | LR: 3.46e-05
        Epoch 53/90 | Train: 0.1887/0.9971 | Val: 0.3169/0.8725 | LR: 3.09e-05
        Epoch 54/90 | Train: 0.1739/0.9965 | Val: 0.3165/0.8725 | LR: 2.74e-05
        Epoch 55/90 | Train: 0.1801/0.9966 | Val: 0.3160/0.8725 | LR: 2.40e-05
        Epoch 56/90 | Train: 0.1813/0.9957 | Val: 0.3156/0.8700 | LR: 2.07e-05
        Epoch 57/90 | Train: 0.1871/0.9985 | Val: 0.3151/0.8700 | LR: 1.76e-05
        Epoch 58/90 | Train: 0.1821/0.9964 | Val: 0.3158/0.8663 | LR: 1.47e-05
        Epoch 59/90 | Train: 0.1747/0.9960 | Val: 0.3162/0.8627 | LR: 1.21e-05
        Epoch 60/90 | Train: 0.1807/0.9993 | Val: 0.3161/0.8651 | LR: 9.64e-06
        Early stopping at epoch 60 (patience=15)
    
      Best: Epoch 45, Val F1: 0.8752
      EfficientNet-B4 fold 1: val F1 = 0.8752
    
    ============================================================
    FOLD 2 — effnet_b4 (efficientnet_b4)
    Epochs: 90 | Freeze: 5 | Warmup: 3 | CutMix: True
    ============================================================
    
      Total:    17.6M params
      Head:     10.8K params (lr=1e-03)
      Backbone: 17.5M params (lr=1e-04)
      EMA:      decay=0.999
    
      Stage 1: Head-only (5 epochs, no CutMix)
        Epoch 1/5 | Train: 3.1105/0.2672 | Val: 1.6667/0.1788
        Epoch 2/5 | Train: 2.3833/0.4599 | Val: 1.6125/0.2033
        Epoch 3/5 | Train: 1.9666/0.5446 | Val: 1.5390/0.2102
        Epoch 4/5 | Train: 1.6870/0.6421 | Val: 1.4546/0.2436
        Epoch 5/5 | Train: 1.4771/0.6661 | Val: 1.3665/0.2840
    
      Stage 2: Full fine-tuning (85 epochs, CutMix=True)
        Epoch 06/90 | Train: 1.2802/0.7035 | Val: 0.6655/0.7034 | LR: 4.00e-05 ★
        Epoch 07/90 | Train: 0.9160/0.7769 | Val: 0.6505/0.7066 | LR: 7.00e-05 ★
        Epoch 08/90 | Train: 0.6999/0.8306 | Val: 0.6298/0.7198 | LR: 1.00e-04 ★
        Epoch 09/90 | Train: 0.5652/0.8729 | Val: 0.6069/0.7227 | LR: 9.94e-05 ★
        Epoch 10/90 | Train: 0.5061/0.8962 | Val: 0.5838/0.7363 | LR: 9.76e-05 ★
        Epoch 11/90 | Train: 0.4438/0.9171 | Val: 0.5588/0.7441 | LR: 9.46e-05 ★
        Epoch 12/90 | Train: 0.3390/0.9334 | Val: 0.5352/0.7564 | LR: 9.05e-05 ★
        Epoch 13/90 | Train: 0.3255/0.9560 | Val: 0.5121/0.7640 | LR: 8.54e-05 ★
        Epoch 14/90 | Train: 0.2938/0.9654 | Val: 0.4902/0.7811 | LR: 7.94e-05 ★
        Epoch 15/90 | Train: 0.2801/0.9644 | Val: 0.4688/0.7962 | LR: 7.27e-05 ★
        Epoch 16/90 | Train: 0.2833/0.9740 | Val: 0.4510/0.8138 | LR: 6.55e-05 ★
        Epoch 17/90 | Train: 0.2442/0.9799 | Val: 0.4340/0.8207 | LR: 5.79e-05 ★
        Epoch 18/90 | Train: 0.2232/0.9827 | Val: 0.4196/0.8326 | LR: 5.01e-05 ★
        Epoch 19/90 | Train: 0.2464/0.9818 | Val: 0.4062/0.8292 | LR: 4.22e-05
        Epoch 20/90 | Train: 0.2302/0.9902 | Val: 0.3952/0.8343 | LR: 3.46e-05 ★
        Epoch 21/90 | Train: 0.2304/0.9828 | Val: 0.3856/0.8477 | LR: 2.74e-05 ★
        Epoch 22/90 | Train: 0.2110/0.9892 | Val: 0.3754/0.8628 | LR: 2.07e-05 ★
        Epoch 23/90 | Train: 0.2088/0.9894 | Val: 0.3667/0.8628 | LR: 1.47e-05
        Epoch 24/90 | Train: 0.2352/0.9801 | Val: 0.3588/0.8703 | LR: 9.64e-06 ★
        Epoch 25/90 | Train: 0.1990/0.9969 | Val: 0.3527/0.8710 | LR: 5.54e-06 ★
        Epoch 26/90 | Train: 0.2134/0.9942 | Val: 0.3465/0.8702 | LR: 2.54e-06
        Epoch 27/90 | Train: 0.2071/0.9901 | Val: 0.3419/0.8702 | LR: 7.15e-07
        Epoch 28/90 | Train: 0.2344/0.9864 | Val: 0.3379/0.8780 | LR: 1.00e-04 ★
        Epoch 29/90 | Train: 0.2313/0.9844 | Val: 0.3335/0.8777 | LR: 9.98e-05
        Epoch 30/90 | Train: 0.2421/0.9864 | Val: 0.3300/0.8864 | LR: 9.94e-05 ★
        Epoch 31/90 | Train: 0.2498/0.9819 | Val: 0.3280/0.8892 | LR: 9.86e-05 ★
        Epoch 32/90 | Train: 0.2346/0.9834 | Val: 0.3272/0.8864 | LR: 9.76e-05
        Epoch 33/90 | Train: 0.2352/0.9888 | Val: 0.3253/0.8864 | LR: 9.62e-05
        Epoch 34/90 | Train: 0.2213/0.9831 | Val: 0.3229/0.8889 | LR: 9.46e-05
        Epoch 35/90 | Train: 0.2173/0.9901 | Val: 0.3196/0.8884 | LR: 9.26e-05
        Epoch 36/90 | Train: 0.2068/0.9897 | Val: 0.3183/0.8871 | LR: 9.05e-05
        Epoch 37/90 | Train: 0.1992/0.9946 | Val: 0.3169/0.8832 | LR: 8.80e-05
        Epoch 38/90 | Train: 0.2097/0.9906 | Val: 0.3161/0.8832 | LR: 8.54e-05
        Epoch 39/90 | Train: 0.2111/0.9912 | Val: 0.3165/0.8857 | LR: 8.25e-05
        Epoch 40/90 | Train: 0.1925/0.9932 | Val: 0.3163/0.8832 | LR: 7.94e-05
        Epoch 41/90 | Train: 0.1979/0.9936 | Val: 0.3158/0.8832 | LR: 7.61e-05
        Epoch 42/90 | Train: 0.1916/0.9942 | Val: 0.3166/0.8867 | LR: 7.27e-05
        Epoch 43/90 | Train: 0.1816/0.9969 | Val: 0.3158/0.8867 | LR: 6.92e-05
        Epoch 44/90 | Train: 0.2045/0.9957 | Val: 0.3157/0.8892 | LR: 6.55e-05
        Epoch 45/90 | Train: 0.1950/0.9931 | Val: 0.3145/0.8892 | LR: 6.17e-05
        Epoch 46/90 | Train: 0.2033/0.9920 | Val: 0.3134/0.8892 | LR: 5.79e-05
        Early stopping at epoch 46 (patience=15)
    
      Best: Epoch 31, Val F1: 0.8892
      EfficientNet-B4 fold 2: val F1 = 0.8892
    
    ============================================================
    FOLD 3 — effnet_b4 (efficientnet_b4)
    Epochs: 90 | Freeze: 5 | Warmup: 3 | CutMix: True
    ============================================================
    
      Total:    17.6M params
      Head:     10.8K params (lr=1e-03)
      Backbone: 17.5M params (lr=1e-04)
      EMA:      decay=0.999
    
      Stage 1: Head-only (5 epochs, no CutMix)
        Epoch 1/5 | Train: 3.3107/0.2432 | Val: 1.9510/0.1432
        Epoch 2/5 | Train: 2.4818/0.4535 | Val: 1.8872/0.1455
        Epoch 3/5 | Train: 1.9835/0.5680 | Val: 1.7999/0.1488
        Epoch 4/5 | Train: 1.6921/0.6169 | Val: 1.7001/0.1787
        Epoch 5/5 | Train: 1.4456/0.6641 | Val: 1.5934/0.2047
    
      Stage 2: Full fine-tuning (85 epochs, CutMix=True)
        Epoch 06/90 | Train: 1.2962/0.7075 | Val: 0.6739/0.6885 | LR: 4.00e-05 ★
        Epoch 07/90 | Train: 0.9146/0.7657 | Val: 0.6621/0.6907 | LR: 7.00e-05 ★
        Epoch 08/90 | Train: 0.7082/0.8270 | Val: 0.6464/0.7016 | LR: 1.00e-04 ★
        Epoch 09/90 | Train: 0.5164/0.8741 | Val: 0.6291/0.7065 | LR: 9.94e-05 ★
        Epoch 10/90 | Train: 0.4165/0.9078 | Val: 0.6093/0.7141 | LR: 9.76e-05 ★
        Epoch 11/90 | Train: 0.3674/0.9347 | Val: 0.5905/0.7282 | LR: 9.46e-05 ★
        Epoch 12/90 | Train: 0.3460/0.9378 | Val: 0.5739/0.7287 | LR: 9.05e-05 ★
        Epoch 13/90 | Train: 0.2948/0.9574 | Val: 0.5597/0.7357 | LR: 8.54e-05 ★
        Epoch 14/90 | Train: 0.2969/0.9636 | Val: 0.5455/0.7449 | LR: 7.94e-05 ★
        Epoch 15/90 | Train: 0.2702/0.9721 | Val: 0.5332/0.7489 | LR: 7.27e-05 ★
        Epoch 16/90 | Train: 0.2774/0.9680 | Val: 0.5208/0.7488 | LR: 6.55e-05
        Epoch 17/90 | Train: 0.2603/0.9755 | Val: 0.5078/0.7585 | LR: 5.79e-05 ★
        Epoch 18/90 | Train: 0.2323/0.9798 | Val: 0.4968/0.7620 | LR: 5.01e-05 ★
        Epoch 19/90 | Train: 0.2542/0.9808 | Val: 0.4862/0.7549 | LR: 4.22e-05
        Epoch 20/90 | Train: 0.2357/0.9870 | Val: 0.4782/0.7582 | LR: 3.46e-05
        Epoch 21/90 | Train: 0.2131/0.9870 | Val: 0.4714/0.7620 | LR: 2.74e-05
        Epoch 22/90 | Train: 0.2183/0.9920 | Val: 0.4637/0.7648 | LR: 2.07e-05 ★
        Epoch 23/90 | Train: 0.2240/0.9860 | Val: 0.4563/0.7648 | LR: 1.47e-05
        Epoch 24/90 | Train: 0.2243/0.9896 | Val: 0.4488/0.7680 | LR: 9.64e-06 ★
        Epoch 25/90 | Train: 0.2036/0.9900 | Val: 0.4430/0.7709 | LR: 5.54e-06 ★
        Epoch 26/90 | Train: 0.2061/0.9918 | Val: 0.4375/0.7805 | LR: 2.54e-06 ★
        Epoch 27/90 | Train: 0.2113/0.9914 | Val: 0.4325/0.7832 | LR: 7.15e-07 ★
        Epoch 28/90 | Train: 0.2071/0.9899 | Val: 0.4276/0.7854 | LR: 1.00e-04 ★
        Epoch 29/90 | Train: 0.2036/0.9892 | Val: 0.4237/0.7953 | LR: 9.98e-05 ★
        Epoch 30/90 | Train: 0.2373/0.9851 | Val: 0.4193/0.7925 | LR: 9.94e-05
        Epoch 31/90 | Train: 0.2155/0.9858 | Val: 0.4154/0.7994 | LR: 9.86e-05 ★
        Epoch 32/90 | Train: 0.2238/0.9900 | Val: 0.4118/0.7994 | LR: 9.76e-05
        Epoch 33/90 | Train: 0.2331/0.9858 | Val: 0.4077/0.7967 | LR: 9.62e-05
        Epoch 34/90 | Train: 0.2342/0.9889 | Val: 0.4051/0.7982 | LR: 9.46e-05
        Epoch 35/90 | Train: 0.2277/0.9859 | Val: 0.4026/0.8010 | LR: 9.26e-05 ★
        Epoch 36/90 | Train: 0.2165/0.9827 | Val: 0.3967/0.8037 | LR: 9.05e-05 ★
        Epoch 37/90 | Train: 0.2090/0.9929 | Val: 0.3922/0.8106 | LR: 8.80e-05 ★
        Epoch 38/90 | Train: 0.1996/0.9929 | Val: 0.3887/0.8121 | LR: 8.54e-05 ★
        Epoch 39/90 | Train: 0.2066/0.9903 | Val: 0.3849/0.8121 | LR: 8.25e-05
        Epoch 40/90 | Train: 0.2043/0.9925 | Val: 0.3815/0.8121 | LR: 7.94e-05
        Epoch 41/90 | Train: 0.2202/0.9859 | Val: 0.3785/0.8121 | LR: 7.61e-05
        Epoch 42/90 | Train: 0.2030/0.9951 | Val: 0.3747/0.8158 | LR: 7.27e-05 ★
        Epoch 43/90 | Train: 0.2117/0.9906 | Val: 0.3706/0.8127 | LR: 6.92e-05
        Epoch 44/90 | Train: 0.1956/0.9966 | Val: 0.3677/0.8180 | LR: 6.55e-05 ★
        Epoch 45/90 | Train: 0.2069/0.9908 | Val: 0.3642/0.8180 | LR: 6.17e-05
        Epoch 46/90 | Train: 0.1973/0.9901 | Val: 0.3616/0.8214 | LR: 5.79e-05 ★
        Epoch 47/90 | Train: 0.1840/0.9963 | Val: 0.3582/0.8232 | LR: 5.40e-05 ★
        Epoch 48/90 | Train: 0.1886/0.9952 | Val: 0.3553/0.8232 | LR: 5.01e-05
        Epoch 49/90 | Train: 0.1850/0.9939 | Val: 0.3520/0.8232 | LR: 4.61e-05
        Epoch 50/90 | Train: 0.1902/0.9976 | Val: 0.3494/0.8275 | LR: 4.22e-05 ★
        Epoch 51/90 | Train: 0.1912/0.9951 | Val: 0.3484/0.8275 | LR: 3.84e-05
        Epoch 52/90 | Train: 0.1839/0.9950 | Val: 0.3475/0.8298 | LR: 3.46e-05 ★
        Epoch 53/90 | Train: 0.1895/0.9953 | Val: 0.3450/0.8298 | LR: 3.09e-05
        Epoch 54/90 | Train: 0.1851/0.9981 | Val: 0.3431/0.8347 | LR: 2.74e-05 ★
        Epoch 55/90 | Train: 0.1793/0.9991 | Val: 0.3422/0.8280 | LR: 2.40e-05
        Epoch 56/90 | Train: 0.1803/0.9983 | Val: 0.3402/0.8280 | LR: 2.07e-05
        Epoch 57/90 | Train: 0.1752/0.9985 | Val: 0.3393/0.8303 | LR: 1.76e-05
        Epoch 58/90 | Train: 0.1860/0.9971 | Val: 0.3381/0.8352 | LR: 1.47e-05 ★
        Epoch 59/90 | Train: 0.1721/0.9975 | Val: 0.3376/0.8352 | LR: 1.21e-05
        Epoch 60/90 | Train: 0.1835/0.9955 | Val: 0.3370/0.8352 | LR: 9.64e-06
        Epoch 61/90 | Train: 0.1808/0.9959 | Val: 0.3371/0.8352 | LR: 7.46e-06
        Epoch 62/90 | Train: 0.1818/0.9952 | Val: 0.3369/0.8315 | LR: 5.54e-06
        Epoch 63/90 | Train: 0.1813/0.9973 | Val: 0.3369/0.8315 | LR: 3.90e-06
        Epoch 64/90 | Train: 0.1957/0.9951 | Val: 0.3369/0.8386 | LR: 2.54e-06 ★
        Epoch 65/90 | Train: 0.1769/0.9963 | Val: 0.3368/0.8386 | LR: 1.48e-06
        Epoch 66/90 | Train: 0.1888/0.9952 | Val: 0.3363/0.8354 | LR: 7.15e-07
        Epoch 67/90 | Train: 0.1722/0.9961 | Val: 0.3360/0.8342 | LR: 2.54e-07
        Epoch 68/90 | Train: 0.1809/0.9987 | Val: 0.3357/0.8342 | LR: 1.00e-04
        Epoch 69/90 | Train: 0.1869/0.9963 | Val: 0.3356/0.8342 | LR: 1.00e-04
        Epoch 70/90 | Train: 0.1809/0.9966 | Val: 0.3353/0.8342 | LR: 9.98e-05
        Epoch 71/90 | Train: 0.1857/0.9986 | Val: 0.3364/0.8342 | LR: 9.97e-05
        Epoch 72/90 | Train: 0.2137/0.9899 | Val: 0.3379/0.8396 | LR: 9.94e-05 ★
        Epoch 73/90 | Train: 0.1962/0.9987 | Val: 0.3402/0.8342 | LR: 9.90e-05
        Epoch 74/90 | Train: 0.1916/0.9973 | Val: 0.3422/0.8342 | LR: 9.86e-05
        Epoch 75/90 | Train: 0.1950/0.9947 | Val: 0.3430/0.8342 | LR: 9.81e-05
        Epoch 76/90 | Train: 0.1955/0.9932 | Val: 0.3441/0.8359 | LR: 9.76e-05
        Epoch 77/90 | Train: 0.1992/0.9893 | Val: 0.3443/0.8359 | LR: 9.69e-05
        Epoch 78/90 | Train: 0.2055/0.9933 | Val: 0.3454/0.8379 | LR: 9.62e-05
        Epoch 79/90 | Train: 0.1958/0.9926 | Val: 0.3469/0.8379 | LR: 9.54e-05
        Epoch 80/90 | Train: 0.1923/0.9898 | Val: 0.3481/0.8379 | LR: 9.46e-05
        Epoch 81/90 | Train: 0.1883/0.9957 | Val: 0.3485/0.8402 | LR: 9.36e-05 ★
        Epoch 82/90 | Train: 0.1763/0.9975 | Val: 0.3482/0.8382 | LR: 9.26e-05
        Epoch 83/90 | Train: 0.1861/0.9967 | Val: 0.3490/0.8382 | LR: 9.16e-05
        Epoch 84/90 | Train: 0.1845/0.9965 | Val: 0.3496/0.8366 | LR: 9.05e-05
        Epoch 85/90 | Train: 0.1884/0.9947 | Val: 0.3501/0.8362 | LR: 8.93e-05
        Epoch 86/90 | Train: 0.1739/0.9968 | Val: 0.3501/0.8386 | LR: 8.80e-05
        Epoch 87/90 | Train: 0.1803/0.9985 | Val: 0.3501/0.8420 | LR: 8.67e-05 ★
        Epoch 88/90 | Train: 0.1783/0.9979 | Val: 0.3490/0.8420 | LR: 8.54e-05
        Epoch 89/90 | Train: 0.1760/0.9981 | Val: 0.3480/0.8448 | LR: 8.40e-05 ★
        Epoch 90/90 | Train: 0.1847/0.9966 | Val: 0.3470/0.8448 | LR: 8.25e-05
    
      Best: Epoch 89, Val F1: 0.8448
      EfficientNet-B4 fold 3: val F1 = 0.8448
    
    ============================================================
    FOLD 4 — effnet_b4 (efficientnet_b4)
    Epochs: 90 | Freeze: 5 | Warmup: 3 | CutMix: True
    ============================================================
    
      Total:    17.6M params
      Head:     10.8K params (lr=1e-03)
      Backbone: 17.5M params (lr=1e-04)
      EMA:      decay=0.999
    
      Stage 1: Head-only (5 epochs, no CutMix)
        Epoch 1/5 | Train: 3.3418/0.2478 | Val: 1.7487/0.1366
        Epoch 2/5 | Train: 2.4554/0.4364 | Val: 1.6956/0.1429
        Epoch 3/5 | Train: 2.0301/0.5608 | Val: 1.6207/0.1607
        Epoch 4/5 | Train: 1.7132/0.6095 | Val: 1.5346/0.2085
        Epoch 5/5 | Train: 1.5627/0.6569 | Val: 1.4427/0.2578
    
      Stage 2: Full fine-tuning (85 epochs, CutMix=True)
        Epoch 06/90 | Train: 1.2836/0.6994 | Val: 0.6546/0.6655 | LR: 4.00e-05 ★
        Epoch 07/90 | Train: 0.9236/0.7732 | Val: 0.6405/0.6778 | LR: 7.00e-05 ★
        Epoch 08/90 | Train: 0.6644/0.8410 | Val: 0.6194/0.6803 | LR: 1.00e-04 ★
        Epoch 09/90 | Train: 0.5761/0.8736 | Val: 0.5958/0.6859 | LR: 9.94e-05 ★
        Epoch 10/90 | Train: 0.4423/0.9095 | Val: 0.5688/0.7181 | LR: 9.76e-05 ★
        Epoch 11/90 | Train: 0.3818/0.9371 | Val: 0.5442/0.7390 | LR: 9.46e-05 ★
        Epoch 12/90 | Train: 0.3355/0.9531 | Val: 0.5222/0.7503 | LR: 9.05e-05 ★
        Epoch 13/90 | Train: 0.3168/0.9502 | Val: 0.5000/0.7587 | LR: 8.54e-05 ★
        Epoch 14/90 | Train: 0.2916/0.9591 | Val: 0.4793/0.7738 | LR: 7.94e-05 ★
        Epoch 15/90 | Train: 0.2768/0.9745 | Val: 0.4621/0.7785 | LR: 7.27e-05 ★
        Epoch 16/90 | Train: 0.2438/0.9729 | Val: 0.4473/0.7813 | LR: 6.55e-05 ★
        Epoch 17/90 | Train: 0.2454/0.9783 | Val: 0.4352/0.7834 | LR: 5.79e-05 ★
        Epoch 18/90 | Train: 0.2245/0.9792 | Val: 0.4238/0.7960 | LR: 5.01e-05 ★
        Epoch 19/90 | Train: 0.2141/0.9904 | Val: 0.4135/0.7994 | LR: 4.22e-05 ★
        Epoch 20/90 | Train: 0.2379/0.9819 | Val: 0.4046/0.7994 | LR: 3.46e-05
        Epoch 21/90 | Train: 0.2336/0.9807 | Val: 0.3973/0.8098 | LR: 2.74e-05 ★
        Epoch 22/90 | Train: 0.2134/0.9904 | Val: 0.3907/0.8252 | LR: 2.07e-05 ★
        Epoch 23/90 | Train: 0.2384/0.9903 | Val: 0.3845/0.8252 | LR: 1.47e-05
        Epoch 24/90 | Train: 0.2072/0.9896 | Val: 0.3792/0.8290 | LR: 9.64e-06 ★
        Epoch 25/90 | Train: 0.2019/0.9934 | Val: 0.3754/0.8345 | LR: 5.54e-06 ★
        Epoch 26/90 | Train: 0.2123/0.9837 | Val: 0.3717/0.8365 | LR: 2.54e-06 ★
        Epoch 27/90 | Train: 0.2115/0.9900 | Val: 0.3684/0.8365 | LR: 7.15e-07
        Epoch 28/90 | Train: 0.2233/0.9851 | Val: 0.3650/0.8365 | LR: 1.00e-04
        Epoch 29/90 | Train: 0.2284/0.9852 | Val: 0.3629/0.8389 | LR: 9.98e-05 ★
        Epoch 30/90 | Train: 0.2509/0.9818 | Val: 0.3611/0.8415 | LR: 9.94e-05 ★
        Epoch 31/90 | Train: 0.2345/0.9780 | Val: 0.3598/0.8415 | LR: 9.86e-05
        Epoch 32/90 | Train: 0.2020/0.9879 | Val: 0.3577/0.8469 | LR: 9.76e-05 ★
        Epoch 33/90 | Train: 0.2282/0.9793 | Val: 0.3554/0.8499 | LR: 9.62e-05 ★
        Epoch 34/90 | Train: 0.2227/0.9873 | Val: 0.3520/0.8503 | LR: 9.46e-05 ★
        Epoch 35/90 | Train: 0.2035/0.9918 | Val: 0.3492/0.8518 | LR: 9.26e-05 ★
        Epoch 36/90 | Train: 0.2062/0.9903 | Val: 0.3462/0.8601 | LR: 9.05e-05 ★
        Epoch 37/90 | Train: 0.2042/0.9958 | Val: 0.3443/0.8611 | LR: 8.80e-05 ★
        Epoch 38/90 | Train: 0.2253/0.9889 | Val: 0.3430/0.8611 | LR: 8.54e-05
        Epoch 39/90 | Train: 0.2042/0.9884 | Val: 0.3419/0.8611 | LR: 8.25e-05
        Epoch 40/90 | Train: 0.1921/0.9923 | Val: 0.3396/0.8661 | LR: 7.94e-05 ★
        Epoch 41/90 | Train: 0.2051/0.9926 | Val: 0.3370/0.8661 | LR: 7.61e-05
        Epoch 42/90 | Train: 0.2047/0.9923 | Val: 0.3363/0.8661 | LR: 7.27e-05
        Epoch 43/90 | Train: 0.1891/0.9975 | Val: 0.3341/0.8661 | LR: 6.92e-05
        Epoch 44/90 | Train: 0.1978/0.9927 | Val: 0.3320/0.8661 | LR: 6.55e-05
        Epoch 45/90 | Train: 0.1946/0.9944 | Val: 0.3296/0.8637 | LR: 6.17e-05
        Epoch 46/90 | Train: 0.1764/0.9982 | Val: 0.3282/0.8661 | LR: 5.79e-05
        Epoch 47/90 | Train: 0.1881/0.9933 | Val: 0.3270/0.8651 | LR: 5.40e-05
        Epoch 48/90 | Train: 0.1846/0.9974 | Val: 0.3256/0.8614 | LR: 5.01e-05
        Epoch 49/90 | Train: 0.1920/0.9925 | Val: 0.3239/0.8614 | LR: 4.61e-05
        Epoch 50/90 | Train: 0.1791/0.9969 | Val: 0.3228/0.8614 | LR: 4.22e-05
        Epoch 51/90 | Train: 0.1895/0.9976 | Val: 0.3209/0.8614 | LR: 3.84e-05
        Epoch 52/90 | Train: 0.1861/0.9970 | Val: 0.3199/0.8614 | LR: 3.46e-05
        Epoch 53/90 | Train: 0.1863/0.9981 | Val: 0.3189/0.8614 | LR: 3.09e-05
        Epoch 54/90 | Train: 0.1775/0.9993 | Val: 0.3177/0.8614 | LR: 2.74e-05
        Epoch 55/90 | Train: 0.1815/0.9964 | Val: 0.3166/0.8614 | LR: 2.40e-05
        Early stopping at epoch 55 (patience=15)
    
      Best: Epoch 40, Val F1: 0.8661
      EfficientNet-B4 fold 4: val F1 = 0.8661
    
    EfficientNet-B4 summary: mean=0.8826 ± 0.0310
      exp06 baseline: mean=0.8902 ± 0.0309


## 10 · Train SwinV2-Base  (70 epochs)

Note: SwinV2 fold 4 hit the epoch ceiling in exp06 (best_epoch=60 out of 60 total).
The +10 epochs here directly addresses that.


```python
swinv2_results = {}

for fold in range(N_FOLDS):
    result = train_fold(
        fold         = fold,
        model_key    = 'swinv2',
        train_df     = train_df,
        class_weights= class_weights,
        use_cutmix   = True,
        use_sampler  = False,
        exp_id       = EXP_ID,
    )
    swinv2_results[fold] = result
    print(f'  SwinV2 fold {fold}: val F1 = {result["best_f1"]:.4f}')

scores = [r['best_f1'] for r in swinv2_results.values()]
print(f'\nSwinV2 summary: mean={np.mean(scores):.4f} ± {np.std(scores):.4f}')
print(f'  exp06 baseline: mean=0.9089 ± 0.0348')
```

    
    ============================================================
    FOLD 0 — swinv2 (swinv2_base_window12_192.ms_in22k)
    Epochs: 90 | Freeze: 10 | Warmup: 5 | CutMix: True
    ============================================================



    model.safetensors:   0%|          | 0.00/439M [00:00<?, ?B/s]


    
      Total:    86.9M params
      Head:     6.2K params (lr=5e-04)
      Backbone: 86.9M params (lr=2e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (10 epochs, no CutMix)
        Epoch 1/10 | Train: 2.5034/0.4391 | Val: 1.7143/0.1081
        Epoch 2/10 | Train: 1.3534/0.6998 | Val: 1.6547/0.1276
        Epoch 3/10 | Train: 1.0315/0.7464 | Val: 1.5794/0.1381
        Epoch 4/10 | Train: 0.9416/0.7840 | Val: 1.4977/0.1722
        Epoch 5/10 | Train: 0.8261/0.7976 | Val: 1.4141/0.2170
        Epoch 6/10 | Train: 0.8681/0.7955 | Val: 1.3303/0.2650
        Epoch 7/10 | Train: 0.7725/0.8231 | Val: 1.2478/0.3111
        Epoch 8/10 | Train: 0.7450/0.8204 | Val: 1.1670/0.3693
        Epoch 9/10 | Train: 0.6667/0.8267 | Val: 1.0908/0.4154
        Epoch 10/10 | Train: 0.6479/0.8478 | Val: 1.0181/0.4540
    
      Stage 2: Full fine-tuning (80 epochs, CutMix=True)
        Epoch 11/90 | Train: 0.6229/0.8452 | Val: 0.2331/0.8843 | LR: 1.40e-04 ★
        Epoch 12/90 | Train: 0.5786/0.8574 | Val: 0.2321/0.8843 | LR: 2.30e-04
        Epoch 13/90 | Train: 0.5852/0.8665 | Val: 0.2311/0.8843 | LR: 3.20e-04
        Epoch 14/90 | Train: 0.5452/0.8725 | Val: 0.2293/0.8908 | LR: 4.10e-04 ★
        Epoch 15/90 | Train: 0.4892/0.8876 | Val: 0.2280/0.8908 | LR: 5.00e-04
        Epoch 16/90 | Train: 0.4426/0.9125 | Val: 0.2262/0.8908 | LR: 4.97e-04
        Epoch 17/90 | Train: 0.3815/0.9242 | Val: 0.2234/0.8897 | LR: 4.88e-04
        Epoch 18/90 | Train: 0.4022/0.9306 | Val: 0.2207/0.8992 | LR: 4.73e-04 ★
        Epoch 19/90 | Train: 0.3854/0.9226 | Val: 0.2183/0.8992 | LR: 4.52e-04
        Epoch 20/90 | Train: 0.3669/0.9434 | Val: 0.2152/0.8992 | LR: 4.27e-04
        Epoch 21/90 | Train: 0.3145/0.9542 | Val: 0.2118/0.8992 | LR: 3.97e-04
        Epoch 22/90 | Train: 0.3184/0.9516 | Val: 0.2089/0.9041 | LR: 3.64e-04 ★
        Epoch 23/90 | Train: 0.2986/0.9628 | Val: 0.2054/0.9041 | LR: 3.27e-04
        Epoch 24/90 | Train: 0.2725/0.9744 | Val: 0.2020/0.9041 | LR: 2.89e-04
        Epoch 25/90 | Train: 0.2804/0.9688 | Val: 0.1990/0.9041 | LR: 2.50e-04
        Epoch 26/90 | Train: 0.2607/0.9737 | Val: 0.1955/0.9041 | LR: 2.11e-04
        Epoch 27/90 | Train: 0.2441/0.9741 | Val: 0.1917/0.9041 | LR: 1.73e-04
        Epoch 28/90 | Train: 0.2633/0.9704 | Val: 0.1887/0.9041 | LR: 1.37e-04
        Epoch 29/90 | Train: 0.2563/0.9777 | Val: 0.1858/0.9076 | LR: 1.03e-04 ★
        Epoch 30/90 | Train: 0.2430/0.9724 | Val: 0.1821/0.9076 | LR: 7.33e-05
        Epoch 31/90 | Train: 0.2340/0.9817 | Val: 0.1792/0.9076 | LR: 4.78e-05
        Epoch 32/90 | Train: 0.2190/0.9901 | Val: 0.1764/0.9076 | LR: 2.73e-05
        Epoch 33/90 | Train: 0.2331/0.9864 | Val: 0.1732/0.9076 | LR: 1.23e-05
        Epoch 34/90 | Train: 0.2327/0.9795 | Val: 0.1704/0.9076 | LR: 3.18e-06
        Epoch 35/90 | Train: 0.2235/0.9879 | Val: 0.1679/0.9115 | LR: 5.00e-04 ★
        Epoch 36/90 | Train: 0.2489/0.9803 | Val: 0.1660/0.9115 | LR: 4.99e-04
        Epoch 37/90 | Train: 0.2470/0.9755 | Val: 0.1643/0.9115 | LR: 4.97e-04
        Epoch 38/90 | Train: 0.2705/0.9684 | Val: 0.1618/0.9115 | LR: 4.93e-04
        Epoch 39/90 | Train: 0.2880/0.9675 | Val: 0.1591/0.9115 | LR: 4.88e-04
        Epoch 40/90 | Train: 0.2463/0.9766 | Val: 0.1563/0.9206 | LR: 4.81e-04 ★
        Epoch 41/90 | Train: 0.2382/0.9793 | Val: 0.1536/0.9272 | LR: 4.73e-04 ★
        Epoch 42/90 | Train: 0.2581/0.9726 | Val: 0.1516/0.9272 | LR: 4.63e-04
        Epoch 43/90 | Train: 0.2592/0.9693 | Val: 0.1503/0.9272 | LR: 4.52e-04
        Epoch 44/90 | Train: 0.2469/0.9758 | Val: 0.1489/0.9237 | LR: 4.40e-04
        Epoch 45/90 | Train: 0.2266/0.9858 | Val: 0.1474/0.9237 | LR: 4.27e-04
        Epoch 46/90 | Train: 0.2186/0.9887 | Val: 0.1463/0.9237 | LR: 4.12e-04
        Epoch 47/90 | Train: 0.2231/0.9875 | Val: 0.1453/0.9237 | LR: 3.97e-04
        Epoch 48/90 | Train: 0.2233/0.9865 | Val: 0.1445/0.9237 | LR: 3.81e-04
        Epoch 49/90 | Train: 0.2239/0.9865 | Val: 0.1436/0.9237 | LR: 3.64e-04
        Epoch 50/90 | Train: 0.2174/0.9877 | Val: 0.1430/0.9325 | LR: 3.46e-04 ★
        Epoch 51/90 | Train: 0.2088/0.9897 | Val: 0.1425/0.9325 | LR: 3.27e-04
        Epoch 52/90 | Train: 0.2124/0.9876 | Val: 0.1416/0.9325 | LR: 3.08e-04
        Epoch 53/90 | Train: 0.2179/0.9860 | Val: 0.1411/0.9325 | LR: 2.89e-04
        Epoch 54/90 | Train: 0.2055/0.9912 | Val: 0.1406/0.9325 | LR: 2.70e-04
        Epoch 55/90 | Train: 0.2065/0.9902 | Val: 0.1401/0.9325 | LR: 2.50e-04
        Epoch 56/90 | Train: 0.2126/0.9909 | Val: 0.1396/0.9325 | LR: 2.30e-04
        Epoch 57/90 | Train: 0.2038/0.9908 | Val: 0.1395/0.9325 | LR: 2.11e-04
        Epoch 58/90 | Train: 0.2091/0.9867 | Val: 0.1392/0.9325 | LR: 1.92e-04
        Epoch 59/90 | Train: 0.2019/0.9911 | Val: 0.1391/0.9325 | LR: 1.73e-04
        Epoch 60/90 | Train: 0.1984/0.9907 | Val: 0.1384/0.9325 | LR: 1.54e-04
        Epoch 61/90 | Train: 0.1980/0.9918 | Val: 0.1381/0.9325 | LR: 1.37e-04
        Epoch 62/90 | Train: 0.1917/0.9937 | Val: 0.1373/0.9325 | LR: 1.19e-04
        Epoch 63/90 | Train: 0.1970/0.9920 | Val: 0.1374/0.9325 | LR: 1.03e-04
        Epoch 64/90 | Train: 0.1949/0.9940 | Val: 0.1377/0.9325 | LR: 8.77e-05
        Epoch 65/90 | Train: 0.1881/0.9944 | Val: 0.1368/0.9325 | LR: 7.33e-05
        Early stopping at epoch 65 (patience=15)
    
      Best: Epoch 50, Val F1: 0.9325
      SwinV2 fold 0: val F1 = 0.9325
    
    ============================================================
    FOLD 1 — swinv2 (swinv2_base_window12_192.ms_in22k)
    Epochs: 90 | Freeze: 10 | Warmup: 5 | CutMix: True
    ============================================================
    
      Total:    86.9M params
      Head:     6.2K params (lr=5e-04)
      Backbone: 86.9M params (lr=2e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (10 epochs, no CutMix)
        Epoch 1/10 | Train: 2.6548/0.4156 | Val: 1.6640/0.1513
        Epoch 2/10 | Train: 1.3964/0.7019 | Val: 1.6080/0.1643
        Epoch 3/10 | Train: 1.0835/0.7685 | Val: 1.5364/0.1822
        Epoch 4/10 | Train: 0.8928/0.7791 | Val: 1.4582/0.1840
        Epoch 5/10 | Train: 0.8014/0.8049 | Val: 1.3784/0.2142
        Epoch 6/10 | Train: 0.7108/0.8342 | Val: 1.2990/0.2667
        Epoch 7/10 | Train: 0.6513/0.8479 | Val: 1.2210/0.3101
        Epoch 8/10 | Train: 0.7247/0.8328 | Val: 1.1455/0.3698
        Epoch 9/10 | Train: 0.6827/0.8484 | Val: 1.0748/0.4208
        Epoch 10/10 | Train: 0.6365/0.8446 | Val: 1.0076/0.4639
    
      Stage 2: Full fine-tuning (80 epochs, CutMix=True)
        Epoch 11/90 | Train: 0.5960/0.8594 | Val: 0.3082/0.8417 | LR: 1.40e-04 ★
        Epoch 12/90 | Train: 0.5374/0.8665 | Val: 0.3079/0.8417 | LR: 2.30e-04
        Epoch 13/90 | Train: 0.5399/0.8810 | Val: 0.3066/0.8417 | LR: 3.20e-04
        Epoch 14/90 | Train: 0.4456/0.8938 | Val: 0.3050/0.8451 | LR: 4.10e-04 ★
        Epoch 15/90 | Train: 0.4322/0.9087 | Val: 0.3046/0.8497 | LR: 5.00e-04 ★
        Epoch 16/90 | Train: 0.5244/0.9062 | Val: 0.3021/0.8497 | LR: 4.97e-04
        Epoch 17/90 | Train: 0.4196/0.9105 | Val: 0.2998/0.8544 | LR: 4.88e-04 ★
        Epoch 18/90 | Train: 0.3736/0.9298 | Val: 0.2972/0.8489 | LR: 4.73e-04
        Epoch 19/90 | Train: 0.3242/0.9503 | Val: 0.2952/0.8489 | LR: 4.52e-04
        Epoch 20/90 | Train: 0.3155/0.9501 | Val: 0.2931/0.8493 | LR: 4.27e-04
        Epoch 21/90 | Train: 0.3084/0.9576 | Val: 0.2916/0.8498 | LR: 3.97e-04
        Epoch 22/90 | Train: 0.2980/0.9668 | Val: 0.2902/0.8547 | LR: 3.64e-04 ★
        Epoch 23/90 | Train: 0.3070/0.9631 | Val: 0.2881/0.8547 | LR: 3.27e-04
        Epoch 24/90 | Train: 0.2853/0.9621 | Val: 0.2852/0.8547 | LR: 2.89e-04
        Epoch 25/90 | Train: 0.2601/0.9715 | Val: 0.2832/0.8571 | LR: 2.50e-04 ★
        Epoch 26/90 | Train: 0.2438/0.9738 | Val: 0.2813/0.8596 | LR: 2.11e-04 ★
        Epoch 27/90 | Train: 0.2459/0.9851 | Val: 0.2798/0.8650 | LR: 1.73e-04 ★
        Epoch 28/90 | Train: 0.2374/0.9832 | Val: 0.2786/0.8691 | LR: 1.37e-04 ★
        Epoch 29/90 | Train: 0.2208/0.9868 | Val: 0.2774/0.8716 | LR: 1.03e-04 ★
        Epoch 30/90 | Train: 0.2356/0.9872 | Val: 0.2768/0.8716 | LR: 7.33e-05
        Epoch 31/90 | Train: 0.2257/0.9875 | Val: 0.2756/0.8745 | LR: 4.78e-05 ★
        Epoch 32/90 | Train: 0.2270/0.9827 | Val: 0.2747/0.8816 | LR: 2.73e-05 ★
        Epoch 33/90 | Train: 0.2327/0.9864 | Val: 0.2737/0.8816 | LR: 1.23e-05
        Epoch 34/90 | Train: 0.2243/0.9884 | Val: 0.2727/0.8816 | LR: 3.18e-06
        Epoch 35/90 | Train: 0.2224/0.9853 | Val: 0.2717/0.8855 | LR: 5.00e-04 ★
        Epoch 36/90 | Train: 0.2400/0.9819 | Val: 0.2710/0.8855 | LR: 4.99e-04
        Epoch 37/90 | Train: 0.2621/0.9798 | Val: 0.2700/0.8855 | LR: 4.97e-04
        Epoch 38/90 | Train: 0.2536/0.9778 | Val: 0.2696/0.8816 | LR: 4.93e-04
        Epoch 39/90 | Train: 0.2219/0.9859 | Val: 0.2683/0.8867 | LR: 4.88e-04 ★
        Epoch 40/90 | Train: 0.2444/0.9799 | Val: 0.2672/0.8867 | LR: 4.81e-04
        Epoch 41/90 | Train: 0.2502/0.9745 | Val: 0.2664/0.8867 | LR: 4.73e-04
        Epoch 42/90 | Train: 0.2414/0.9824 | Val: 0.2658/0.8867 | LR: 4.63e-04
        Epoch 43/90 | Train: 0.2264/0.9845 | Val: 0.2657/0.8827 | LR: 4.52e-04
        Epoch 44/90 | Train: 0.2662/0.9830 | Val: 0.2656/0.8797 | LR: 4.40e-04
        Epoch 45/90 | Train: 0.2259/0.9865 | Val: 0.2657/0.8797 | LR: 4.27e-04
        Epoch 46/90 | Train: 0.2443/0.9800 | Val: 0.2663/0.8797 | LR: 4.12e-04
        Epoch 47/90 | Train: 0.2171/0.9920 | Val: 0.2670/0.8836 | LR: 3.97e-04
        Epoch 48/90 | Train: 0.2230/0.9897 | Val: 0.2671/0.8836 | LR: 3.81e-04
        Epoch 49/90 | Train: 0.2025/0.9901 | Val: 0.2672/0.8836 | LR: 3.64e-04
        Epoch 50/90 | Train: 0.2025/0.9926 | Val: 0.2671/0.8836 | LR: 3.46e-04
        Epoch 51/90 | Train: 0.2001/0.9927 | Val: 0.2673/0.8836 | LR: 3.27e-04
        Epoch 52/90 | Train: 0.2193/0.9915 | Val: 0.2674/0.8836 | LR: 3.08e-04
        Epoch 53/90 | Train: 0.2035/0.9884 | Val: 0.2678/0.8859 | LR: 2.89e-04
        Epoch 54/90 | Train: 0.2112/0.9907 | Val: 0.2681/0.8885 | LR: 2.70e-04 ★
        Epoch 55/90 | Train: 0.2177/0.9894 | Val: 0.2675/0.8885 | LR: 2.50e-04
        Epoch 56/90 | Train: 0.1978/0.9925 | Val: 0.2673/0.8915 | LR: 2.30e-04 ★
        Epoch 57/90 | Train: 0.1936/0.9964 | Val: 0.2667/0.8915 | LR: 2.11e-04
        Epoch 58/90 | Train: 0.2031/0.9890 | Val: 0.2662/0.8915 | LR: 1.92e-04
        Epoch 59/90 | Train: 0.1987/0.9953 | Val: 0.2659/0.8915 | LR: 1.73e-04
        Epoch 60/90 | Train: 0.1883/0.9956 | Val: 0.2652/0.8915 | LR: 1.54e-04
        Epoch 61/90 | Train: 0.1957/0.9933 | Val: 0.2647/0.8915 | LR: 1.37e-04
        Epoch 62/90 | Train: 0.1877/0.9938 | Val: 0.2642/0.8915 | LR: 1.19e-04
        Epoch 63/90 | Train: 0.1968/0.9947 | Val: 0.2635/0.8915 | LR: 1.03e-04
        Epoch 64/90 | Train: 0.1944/0.9972 | Val: 0.2627/0.8915 | LR: 8.77e-05
        Epoch 65/90 | Train: 0.1976/0.9928 | Val: 0.2617/0.8915 | LR: 7.33e-05
        Epoch 66/90 | Train: 0.1918/0.9959 | Val: 0.2608/0.8980 | LR: 6.00e-05 ★
        Epoch 67/90 | Train: 0.1857/0.9955 | Val: 0.2602/0.8980 | LR: 4.78e-05
        Epoch 68/90 | Train: 0.1769/0.9972 | Val: 0.2594/0.8980 | LR: 3.69e-05
        Epoch 69/90 | Train: 0.1848/0.9955 | Val: 0.2584/0.8979 | LR: 2.73e-05
        Epoch 70/90 | Train: 0.1843/0.9970 | Val: 0.2576/0.8979 | LR: 1.91e-05
        Epoch 71/90 | Train: 0.1870/0.9957 | Val: 0.2566/0.8979 | LR: 1.23e-05
        Epoch 72/90 | Train: 0.1818/0.9982 | Val: 0.2561/0.8979 | LR: 7.01e-06
        Epoch 73/90 | Train: 0.1929/0.9951 | Val: 0.2555/0.8979 | LR: 3.18e-06
        Epoch 74/90 | Train: 0.1963/0.9939 | Val: 0.2550/0.8979 | LR: 8.71e-07
        Epoch 75/90 | Train: 0.1854/0.9958 | Val: 0.2540/0.8980 | LR: 5.00e-04
        Epoch 76/90 | Train: 0.1885/0.9964 | Val: 0.2536/0.8980 | LR: 5.00e-04
        Epoch 77/90 | Train: 0.1901/0.9959 | Val: 0.2526/0.8980 | LR: 4.99e-04
        Epoch 78/90 | Train: 0.2048/0.9885 | Val: 0.2516/0.8980 | LR: 4.98e-04
        Epoch 79/90 | Train: 0.2077/0.9881 | Val: 0.2513/0.8980 | LR: 4.97e-04
        Epoch 80/90 | Train: 0.2146/0.9941 | Val: 0.2513/0.8980 | LR: 4.95e-04
        Epoch 81/90 | Train: 0.2158/0.9894 | Val: 0.2513/0.8980 | LR: 4.93e-04
        Early stopping at epoch 81 (patience=15)
    
      Best: Epoch 66, Val F1: 0.8980
      SwinV2 fold 1: val F1 = 0.8980
    
    ============================================================
    FOLD 2 — swinv2 (swinv2_base_window12_192.ms_in22k)
    Epochs: 90 | Freeze: 10 | Warmup: 5 | CutMix: True
    ============================================================
    
      Total:    86.9M params
      Head:     6.2K params (lr=5e-04)
      Backbone: 86.9M params (lr=2e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (10 epochs, no CutMix)
        Epoch 1/10 | Train: 2.5409/0.4149 | Val: 1.6450/0.1593
        Epoch 2/10 | Train: 1.3622/0.6812 | Val: 1.5915/0.1677
        Epoch 3/10 | Train: 1.0278/0.7619 | Val: 1.5231/0.1892
        Epoch 4/10 | Train: 0.9883/0.7805 | Val: 1.4486/0.2109
        Epoch 5/10 | Train: 0.8539/0.8177 | Val: 1.3729/0.2379
        Epoch 6/10 | Train: 0.7866/0.8017 | Val: 1.2974/0.2706
        Epoch 7/10 | Train: 0.7418/0.8123 | Val: 1.2242/0.2960
        Epoch 8/10 | Train: 0.7548/0.8132 | Val: 1.1531/0.3379
        Epoch 9/10 | Train: 0.6584/0.8413 | Val: 1.0833/0.3844
        Epoch 10/10 | Train: 0.6099/0.8471 | Val: 1.0167/0.4342
    
      Stage 2: Full fine-tuning (80 epochs, CutMix=True)
        Epoch 11/90 | Train: 0.6325/0.8571 | Val: 0.2868/0.8348 | LR: 1.40e-04 ★
        Epoch 12/90 | Train: 0.5620/0.8649 | Val: 0.2862/0.8348 | LR: 2.30e-04
        Epoch 13/90 | Train: 0.5421/0.8718 | Val: 0.2847/0.8383 | LR: 3.20e-04 ★
        Epoch 14/90 | Train: 0.4778/0.8853 | Val: 0.2839/0.8383 | LR: 4.10e-04
        Epoch 15/90 | Train: 0.4473/0.9064 | Val: 0.2830/0.8383 | LR: 5.00e-04
        Epoch 16/90 | Train: 0.4171/0.9181 | Val: 0.2805/0.8417 | LR: 4.97e-04 ★
        Epoch 17/90 | Train: 0.3763/0.9241 | Val: 0.2774/0.8417 | LR: 4.88e-04
        Epoch 18/90 | Train: 0.3715/0.9263 | Val: 0.2747/0.8495 | LR: 4.73e-04 ★
        Epoch 19/90 | Train: 0.3769/0.9289 | Val: 0.2711/0.8495 | LR: 4.52e-04
        Epoch 20/90 | Train: 0.3321/0.9481 | Val: 0.2680/0.8485 | LR: 4.27e-04
        Epoch 21/90 | Train: 0.2940/0.9599 | Val: 0.2658/0.8520 | LR: 3.97e-04 ★
        Epoch 22/90 | Train: 0.2814/0.9611 | Val: 0.2621/0.8520 | LR: 3.64e-04
        Epoch 23/90 | Train: 0.2625/0.9661 | Val: 0.2587/0.8516 | LR: 3.27e-04
        Epoch 24/90 | Train: 0.2758/0.9707 | Val: 0.2551/0.8584 | LR: 2.89e-04 ★
        Epoch 25/90 | Train: 0.2587/0.9728 | Val: 0.2512/0.8611 | LR: 2.50e-04 ★
        Epoch 26/90 | Train: 0.2817/0.9674 | Val: 0.2483/0.8635 | LR: 2.11e-04 ★
        Epoch 27/90 | Train: 0.2515/0.9753 | Val: 0.2452/0.8635 | LR: 1.73e-04
        Epoch 28/90 | Train: 0.2614/0.9734 | Val: 0.2422/0.8646 | LR: 1.37e-04 ★
        Epoch 29/90 | Train: 0.2388/0.9807 | Val: 0.2391/0.8714 | LR: 1.03e-04 ★
        Epoch 30/90 | Train: 0.2425/0.9874 | Val: 0.2357/0.8761 | LR: 7.33e-05 ★
        Epoch 31/90 | Train: 0.2320/0.9859 | Val: 0.2329/0.8761 | LR: 4.78e-05
        Epoch 32/90 | Train: 0.2285/0.9820 | Val: 0.2305/0.8798 | LR: 2.73e-05 ★
        Epoch 33/90 | Train: 0.2166/0.9912 | Val: 0.2286/0.8798 | LR: 1.23e-05
        Epoch 34/90 | Train: 0.2292/0.9855 | Val: 0.2270/0.8798 | LR: 3.18e-06
        Epoch 35/90 | Train: 0.2200/0.9894 | Val: 0.2254/0.8823 | LR: 5.00e-04 ★
        Epoch 36/90 | Train: 0.2656/0.9742 | Val: 0.2232/0.8823 | LR: 4.99e-04
        Epoch 37/90 | Train: 0.2793/0.9694 | Val: 0.2211/0.8850 | LR: 4.97e-04 ★
        Epoch 38/90 | Train: 0.2818/0.9634 | Val: 0.2198/0.8888 | LR: 4.93e-04 ★
        Epoch 39/90 | Train: 0.2578/0.9763 | Val: 0.2184/0.8888 | LR: 4.88e-04
        Epoch 40/90 | Train: 0.2428/0.9769 | Val: 0.2177/0.8888 | LR: 4.81e-04
        Epoch 41/90 | Train: 0.2455/0.9819 | Val: 0.2162/0.8888 | LR: 4.73e-04
        Epoch 42/90 | Train: 0.2442/0.9784 | Val: 0.2146/0.8888 | LR: 4.63e-04
        Epoch 43/90 | Train: 0.2345/0.9783 | Val: 0.2133/0.8927 | LR: 4.52e-04 ★
        Epoch 44/90 | Train: 0.2493/0.9742 | Val: 0.2118/0.8936 | LR: 4.40e-04 ★
        Epoch 45/90 | Train: 0.2756/0.9726 | Val: 0.2104/0.8936 | LR: 4.27e-04
        Epoch 46/90 | Train: 0.2449/0.9795 | Val: 0.2094/0.8992 | LR: 4.12e-04 ★
        Epoch 47/90 | Train: 0.2340/0.9882 | Val: 0.2085/0.8982 | LR: 3.97e-04
        Epoch 48/90 | Train: 0.2277/0.9813 | Val: 0.2076/0.8982 | LR: 3.81e-04
        Epoch 49/90 | Train: 0.2139/0.9893 | Val: 0.2068/0.8982 | LR: 3.64e-04
        Epoch 50/90 | Train: 0.2321/0.9887 | Val: 0.2062/0.8982 | LR: 3.46e-04
        Epoch 51/90 | Train: 0.2125/0.9893 | Val: 0.2051/0.8982 | LR: 3.27e-04
        Epoch 52/90 | Train: 0.2086/0.9913 | Val: 0.2045/0.9008 | LR: 3.08e-04 ★
        Epoch 53/90 | Train: 0.2182/0.9878 | Val: 0.2039/0.9008 | LR: 2.89e-04
        Epoch 54/90 | Train: 0.2251/0.9893 | Val: 0.2038/0.9008 | LR: 2.70e-04
        Epoch 55/90 | Train: 0.2111/0.9899 | Val: 0.2032/0.9034 | LR: 2.50e-04 ★
        Epoch 56/90 | Train: 0.2123/0.9936 | Val: 0.2030/0.9034 | LR: 2.30e-04
        Epoch 57/90 | Train: 0.2074/0.9927 | Val: 0.2038/0.9025 | LR: 2.11e-04
        Epoch 58/90 | Train: 0.2078/0.9917 | Val: 0.2038/0.9025 | LR: 1.92e-04
        Epoch 59/90 | Train: 0.2028/0.9925 | Val: 0.2039/0.9025 | LR: 1.73e-04
        Epoch 60/90 | Train: 0.1870/0.9932 | Val: 0.2030/0.9025 | LR: 1.54e-04
        Epoch 61/90 | Train: 0.1996/0.9937 | Val: 0.2024/0.9050 | LR: 1.37e-04 ★
        Epoch 62/90 | Train: 0.1941/0.9950 | Val: 0.2019/0.9050 | LR: 1.19e-04
        Epoch 63/90 | Train: 0.1957/0.9959 | Val: 0.2019/0.9050 | LR: 1.03e-04
        Epoch 64/90 | Train: 0.1922/0.9960 | Val: 0.2020/0.9050 | LR: 8.77e-05
        Epoch 65/90 | Train: 0.2033/0.9928 | Val: 0.2030/0.9050 | LR: 7.33e-05
        Epoch 66/90 | Train: 0.1929/0.9930 | Val: 0.2041/0.9050 | LR: 6.00e-05
        Epoch 67/90 | Train: 0.1861/0.9982 | Val: 0.2040/0.9050 | LR: 4.78e-05
        Epoch 68/90 | Train: 0.1872/0.9947 | Val: 0.2039/0.9050 | LR: 3.69e-05
        Epoch 69/90 | Train: 0.1991/0.9974 | Val: 0.2037/0.9050 | LR: 2.73e-05
        Epoch 70/90 | Train: 0.1878/0.9949 | Val: 0.2038/0.9050 | LR: 1.91e-05
        Epoch 71/90 | Train: 0.1973/0.9945 | Val: 0.2037/0.9050 | LR: 1.23e-05
        Epoch 72/90 | Train: 0.1936/0.9957 | Val: 0.2038/0.9050 | LR: 7.01e-06
        Epoch 73/90 | Train: 0.1912/0.9943 | Val: 0.2043/0.9050 | LR: 3.18e-06
        Epoch 74/90 | Train: 0.1932/0.9978 | Val: 0.2042/0.9076 | LR: 8.71e-07 ★
        Epoch 75/90 | Train: 0.1876/0.9933 | Val: 0.2042/0.9076 | LR: 5.00e-04
        Epoch 76/90 | Train: 0.2086/0.9909 | Val: 0.2040/0.9076 | LR: 5.00e-04
        Epoch 77/90 | Train: 0.2152/0.9938 | Val: 0.2038/0.9076 | LR: 4.99e-04
        Epoch 78/90 | Train: 0.2114/0.9913 | Val: 0.2048/0.9076 | LR: 4.98e-04
        Epoch 79/90 | Train: 0.2111/0.9910 | Val: 0.2046/0.9085 | LR: 4.97e-04 ★
        Epoch 80/90 | Train: 0.2094/0.9914 | Val: 0.2062/0.9085 | LR: 4.95e-04
        Epoch 81/90 | Train: 0.2076/0.9904 | Val: 0.2065/0.9085 | LR: 4.93e-04
        Epoch 82/90 | Train: 0.2118/0.9872 | Val: 0.2066/0.9085 | LR: 4.91e-04
        Epoch 83/90 | Train: 0.2186/0.9888 | Val: 0.2065/0.9085 | LR: 4.88e-04
        Epoch 84/90 | Train: 0.2143/0.9941 | Val: 0.2067/0.9085 | LR: 4.85e-04
        Epoch 85/90 | Train: 0.2134/0.9885 | Val: 0.2074/0.9085 | LR: 4.81e-04
        Epoch 86/90 | Train: 0.2099/0.9903 | Val: 0.2082/0.9008 | LR: 4.77e-04
        Epoch 87/90 | Train: 0.1944/0.9928 | Val: 0.2096/0.9008 | LR: 4.73e-04
        Epoch 88/90 | Train: 0.1944/0.9952 | Val: 0.2105/0.9000 | LR: 4.68e-04
        Epoch 89/90 | Train: 0.1917/0.9896 | Val: 0.2116/0.9000 | LR: 4.63e-04
        Epoch 90/90 | Train: 0.2040/0.9901 | Val: 0.2126/0.9000 | LR: 4.58e-04
    
      Best: Epoch 79, Val F1: 0.9085
      SwinV2 fold 2: val F1 = 0.9085
    
    ============================================================
    FOLD 3 — swinv2 (swinv2_base_window12_192.ms_in22k)
    Epochs: 90 | Freeze: 10 | Warmup: 5 | CutMix: True
    ============================================================
    
      Total:    86.9M params
      Head:     6.2K params (lr=5e-04)
      Backbone: 86.9M params (lr=2e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (10 epochs, no CutMix)
        Epoch 1/10 | Train: 2.6804/0.4041 | Val: 1.7473/0.1700
        Epoch 2/10 | Train: 1.4078/0.6897 | Val: 1.6896/0.1784
        Epoch 3/10 | Train: 1.0555/0.7654 | Val: 1.6196/0.1887
        Epoch 4/10 | Train: 0.8921/0.7923 | Val: 1.5456/0.2098
        Epoch 5/10 | Train: 0.8517/0.7936 | Val: 1.4693/0.2369
        Epoch 6/10 | Train: 0.7802/0.8126 | Val: 1.3936/0.2802
        Epoch 7/10 | Train: 0.6997/0.8313 | Val: 1.3198/0.2979
        Epoch 8/10 | Train: 0.7075/0.8258 | Val: 1.2486/0.3411
        Epoch 9/10 | Train: 0.6693/0.8283 | Val: 1.1800/0.3567
        Epoch 10/10 | Train: 0.6541/0.8450 | Val: 1.1151/0.3854
    
      Stage 2: Full fine-tuning (80 epochs, CutMix=True)
        Epoch 11/90 | Train: 0.6044/0.8494 | Val: 0.3652/0.7897 | LR: 1.40e-04 ★
        Epoch 12/90 | Train: 0.5359/0.8885 | Val: 0.3645/0.7897 | LR: 2.30e-04
        Epoch 13/90 | Train: 0.5227/0.8799 | Val: 0.3641/0.7897 | LR: 3.20e-04
        Epoch 14/90 | Train: 0.4687/0.9047 | Val: 0.3627/0.7903 | LR: 4.10e-04 ★
        Epoch 15/90 | Train: 0.4550/0.9121 | Val: 0.3603/0.7897 | LR: 5.00e-04
        Epoch 16/90 | Train: 0.4839/0.9005 | Val: 0.3584/0.7903 | LR: 4.97e-04
        Epoch 17/90 | Train: 0.4118/0.9333 | Val: 0.3561/0.7943 | LR: 4.88e-04 ★
        Epoch 18/90 | Train: 0.3943/0.9312 | Val: 0.3541/0.7943 | LR: 4.73e-04
        Epoch 19/90 | Train: 0.3335/0.9476 | Val: 0.3516/0.7943 | LR: 4.52e-04
        Epoch 20/90 | Train: 0.3207/0.9530 | Val: 0.3495/0.7954 | LR: 4.27e-04 ★
        Epoch 21/90 | Train: 0.3087/0.9570 | Val: 0.3470/0.7954 | LR: 3.97e-04
        Epoch 22/90 | Train: 0.3093/0.9550 | Val: 0.3441/0.7954 | LR: 3.64e-04
        Epoch 23/90 | Train: 0.2833/0.9693 | Val: 0.3421/0.7954 | LR: 3.27e-04
        Epoch 24/90 | Train: 0.2812/0.9724 | Val: 0.3394/0.7989 | LR: 2.89e-04 ★
        Epoch 25/90 | Train: 0.2852/0.9654 | Val: 0.3360/0.7995 | LR: 2.50e-04 ★
        Epoch 26/90 | Train: 0.2406/0.9773 | Val: 0.3338/0.7995 | LR: 2.11e-04
        Epoch 27/90 | Train: 0.2528/0.9772 | Val: 0.3310/0.8007 | LR: 1.73e-04 ★
        Epoch 28/90 | Train: 0.2381/0.9796 | Val: 0.3281/0.8007 | LR: 1.37e-04
        Epoch 29/90 | Train: 0.2197/0.9806 | Val: 0.3256/0.7995 | LR: 1.03e-04
        Epoch 30/90 | Train: 0.2301/0.9904 | Val: 0.3231/0.8030 | LR: 7.33e-05 ★
        Epoch 31/90 | Train: 0.2246/0.9800 | Val: 0.3209/0.8058 | LR: 4.78e-05 ★
        Epoch 32/90 | Train: 0.2149/0.9884 | Val: 0.3193/0.8123 | LR: 2.73e-05 ★
        Epoch 33/90 | Train: 0.2250/0.9852 | Val: 0.3177/0.8151 | LR: 1.23e-05 ★
        Epoch 34/90 | Train: 0.2183/0.9888 | Val: 0.3165/0.8191 | LR: 3.18e-06 ★
        Epoch 35/90 | Train: 0.2253/0.9876 | Val: 0.3152/0.8212 | LR: 5.00e-04 ★
        Epoch 36/90 | Train: 0.2572/0.9786 | Val: 0.3135/0.8239 | LR: 4.99e-04 ★
        Epoch 37/90 | Train: 0.2485/0.9820 | Val: 0.3119/0.8236 | LR: 4.97e-04
        Epoch 38/90 | Train: 0.2634/0.9806 | Val: 0.3106/0.8344 | LR: 4.93e-04 ★
        Epoch 39/90 | Train: 0.2284/0.9767 | Val: 0.3100/0.8344 | LR: 4.88e-04
        Epoch 40/90 | Train: 0.2301/0.9832 | Val: 0.3087/0.8344 | LR: 4.81e-04
        Epoch 41/90 | Train: 0.2404/0.9851 | Val: 0.3075/0.8373 | LR: 4.73e-04 ★
        Epoch 42/90 | Train: 0.2320/0.9829 | Val: 0.3069/0.8426 | LR: 4.63e-04 ★
        Epoch 43/90 | Train: 0.2492/0.9783 | Val: 0.3056/0.8426 | LR: 4.52e-04
        Epoch 44/90 | Train: 0.2263/0.9849 | Val: 0.3039/0.8426 | LR: 4.40e-04
        Epoch 45/90 | Train: 0.2360/0.9880 | Val: 0.3031/0.8426 | LR: 4.27e-04
        Epoch 46/90 | Train: 0.2381/0.9823 | Val: 0.3012/0.8497 | LR: 4.12e-04 ★
        Epoch 47/90 | Train: 0.2211/0.9886 | Val: 0.3000/0.8497 | LR: 3.97e-04
        Epoch 48/90 | Train: 0.2289/0.9854 | Val: 0.2985/0.8497 | LR: 3.81e-04
        Epoch 49/90 | Train: 0.2302/0.9888 | Val: 0.2974/0.8497 | LR: 3.64e-04
        Epoch 50/90 | Train: 0.2322/0.9874 | Val: 0.2967/0.8497 | LR: 3.46e-04
        Epoch 51/90 | Train: 0.2092/0.9875 | Val: 0.2959/0.8554 | LR: 3.27e-04 ★
        Epoch 52/90 | Train: 0.2022/0.9913 | Val: 0.2958/0.8581 | LR: 3.08e-04 ★
        Epoch 53/90 | Train: 0.2081/0.9896 | Val: 0.2958/0.8581 | LR: 2.89e-04
        Epoch 54/90 | Train: 0.2019/0.9942 | Val: 0.2955/0.8633 | LR: 2.70e-04 ★
        Epoch 55/90 | Train: 0.2148/0.9927 | Val: 0.2953/0.8633 | LR: 2.50e-04
        Epoch 56/90 | Train: 0.1954/0.9940 | Val: 0.2949/0.8606 | LR: 2.30e-04
        Epoch 57/90 | Train: 0.2043/0.9937 | Val: 0.2943/0.8606 | LR: 2.11e-04
        Epoch 58/90 | Train: 0.1954/0.9925 | Val: 0.2937/0.8606 | LR: 1.92e-04
        Epoch 59/90 | Train: 0.2006/0.9926 | Val: 0.2929/0.8633 | LR: 1.73e-04
        Epoch 60/90 | Train: 0.1998/0.9919 | Val: 0.2918/0.8688 | LR: 1.54e-04 ★
        Epoch 61/90 | Train: 0.2147/0.9912 | Val: 0.2910/0.8688 | LR: 1.37e-04
        Epoch 62/90 | Train: 0.1974/0.9944 | Val: 0.2901/0.8688 | LR: 1.19e-04
        Epoch 63/90 | Train: 0.1930/0.9973 | Val: 0.2897/0.8688 | LR: 1.03e-04
        Epoch 64/90 | Train: 0.1949/0.9934 | Val: 0.2890/0.8715 | LR: 8.77e-05 ★
        Epoch 65/90 | Train: 0.1856/0.9956 | Val: 0.2885/0.8742 | LR: 7.33e-05 ★
        Epoch 66/90 | Train: 0.1970/0.9930 | Val: 0.2879/0.8742 | LR: 6.00e-05
        Epoch 67/90 | Train: 0.1948/0.9929 | Val: 0.2868/0.8742 | LR: 4.78e-05
        Epoch 68/90 | Train: 0.1860/0.9984 | Val: 0.2862/0.8742 | LR: 3.69e-05
        Epoch 69/90 | Train: 0.1833/0.9982 | Val: 0.2855/0.8736 | LR: 2.73e-05
        Epoch 70/90 | Train: 0.2017/0.9946 | Val: 0.2853/0.8736 | LR: 1.91e-05
        Epoch 71/90 | Train: 0.1883/0.9944 | Val: 0.2841/0.8736 | LR: 1.23e-05
        Epoch 72/90 | Train: 0.1895/0.9952 | Val: 0.2837/0.8736 | LR: 7.01e-06
        Epoch 73/90 | Train: 0.1894/0.9953 | Val: 0.2831/0.8736 | LR: 3.18e-06
        Epoch 74/90 | Train: 0.1889/0.9993 | Val: 0.2828/0.8736 | LR: 8.71e-07
        Epoch 75/90 | Train: 0.1843/0.9952 | Val: 0.2821/0.8736 | LR: 5.00e-04
        Epoch 76/90 | Train: 0.1991/0.9930 | Val: 0.2815/0.8736 | LR: 5.00e-04
        Epoch 77/90 | Train: 0.2149/0.9872 | Val: 0.2805/0.8736 | LR: 4.99e-04
        Epoch 78/90 | Train: 0.2085/0.9963 | Val: 0.2794/0.8736 | LR: 4.98e-04
        Epoch 79/90 | Train: 0.1997/0.9887 | Val: 0.2783/0.8804 | LR: 4.97e-04 ★
        Epoch 80/90 | Train: 0.2215/0.9901 | Val: 0.2775/0.8804 | LR: 4.95e-04
        Epoch 81/90 | Train: 0.2008/0.9934 | Val: 0.2778/0.8804 | LR: 4.93e-04
        Epoch 82/90 | Train: 0.2138/0.9886 | Val: 0.2777/0.8838 | LR: 4.91e-04 ★
        Epoch 83/90 | Train: 0.1926/0.9927 | Val: 0.2779/0.8838 | LR: 4.88e-04
        Epoch 84/90 | Train: 0.1978/0.9913 | Val: 0.2776/0.8838 | LR: 4.85e-04
        Epoch 85/90 | Train: 0.2142/0.9820 | Val: 0.2769/0.8838 | LR: 4.81e-04
        Epoch 86/90 | Train: 0.2267/0.9855 | Val: 0.2762/0.8838 | LR: 4.77e-04
        Epoch 87/90 | Train: 0.2086/0.9866 | Val: 0.2761/0.8838 | LR: 4.73e-04
        Epoch 88/90 | Train: 0.1997/0.9915 | Val: 0.2760/0.8838 | LR: 4.68e-04
        Epoch 89/90 | Train: 0.1924/0.9971 | Val: 0.2754/0.8838 | LR: 4.63e-04
        Epoch 90/90 | Train: 0.1897/0.9945 | Val: 0.2749/0.8838 | LR: 4.58e-04
    
      Best: Epoch 82, Val F1: 0.8838
      SwinV2 fold 3: val F1 = 0.8838
    
    ============================================================
    FOLD 4 — swinv2 (swinv2_base_window12_192.ms_in22k)
    Epochs: 90 | Freeze: 10 | Warmup: 5 | CutMix: True
    ============================================================
    
      Total:    86.9M params
      Head:     6.2K params (lr=5e-04)
      Backbone: 86.9M params (lr=2e-05)
      EMA:      decay=0.999
    
      Stage 1: Head-only (10 epochs, no CutMix)
        Epoch 1/10 | Train: 2.3438/0.4804 | Val: 1.5646/0.2067
        Epoch 2/10 | Train: 1.3200/0.7104 | Val: 1.5122/0.2308
        Epoch 3/10 | Train: 0.9614/0.7701 | Val: 1.4471/0.2650
        Epoch 4/10 | Train: 0.8725/0.8007 | Val: 1.3770/0.2889
        Epoch 5/10 | Train: 0.8340/0.7971 | Val: 1.3056/0.3420
        Epoch 6/10 | Train: 0.7423/0.8032 | Val: 1.2357/0.3783
        Epoch 7/10 | Train: 0.6846/0.8274 | Val: 1.1664/0.4029
        Epoch 8/10 | Train: 0.6870/0.8286 | Val: 1.0990/0.4304
        Epoch 9/10 | Train: 0.6934/0.8331 | Val: 1.0353/0.4669
        Epoch 10/10 | Train: 0.6990/0.8409 | Val: 0.9754/0.4888
    
      Stage 2: Full fine-tuning (80 epochs, CutMix=True)
        Epoch 11/90 | Train: 0.6384/0.8403 | Val: 0.2654/0.8408 | LR: 1.40e-04 ★
        Epoch 12/90 | Train: 0.5571/0.8689 | Val: 0.2648/0.8408 | LR: 2.30e-04
        Epoch 13/90 | Train: 0.5268/0.8786 | Val: 0.2640/0.8408 | LR: 3.20e-04
        Epoch 14/90 | Train: 0.4725/0.8869 | Val: 0.2626/0.8408 | LR: 4.10e-04
        Epoch 15/90 | Train: 0.4570/0.9063 | Val: 0.2610/0.8408 | LR: 5.00e-04
        Epoch 16/90 | Train: 0.4405/0.9005 | Val: 0.2591/0.8408 | LR: 4.97e-04
        Epoch 17/90 | Train: 0.3725/0.9321 | Val: 0.2575/0.8408 | LR: 4.88e-04
        Epoch 18/90 | Train: 0.3563/0.9324 | Val: 0.2558/0.8408 | LR: 4.73e-04
        Epoch 19/90 | Train: 0.3534/0.9357 | Val: 0.2533/0.8428 | LR: 4.52e-04 ★
        Epoch 20/90 | Train: 0.3358/0.9485 | Val: 0.2512/0.8455 | LR: 4.27e-04 ★
        Epoch 21/90 | Train: 0.3210/0.9463 | Val: 0.2491/0.8502 | LR: 3.97e-04 ★
        Epoch 22/90 | Train: 0.3439/0.9576 | Val: 0.2465/0.8502 | LR: 3.64e-04
        Epoch 23/90 | Train: 0.2792/0.9708 | Val: 0.2434/0.8475 | LR: 3.27e-04
        Epoch 24/90 | Train: 0.2610/0.9724 | Val: 0.2408/0.8475 | LR: 2.89e-04
        Epoch 25/90 | Train: 0.2660/0.9714 | Val: 0.2381/0.8478 | LR: 2.50e-04
        Epoch 26/90 | Train: 0.2581/0.9701 | Val: 0.2360/0.8478 | LR: 2.11e-04
        Epoch 27/90 | Train: 0.2352/0.9860 | Val: 0.2332/0.8537 | LR: 1.73e-04 ★
        Epoch 28/90 | Train: 0.2610/0.9740 | Val: 0.2298/0.8573 | LR: 1.37e-04 ★
        Epoch 29/90 | Train: 0.2343/0.9816 | Val: 0.2276/0.8573 | LR: 1.03e-04
        Epoch 30/90 | Train: 0.2432/0.9703 | Val: 0.2249/0.8573 | LR: 7.33e-05
        Epoch 31/90 | Train: 0.2279/0.9843 | Val: 0.2223/0.8573 | LR: 4.78e-05
        Epoch 32/90 | Train: 0.2336/0.9850 | Val: 0.2191/0.8549 | LR: 2.73e-05
        Epoch 33/90 | Train: 0.2358/0.9837 | Val: 0.2169/0.8612 | LR: 1.23e-05 ★
        Epoch 34/90 | Train: 0.2219/0.9895 | Val: 0.2144/0.8663 | LR: 3.18e-06 ★
        Epoch 35/90 | Train: 0.2289/0.9856 | Val: 0.2122/0.8723 | LR: 5.00e-04 ★
        Epoch 36/90 | Train: 0.2346/0.9787 | Val: 0.2100/0.8768 | LR: 4.99e-04 ★
        Epoch 37/90 | Train: 0.2543/0.9752 | Val: 0.2081/0.8806 | LR: 4.97e-04 ★
        Epoch 38/90 | Train: 0.2563/0.9729 | Val: 0.2067/0.8833 | LR: 4.93e-04 ★
        Epoch 39/90 | Train: 0.2547/0.9781 | Val: 0.2049/0.8833 | LR: 4.88e-04
        Epoch 40/90 | Train: 0.2587/0.9765 | Val: 0.2022/0.8894 | LR: 4.81e-04 ★
        Epoch 41/90 | Train: 0.2422/0.9815 | Val: 0.1998/0.8894 | LR: 4.73e-04
        Epoch 42/90 | Train: 0.2519/0.9806 | Val: 0.1970/0.8892 | LR: 4.63e-04
        Epoch 43/90 | Train: 0.2778/0.9809 | Val: 0.1950/0.8892 | LR: 4.52e-04
        Epoch 44/90 | Train: 0.2446/0.9806 | Val: 0.1932/0.8935 | LR: 4.40e-04 ★
        Epoch 45/90 | Train: 0.2358/0.9839 | Val: 0.1916/0.8962 | LR: 4.27e-04 ★
        Epoch 46/90 | Train: 0.2463/0.9837 | Val: 0.1897/0.8999 | LR: 4.12e-04 ★
        Epoch 47/90 | Train: 0.2217/0.9865 | Val: 0.1882/0.8999 | LR: 3.97e-04
        Epoch 48/90 | Train: 0.2188/0.9837 | Val: 0.1872/0.8999 | LR: 3.81e-04
        Epoch 49/90 | Train: 0.2245/0.9877 | Val: 0.1853/0.8999 | LR: 3.64e-04
        Epoch 50/90 | Train: 0.2253/0.9887 | Val: 0.1842/0.8999 | LR: 3.46e-04
        Epoch 51/90 | Train: 0.2278/0.9853 | Val: 0.1827/0.9070 | LR: 3.27e-04 ★
        Epoch 52/90 | Train: 0.2162/0.9900 | Val: 0.1814/0.9043 | LR: 3.08e-04
        Epoch 53/90 | Train: 0.2075/0.9899 | Val: 0.1799/0.9095 | LR: 2.89e-04 ★
        Epoch 54/90 | Train: 0.2187/0.9929 | Val: 0.1782/0.9095 | LR: 2.70e-04
        Epoch 55/90 | Train: 0.1971/0.9913 | Val: 0.1763/0.9169 | LR: 2.50e-04 ★
        Epoch 56/90 | Train: 0.2076/0.9906 | Val: 0.1750/0.9285 | LR: 2.30e-04 ★
        Epoch 57/90 | Train: 0.2049/0.9937 | Val: 0.1737/0.9334 | LR: 2.11e-04 ★
        Epoch 58/90 | Train: 0.2047/0.9940 | Val: 0.1726/0.9334 | LR: 1.92e-04
        Epoch 59/90 | Train: 0.1992/0.9944 | Val: 0.1712/0.9371 | LR: 1.73e-04 ★
        Epoch 60/90 | Train: 0.1941/0.9941 | Val: 0.1701/0.9371 | LR: 1.54e-04
        Epoch 61/90 | Train: 0.1946/0.9926 | Val: 0.1687/0.9371 | LR: 1.37e-04
        Epoch 62/90 | Train: 0.2138/0.9954 | Val: 0.1673/0.9371 | LR: 1.19e-04
        Epoch 63/90 | Train: 0.1965/0.9923 | Val: 0.1661/0.9371 | LR: 1.03e-04
        Epoch 64/90 | Train: 0.1889/0.9946 | Val: 0.1653/0.9371 | LR: 8.77e-05
        Epoch 65/90 | Train: 0.1858/0.9945 | Val: 0.1643/0.9371 | LR: 7.33e-05
        Epoch 66/90 | Train: 0.1921/0.9959 | Val: 0.1632/0.9371 | LR: 6.00e-05
        Epoch 67/90 | Train: 0.1889/0.9934 | Val: 0.1624/0.9403 | LR: 4.78e-05 ★
        Epoch 68/90 | Train: 0.1781/0.9983 | Val: 0.1616/0.9403 | LR: 3.69e-05
        Epoch 69/90 | Train: 0.1976/0.9955 | Val: 0.1608/0.9403 | LR: 2.73e-05
        Epoch 70/90 | Train: 0.1964/0.9940 | Val: 0.1599/0.9403 | LR: 1.91e-05
        Epoch 71/90 | Train: 0.1933/0.9980 | Val: 0.1594/0.9403 | LR: 1.23e-05
        Epoch 72/90 | Train: 0.1870/0.9941 | Val: 0.1588/0.9403 | LR: 7.01e-06
        Epoch 73/90 | Train: 0.1831/0.9959 | Val: 0.1581/0.9403 | LR: 3.18e-06
        Epoch 74/90 | Train: 0.1889/0.9973 | Val: 0.1576/0.9428 | LR: 8.71e-07 ★
        Epoch 75/90 | Train: 0.1959/0.9963 | Val: 0.1566/0.9428 | LR: 5.00e-04
        Epoch 76/90 | Train: 0.2176/0.9904 | Val: 0.1565/0.9428 | LR: 5.00e-04
        Epoch 77/90 | Train: 0.1953/0.9944 | Val: 0.1565/0.9428 | LR: 4.99e-04
        Epoch 78/90 | Train: 0.1994/0.9916 | Val: 0.1560/0.9395 | LR: 4.98e-04
        Epoch 79/90 | Train: 0.2173/0.9816 | Val: 0.1561/0.9395 | LR: 4.97e-04
        Epoch 80/90 | Train: 0.2250/0.9831 | Val: 0.1563/0.9395 | LR: 4.95e-04
        Epoch 81/90 | Train: 0.2089/0.9863 | Val: 0.1568/0.9395 | LR: 4.93e-04
        Epoch 82/90 | Train: 0.2104/0.9936 | Val: 0.1566/0.9395 | LR: 4.91e-04
        Epoch 83/90 | Train: 0.2006/0.9953 | Val: 0.1565/0.9395 | LR: 4.88e-04
        Epoch 84/90 | Train: 0.1965/0.9942 | Val: 0.1564/0.9395 | LR: 4.85e-04
        Epoch 85/90 | Train: 0.2229/0.9892 | Val: 0.1563/0.9368 | LR: 4.81e-04
        Epoch 86/90 | Train: 0.2137/0.9881 | Val: 0.1560/0.9368 | LR: 4.77e-04
        Epoch 87/90 | Train: 0.1984/0.9954 | Val: 0.1554/0.9368 | LR: 4.73e-04
        Epoch 88/90 | Train: 0.2019/0.9937 | Val: 0.1552/0.9368 | LR: 4.68e-04
        Epoch 89/90 | Train: 0.1924/0.9929 | Val: 0.1554/0.9368 | LR: 4.63e-04
        Early stopping at epoch 89 (patience=15)
    
      Best: Epoch 74, Val F1: 0.9428
      SwinV2 fold 4: val F1 = 0.9428
    
    SwinV2 summary: mean=0.9131 ± 0.0218
      exp06 baseline: mean=0.9089 ± 0.0348


## 11 · Summary


```python
all_results = {
    'convnext'  : convnext_results,
    'eva02'     : eva02_results,
    'dinov2'    : dinov2_results,
    'effnet_b4' : effnet_results,
    'swinv2'    : swinv2_results,
}

# exp06 baselines for delta comparison
EXP06_BASELINES = {
    'convnext'  : 0.9184,
    'eva02'     : 0.9267,
    'dinov2'    : 0.9321,
    'effnet_b4' : 0.8902,
    'swinv2'    : 0.9089,
}

print('=' * 70)
print(f'TRAINING COMPLETE — {EXP_ID}')
print('=' * 70)
for arch, results in all_results.items():
    scores   = [r['best_f1'] for r in results.values()]
    fold_str = ', '.join(f'{s:.4f}' for s in scores)
    mean     = np.mean(scores)
    delta    = mean - EXP06_BASELINES[arch]
    flag     = '↑' if delta > 0.002 else ('↓' if delta < -0.002 else '=')
    print(f'  {arch:<12}: mean={mean:.4f} ± {np.std(scores):.4f}  '
          f'(vs exp06: {delta:+.4f} {flag})  [{fold_str}]')

exp_model_dir = MODEL_DIR / EXP_ID
exp_oof_dir   = OOF_DIR   / EXP_ID
n_ckpts = len(list(exp_model_dir.glob('*.pth'))) if exp_model_dir.exists() else 0
n_oof   = len(list(exp_oof_dir.glob('*.csv')))   if exp_oof_dir.exists()   else 0

print(f'\nCheckpoints : {n_ckpts}  → {exp_model_dir}')
print(f'OOF files   : {n_oof}   → {exp_oof_dir}')
print()
print('Next steps:')
print('  1. Sync back to local:  bash scripts/sync_from_remote.sh')
print('  2. Run 11-analysis-exp07.ipynb  (OOF analysis + ensemble ablation)')
print('  3. Run 11-inference-exp07.ipynb (generate submission CSVs)')
print()
print('Key things to check in analysis:')
print('  • Did fake_screen F1 improve? (was 0.9040 in exp06)')
print('  • Did FLAT-2D→REAL error count drop? (was 34/96 errors in exp06)')
print('  • Did SwinV2 fold variance reduce? (std was 0.0348 in exp06)')
print('  • Did LOO analysis change which arch is "safe to drop"?')
```

    ======================================================================
    TRAINING COMPLETE — exp07
    ======================================================================
      convnext    : mean=0.9073 ± 0.0148  (vs exp06: -0.0111 ↓)  [0.9184, 0.9039, 0.9025, 0.8842, 0.9273]
      eva02       : mean=0.9293 ± 0.0175  (vs exp06: +0.0026 ↑)  [0.9330, 0.9186, 0.9390, 0.9023, 0.9535]
      dinov2      : mean=0.9254 ± 0.0138  (vs exp06: -0.0067 ↓)  [0.9477, 0.9136, 0.9149, 0.9152, 0.9357]
      effnet_b4   : mean=0.8826 ± 0.0310  (vs exp06: -0.0076 ↓)  [0.9375, 0.8752, 0.8892, 0.8448, 0.8661]
      swinv2      : mean=0.9131 ± 0.0218  (vs exp06: +0.0042 ↑)  [0.9325, 0.8980, 0.9085, 0.8838, 0.9428]
    
    Checkpoints : 30  → /workspace/fas-competition/models/exp07
    OOF files   : 25   → /workspace/fas-competition/oof/exp07
    
    Next steps:
      1. Sync back to local:  bash scripts/sync_from_remote.sh
      2. Run 11-analysis-exp07.ipynb  (OOF analysis + ensemble ablation)
      3. Run 11-inference-exp07.ipynb (generate submission CSVs)
    
    Key things to check in analysis:
      • Did fake_screen F1 improve? (was 0.9040 in exp06)
      • Did FLAT-2D→REAL error count drop? (was 34/96 errors in exp06)
      • Did SwinV2 fold variance reduce? (std was 0.0348 in exp06)
      • Did LOO analysis change which arch is "safe to drop"?



```python

```
