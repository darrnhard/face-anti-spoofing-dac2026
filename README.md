# FindIT-DAC 2026 — Face Anti-Spoofing

**Platform**: Kaggle | **Competition**: FindIT-DAC 2026  
**Task**: 6-class face anti-spoofing classification  
**Metric**: Macro F1-score

Classify face images into 6 attack types: `realperson`, `fake_printed`, `fake_screen`, `fake_mask`, `fake_mannequin`, `fake_unknown`.

---

## Results

### exp07 — Single Architecture (OOF F1)

| Architecture | Mean OOF F1 | Std | Per-fold F1 |
|---|---|---|---|
| EVA-02-Base | **0.9293** | 0.0175 | [0.9330, 0.9186, 0.9390, 0.9023, 0.9535] |
| DINOv2-Base (ViT-B/14 reg4) | **0.9254** | 0.0138 | [0.9477, 0.9136, 0.9149, 0.9152, 0.9357] |
| SwinV2-Base | **0.9131** | 0.0218 | [0.9325, 0.8980, 0.9085, 0.8838, 0.9428] |
| ConvNeXt-Base | **0.9073** | 0.0148 | [0.9184, 0.9039, 0.9025, 0.8842, 0.9273] |
| EfficientNet-B4 | 0.8826 | 0.0310 | [0.9375, 0.8752, 0.8892, 0.8448, 0.8661] |

### Ensemble OOF Comparison

| Ensemble | OOF F1 | Notes |
|---|---|---|
| exp04 Top-3 (DINOv2 + EVA-02 + ConvNeXt) | **0.9365** | 4 archs, no pseudo-labels |
| exp06 All-5 | **0.9355** | 5 archs, Round 1 pseudo-labels |
| exp07 Pruned-Mega (exp06×5 + exp07×3) | **0.9319** | Cross-experiment; OOF↑ but LB↓ |
| exp07 Top-4 (drop ConvNeXt) | **0.9306** | exp07-only; LOO +0.0053 vs all-5 |
| exp07 All-5 | **0.9252** | exp07-only; equal weights |

> **OOF-vs-LB inversion**: pruned_mega had the highest OOF (0.9319) but lowest LB (0.78555) among exp07 submissions. exp06 checkpoints in the mega-ensemble appear overfit to the training distribution.

### Final Public Leaderboard

| Submission | Public LB | Strategy |
|---|---|---|
| `all5_argmax.csv` | **0.79207** | exp07 all-5 equal weights, argmax — **BEST** |
| `scipy_opt_argmax.csv` | 0.78673 | Nelder-Mead optimized weights |
| `top4_argmax.csv` | 0.78673 | Drop ConvNeXt (LOO +0.0053 OOF) |
| `pruned_mega_argmax.csv` | 0.78555 | exp06×5 + exp07×3 |
| `convnext_2x_argmax.csv` | 0.77272 | ConvNeXt upweighted 2× |

---

## Setup

```bash
pip install -r requirements.txt
```

**Hardware**: Local dev on GTX 1050 Ti (4GB VRAM); training on remote RTX 3090.

---

## Data

Competition data is not included in this repository (Kaggle redistribution terms).

1. Download from the [FindIT-DAC 2026 Kaggle competition page](https://www.kaggle.com/competitions/data-analytics-competition-dac-find-it-2026)
2. Extract and place class folders under `data/interim/train/<class>/` and `data/interim/test/`
3. Run the data preparation notebook to generate crops and fold assignments

---

## Pretrained Models

exp07 checkpoints (25 total: 5 architectures × 5 folds) and OOF predictions are hosted on HuggingFace Hub.

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Download a single checkpoint
ckpt_path = hf_hub_download(
    repo_id="darrnhard/findit-dac-2026",
    filename="models/exp07/eva02/fold_0/best_model.pth",
)

# Download everything (models + OOF)
local_dir = snapshot_download(repo_id="darrnhard/findit-dac-2026")
```

> HuggingFace repo: [huggingface.co/darrnhard/findit-dac-2026](https://huggingface.co/darrnhard/findit-dac-2026)

---

## Reproducing the Pipeline

### 1. Data Preparation

```
notebooks/preprocessing/02-data-preparation-v3.ipynb
```

- **Source**: 1,642 raw images across 6 class folders in `data/interim/train/`
- **Deduplication**: MD5 hashing removes 178 same-label duplicates → **1,464 clean images**
- **Face cropping**: MTCNN with 1.4× bounding box expansion, resized to 224×224; center-crop fallback for 105 no-face images
- **Identity grouping**: InsightFace Buffalo-L (ArcFace R100) generates 512-D embeddings → HDBSCAN clustering → **807 identity groups**; prevents same-identity images leaking across folds
- **Fold assignment**: `StratifiedGroupKFold(n_splits=5)` — zero group leakage, class balance CV% < 1.9%
- **Outputs**: `data/processed/crops/train/` (224×224 JPGs), `data/processed/train_clean.csv` (with `fold` and `group` columns)

### 2. Pseudo-Labeling

```
notebooks/analysis/pseudo-label.ipynb       # Round 1 (exp04 → exp06)
notebooks/analysis/pseudo-label-r2.ipynb    # Round 2 (exp06 → exp07)
```

- **Round 1**: exp04 ensemble → 239 pseudo-labels (threshold 0.90, mean confidence 0.9597) → `train_pseudo.csv` (1,703 rows: 1,464 real + 239 pseudo)
- **Round 2**: exp06 all-5 ensemble → pseudo-labels → used to train exp07
- **Training**: Real rows use `FocalLoss` on hard integer labels; pseudo rows use `SoftCrossEntropyLoss` on soft probability vectors, weighted by confidence
- **Pseudo rows never appear in any validation fold** (fold = −1); class weights computed from real rows only

### 3. Training

```
notebooks/experiments/exp07-training.ipynb
```

Trains **5 architectures × 5 folds = 25 checkpoints**, saved to `models/exp07/`.

**Two-stage protocol per fold:**

| Stage | What trains | Duration |
|---|---|---|
| 1 — Head warmup | Classification head only (backbone frozen) | CNNs: 5 epochs, ViTs: 10 epochs |
| 2 — Full fine-tune | All params with LLRD | convnext: 100 total epochs, others: 90 total epochs |

**Key hyperparameters:**

| Parameter | Value | Note |
|---|---|---|
| Loss | Focal (γ=1.0) + inverse-freq weights | Lower γ preserves signal on rare classes |
| Pseudo-label loss | SoftCrossEntropy weighted by confidence | Real rows only in validation |
| Optimizer | AdamW | Per-group LRs (backbone vs head) |
| Backbone LR | 1e-5 to 5e-5 | Conservative; preserves pretrained features |
| Head LR | 5e-4 to 1e-3 | — |
| LLRD factor | ConvNeXt 0.85, ViTs/SwinV2 0.75–0.80 | Block-wise LR decay |
| EMA decay | 0.999 | EMA model used for checkpointing |
| CutMix | α=1.0, p=0.5 | Stage 2 only |
| Gradient clip | 1.0 | — |
| Scheduler | CosineAnnealingWarmRestarts (T₀=20, T_mult=2) | After linear warmup |
| Early stopping | patience=15 on OOF macro F1 | — |

**Augmentation pipeline (training):**  
HFlip → RandomResizedCrop → ColorJitter → GaussianBlur → GaussianNoise → RandomRotation → JPEGCompression → DownscaleUpscale → MoiréPattern → ColorGamutReduction → ScreenBezelOverlay → ToTensor → Normalize → RandomErasing

### 4. Inference & Ensemble

```
notebooks/inference/inference-exp07.ipynb
```

- **TTA**: 3-view — original + horizontal flip + scale-centercrop (Resize 115% → CenterCrop) → average softmax
- **Primary submission**: `all5_argmax.csv` — equal-weight average of all 5 exp07 architectures, argmax prediction (best LB: **0.79207**)
- **Threshold optimization**: Nested CV (Nelder-Mead) OOF = 0.9216 (−0.0036 vs argmax) — overfitting signal detected; use argmax

---

## Analysis

```
notebooks/analysis/analysis-exp07.ipynb
```

Post-training analysis produces:
- Per-class and per-sample error analysis
- Cleanlab noisy label detection — **21 samples flagged (1.4%)**, 7 consensus across all 5 models
- Calibration (ECE) per architecture
- Hypothesis scorecard testing

**Key findings (exp07):**
- Hardest class: `fake_screen` F1 = 0.8933
- Dominant error: FLAT-2D (fake_screen/fake_printed) → REAL (realperson) = 35/96 errors
- Hypothesis results: H1 fake_screen↑ ❌ | H2 fewer errors ❌ | H3 SwinV2 std < 0.035 ✅ | H4 ensemble↑ ❌

---

## Project Structure

```
├── src/                         # Importable ML code
│   ├── models/registry.py       # ARCH_CONFIGS — single source of truth (5 architectures)
│   ├── models/loader.py         # load_model_from_checkpoint
│   ├── training/trainer.py      # 2-stage training loop with pseudo-label support
│   ├── training/losses.py       # FocalLoss, Poly1Loss, SoftCrossEntropyLoss, get_class_weights()
│   ├── training/metrics.py      # macro_f1(), nested_cv_thresholds()
│   ├── data/                    # FaceSpoofDataset, preprocessing, augmentation
│   └── inference/               # predict_test() (3-view TTA), threshold, CSV generation
├── configs/
│   ├── base.yaml                # Global hyperparameters
│   └── exp03–exp07/             # Per-fold, per-architecture configs (auto-generated)
├── notebooks/
│   ├── eda/01-main-eda.ipynb
│   ├── preprocessing/02-data-preparation-v3.ipynb
│   ├── experiments/             # exp04-four-arch-ensemble.ipynb → exp07-training.ipynb
│   ├── inference/               # inference-exp07.ipynb, resubmit-exp07.ipynb
│   └── analysis/                # analysis-exp07.ipynb, pseudo-label.ipynb, pseudo-label-r2.ipynb
├── nb-exports/final-export/     # Latest notebook exports as Markdown (readable without Jupyter)
├── reports/                     # Error analysis, cleanlab flags, calibration, thresholds, figures
├── references/                  # Research papers and planning documents
└── scripts/
    ├── sync_to_remote.sh        # Push code + data to remote GPU server
    └── sync_from_remote.sh      # Pull trained checkpoints back
```
