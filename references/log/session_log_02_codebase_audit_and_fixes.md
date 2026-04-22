# Session Log — Codebase Audit & Best-Practice Fixes

**Competition:** FindIT-DAC 2026 — Face Anti-Spoofing (6-class, Macro F1)  
**Session date:** 2026-04-07  
**Status:** ✅ Complete  

---

## Context Going Into This Session

- exp04 training complete: 4 architectures × 5 folds = 20 models
- Best OOF F1: 0.9365 (Top-3 ensemble: DINOv2 + EVA-02 + ConvNeXt)
- Project had no `CLAUDE.md` or `README.md`
- No prior documentation audit of src/ code against library best practices
- Context7 MCP server available for pulling live library docs

---

## What We Did This Session

### 1. Project Documentation (CLAUDE.md + README.md)

**Created `CLAUDE.md`** — initial project guidance file for Claude Code, covering:
- Competition overview (Kaggle, FindIT-DAC 2026, Macro F1)
- Common commands (install, sync scripts, notebook execution)
- Architecture and code structure walkthrough
- Training pipeline summary (2-stage, LLRD, key hyperparameters)
- Known issues at the time (placeholder)

**Created `README.md`** — competition-oriented documentation covering:
- Results table (OOF F1 per architecture + best ensemble)
- Full step-by-step reproduction pipeline (data prep → training → inference)
- Key hyperparameter table with rationale
- Project structure tree
- Remote training commands (Vast.ai sync)

**Updated `CLAUDE.md`** to correct all outdated v1/v2 information discovered during audit:
- Dataset count corrected: 1,464 images (not 1,342)
- Best results updated: 0.9365 ensemble OOF F1 (not 0.8856)
- Data pipeline updated to include InsightFace + HDBSCAN + StratifiedGroupKFold
- Added "Calibration & Ensemble Strategy" section (ECE per architecture, Top-3 rationale)
- Known bugs section split into Fixed vs Still Relevant

---

### 2. Full End-to-End Code Audit

Audited all files in `src/` and all notebooks exported in `nb-exports/audit/` using parallel sub-agents. Each agent focused on a different layer of the stack.

#### What was audited:
- `src/models/registry.py` — architecture configs, parameter group splitting
- `src/training/trainer.py` — 2-stage loop, EMA, GradScaler, scheduler, CutMix
- `src/training/losses.py` — FocalLoss, Poly1Loss
- `src/training/metrics.py` — macro_f1, threshold optimization, nested CV
- `src/data/preprocessing.py` — MTCNN, InsightFace, HDBSCAN, MD5 dedup
- `src/data/dataset.py` — FaceSpoofDataset
- `src/data/augmentation.py` — all custom PIL transforms, get_transforms()
- `src/inference/predict.py` — TTA inference
- `src/inference/threshold.py` — apply_thresholds, robust_threshold_optimization
- `src/inference/submission.py` — make_submission
- `src/utils/config.py` — paths, constants, mkdir loop
- `src/utils/seed.py` — set_seed
- `configs/base.yaml` — global hyperparameters
- `nb-exports/audit/02-data-preparation-v3.md`
- `nb-exports/audit/exp04-four-arch-ensemble.md`
- `nb-exports/audit/06-inference-exp03.md`
- `nb-exports/audit/analysis.md`

#### Confirmed NOT bugs (false positives cleared):
| Item | Why it's fine |
|---|---|
| TTA averaging in predict.py | `all_probs /= len(transforms)` is outside the for loop — correct |
| Stage 1 freezing for LLRD archs | `get_param_groups()[1]` returns head params correctly |
| JPEGCompression BytesIO | `.convert('RGB')` forces eager JPEG decode; buffer not needed after |
| RandomErasing after Normalize | Filling with 0 = ImageNet mean in normalized space — correct per torchvision docs |
| Nelder-Mead without bounds | Intentional design choice with documented rationale |
| MTCNN `detect()` usage | Actual file uses safe `and`/`>` short-circuit pattern |
| InsightFace L2 normalization | Correct and necessary (InsightFace does not normalize internally) |
| `timm.create_model` calls | All correct, including `img_size=224` DINOv2 override (pos_embed resampled) |
| `model.stages` / `model.blocks` | Confirmed in timm source: ConvNeXt has `.stages`, ViTs have `.blocks` |
| `GaussianBlur(kernel_size=5)` | 5 is odd — passes torchvision validation |

---

### 3. Context7 MCP Docs Verification

Used Context7 MCP (`mcp__context7__resolve-library-id` + `mcp__context7__get-library-docs`) to check all key libraries against their latest published documentation. Libraries checked:

- **PyTorch** (AMP, GradScaler, AveragedModel, CosineAnnealingWarmRestarts)
- **timm** (create_model API, model attribute structure)
- **torchvision** (transforms pipeline ordering, RandomErasing, GaussianBlur)
- **scikit-learn** (StratifiedGroupKFold, f1_score)
- **scipy** (Nelder-Mead options keys)
- **Pillow** (Image.BILINEAR removal, BytesIO pattern)
- **facenet-pytorch** (MTCNN detect() return format)
- **insightface** (FaceAnalysis init, embedding normalization)
- **hdbscan** (fit_predict, noise handling)

---

### 4. Bugs Found (10 Total)

| # | Severity | File | Issue |
|---|---|---|---|
| 1 | **Critical** | `augmentation.py:75-76` | `Image.BILINEAR` removed in Pillow 10.0 — `AttributeError` on any current environment |
| 2 | **High** | `trainer.py:115, 378` | `AveragedModel` missing `use_buffers=True` — EfficientNet-B4 BN running stats not EMA'd |
| 3 | **High** | `trainer.py:32`, `predict.py:17` | `from torch.cuda.amp import GradScaler, autocast` deprecated since PyTorch 2.4 |
| 4 | **High** | `threshold.py:49, 111` | No `np.abs()` on thresholds — negative optimizer solution would invert class preference ordering |
| 5 | **Medium** | `trainer.py:598` | `scheduler.step(stage2_ep - warmup_ep)` deprecated in PyTorch 2.4+ (`FutureWarning`) |
| 6 | **Medium** | `metrics.py:56-57` | `optimize_thresholds()` missing all-ones anchor restart — no guarantee result ≥ raw argmax |
| 7 | **Medium** | `preprocessing notebook` | `StratifiedGroupKFold` fold assignment using `.loc[df.index[val_idx]]` — unsafe with non-default index |
| 8 | **Medium** | `configs/base.yaml` | ConvNeXt `use_llrd: false` (wrong); 6 augmentation params out of sync with actual code |
| 9 | **Low** | `config.py:96-98` | `REPORTS_DIR` defined but omitted from auto-mkdir loop — crashes on fresh environment |
| 10 | **Low** | `trainer.py:391-393` | DINOv2 probe: `scaler.step()` without prior `scaler.unscale_()` — inconsistent with main loop |

---

### 5. All Fixes Applied

#### `src/data/augmentation.py`
- **Fix #1**: `Image.BILINEAR` → `Image.Resampling.BILINEAR` in `DownscaleUpscale.__call__()`

#### `src/training/trainer.py`
- **Fix #2**: Added `use_buffers=True` to both `AveragedModel(...)` calls (setup_training + linear probe)
- **Fix #3**: `from torch.cuda.amp import GradScaler, autocast` → `from torch.amp import GradScaler, autocast`
- **Fix #3**: All `GradScaler()` → `GradScaler("cuda")`
- **Fix #3**: All `autocast()` → `autocast("cuda")` (train_one_epoch, validate, dinov2_linear_probe)
- **Fix #5**: `scheduler.step(stage2_ep - warmup_ep)` → `scheduler.step()`
- **Fix #10**: Added `scaler.unscale_(optimizer)` before `scaler.step()` in dinov2_linear_probe

#### `src/inference/predict.py`
- **Fix #3**: `from torch.cuda.amp import autocast` → `from torch.amp import autocast`
- **Fix #3**: `autocast()` → `autocast("cuda")` in predict_test()

#### `src/inference/threshold.py`
- **Fix #4**: `apply_thresholds()` — `(probs * thresholds)` → `(probs * np.abs(thresholds))`
- **Fix #4**: `best_thresh = res.x` → `best_thresh = np.abs(res.x)`

#### `src/training/metrics.py`
- **Fix #6**: Added all-ones anchor as restart 0 in `optimize_thresholds()` — guarantees result ≥ argmax baseline

#### `src/utils/config.py`
- **Fix #9**: Added `REPORTS_DIR` to the `mkdir` loop

#### `src/data/preprocessing.py`
- **Fix (bonus)**: `next_id = int(labels.max()) + 1` → `next_id = max(0, int(labels.max()) + 1)` — HDBSCAN all-noise edge case guard

#### `configs/base.yaml`
- **Fix #8**: ConvNeXt `use_llrd: false` → `use_llrd: true` with correct comment and `llrd_factor: 0.85`
- **Fix #8**: Synced all augmentation parameters to match actual `augmentation.py` values:
  - `jpeg_quality_low`: 30 → 20
  - `color_jitter` brightness/contrast/saturation: 0.2 → 0.3
  - `moire_pattern_p`: 0.2 → 0.15
  - `color_gamut_reduction_p`: 0.15 → 0.1
  - `random_erasing_p`: 0.2 → 0.25
  - Added `screen_bezel_overlay_p: 0.08` (was missing entirely)
- **Fix #8**: Added per-architecture `warmup_epochs` to all 4 architecture blocks (CNNs=3, ViTs=5)
- Updated global `warmup_epochs` comment to clarify it varies per architecture

#### `notebooks/preprocessing/02-data-preparation-v3.ipynb` (cell `297892c7`)
- **Fix #7**: `train_df.loc[train_df.index[val_idx], 'fold'] = fold` → `train_df.iloc[val_idx, fold_col] = fold`

---

## Impact Summary

| Category | Impact |
|---|---|
| **Runtime crash fix** | `Image.BILINEAR` fix prevents `AttributeError` on Pillow 10+ — training data loading would crash silently |
| **Accuracy fix** | `use_buffers=True` fix ensures EfficientNet-B4 EMA model evaluates with correct BN statistics |
| **Safety fix** | `np.abs()` in threshold.py prevents catastrophic prediction inversion if optimizer wanders negative |
| **Warning elimination** | PyTorch 2.4+ `FutureWarning` on `torch.cuda.amp` and `scheduler.step(epoch)` both cleared |
| **Correctness fix** | `metrics.py` threshold optimization now guaranteed ≥ raw argmax baseline |
| **Robustness fix** | Notebook fold assignment now safe regardless of DataFrame index state |
| **Documentation sync** | `base.yaml` now accurately reflects actual training configuration |
| **Fresh-environment fix** | `REPORTS_DIR` auto-created; analysis notebook no longer fails on new machines |

---

## Files Changed

```
src/data/augmentation.py
src/training/trainer.py
src/training/metrics.py
src/inference/predict.py
src/inference/threshold.py
src/utils/config.py
src/data/preprocessing.py
configs/base.yaml
notebooks/preprocessing/02-data-preparation-v3.ipynb
references/README.md           ← created
CLAUDE.md                      ← created then updated
```

---

## Notes for Next Session

- All fixes are in `src/` — **no retraining required**. The exp04 checkpoints remain valid. The `use_buffers=True` fix affects future training runs only; existing EMA checkpoints were saved from `.module.state_dict()` (the underlying model), so inference is unaffected.
- The `scheduler.step()` change (fix #5) resets the scheduler's internal epoch counter rather than manually tracking T_cur. This changes the LR trajectory slightly for any future runs but is the correct PyTorch 2.4+ pattern.
- The `base.yaml` is documentation only — the code reads from `ARCH_CONFIGS` in `registry.py` at runtime. No behavior changed by YAML edits.
- Context7 confirmed: `timm` model attribute access (`model.stages`, `model.blocks`), `create_model` API, and all torchvision transform ordering are all correct and up-to-date.
