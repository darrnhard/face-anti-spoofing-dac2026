# ML/DL Project Structure — Professional Standards

> A deep research synthesis of professional standards used by Kaggle grandmasters, ML engineers,
> and research teams. Applied to the FAS (Face Anti-Spoofing) competition codebase.
>
> **Sources:** Cookiecutter Data Science (DrivenData), *Organizing Code, Experiments, and Research
> for Kaggle Competitions* (TDS, 2025), lightning-hydra-template, neptune.ai DL best practices,
> MLOps Guide (mlops-guide.github.io)

---

## Table of Contents

1. [Directory Structure](#1-directory-structure)
2. [File Naming Conventions](#2-file-naming-conventions)
3. [Notebook Splitting Strategy](#3-notebook-splitting-strategy)
4. [Iteration Versioning](#4-iteration-versioning)
5. [Experiment Tracking Tools](#5-experiment-tracking-tools)
6. [Scripts vs. Notebooks Philosophy](#6-scripts-vs-notebooks-philosophy)
7. [Applied — FAS Competition Template](#7-applied--fas-competition-template)

---

## 1. Directory Structure

The universal standard is **Cookiecutter Data Science (CCDS)** by DrivenData, adopted across
industry teams, academic research groups, and competitive Kaggle work.

**Core principles:**
- `data/raw/` is immutable — never overwrite the original data dump
- `src/` is the real codebase — the source of truth for reusable logic
- `notebooks/` is for exploration only — not the source of truth
- Configs are version-controlled alongside code

```
fas-competition/
│
├── README.md                        # Project overview, setup, experiment summary
├── Makefile                         # make preprocess / make train / make submit
├── pyproject.toml                   # Dependencies (uv or pip-tools managed)
├── .gitignore
│
├── data/                            # NEVER commit — add to .gitignore
│   ├── raw/                         # Original, immutable competition data dump
│   ├── interim/                     # Intermediate transformations (face crops, etc.)
│   ├── processed/                   # Final model-ready data (augmented, split)
│   └── external/                    # 3rd-party data (CelebA-Spoof, NUAA, etc.)
│
├── notebooks/                       # Exploration + PoC only
│   ├── eda/
│   │   └── 01-eda-class-distribution.ipynb
│   ├── preprocessing/
│   │   └── 02-preprocess-face-crop.ipynb
│   ├── experiments/                 # One notebook per experiment attempt
│   │   ├── exp01-autogluon-baseline.ipynb
│   │   ├── exp02-efficientnet-b0.ipynb
│   │   └── exp03-efficientnet-b4-mixup.ipynb
│   ├── analysis/
│   │   ├── 04-error-analysis-confusion.ipynb
│   │   └── 05-experiment-comparison.ipynb
│   ├── inference/
│   │   └── 06-inference-generate-submission.ipynb
│   └── scratch/                     # Throwaway prototyping — delete freely
│       └── scratch-debug-augmentation.ipynb
│
├── src/                             # THE reusable codebase (or name it fas/)
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py               # PyTorch Dataset classes
│   │   ├── augmentation.py          # get_train_transforms(), get_val_transforms()
│   │   └── preprocessing.py         # run_face_crop(), split_dataset()
│   ├── models/
│   │   ├── efficientnet.py          # FASEfficientNet(nn.Module)
│   │   └── ensemble.py              # WeightedEnsemble, OOFEnsemble
│   ├── training/
│   │   ├── trainer.py               # Lightning module or custom train loop
│   │   ├── losses.py                # BCE, FocalLoss
│   │   └── metrics.py               # compute_acer(), compute_apcer()
│   └── utils/
│       ├── config.py                # DATA_DIR, MODELS_DIR, SUBMISSION_DIR paths
│       ├── seed.py                  # set_seed(42)
│       └── logger.py
│
├── scripts/                         # CLI-runnable entry points
│   ├── preprocess.py                # python scripts/preprocess.py
│   ├── train.py                     # python scripts/train.py --config exp03
│   ├── evaluate.py
│   └── predict.py                   # generates submission.csv
│
├── configs/                         # YAML experiment configs — version controlled
│   ├── base.yaml                    # Default hyperparameters
│   └── experiments/
│       ├── exp01_autogluon.yaml
│       ├── exp02_effnet_b0.yaml
│       └── exp03_effnet_b4_mixup.yaml
│
├── models/                          # Saved checkpoints (gitignored)
│   ├── exp02_effnet_b0/
│   │   ├── best_val_auc0.934.ckpt
│   │   └── config_snapshot.yaml     # Config at time of save — CRITICAL for reproducibility
│   └── exp03_effnet_b4_mixup/
│       ├── best_val_auc0.951.ckpt
│       └── config_snapshot.yaml
│
├── submissions/                     # All generated submission CSVs
│   ├── sub_exp02_cv0.934_lb0.109.csv
│   └── sub_exp03_cv0.951_lb0.096.csv
│
├── reports/                         # Generated analysis and figures
│   ├── figures/
│   └── experiment_log.md            # Living lab notebook — update every experiment
│
└── references/                      # Papers, competition rules, data docs
    └── ml_project_structure.md      # ← This file
```

> **Key rule:** For solo Kaggle, the non-negotiables are `data/raw/`, `src/`, `configs/`,
> `notebooks/experiments/`, and `submissions/`. Everything else is additive.

---

## 2. File Naming Conventions

The master rule: **a filename should answer "what is this and how good was it?" without opening
the file.**

### Notebooks

| Pattern | Good Example | Notes |
|---|---|---|
| `[NN]-[desc].ipynb` | `01-eda-class-balance.ipynb` | CCDS standard. Number controls ordering. Lowercase, dash-delimited. |
| `[expNN]-[model]-[tweak].ipynb` | `exp03-effnet-b4-mixup.ipynb` | For experiments. Zero-padded, sequential. `exp` prefix distinguishes from EDA. |
| `scratch-[desc].ipynb` | `scratch-debug-augment.ipynb` | Throwaway. `scratch-` prefix = "don't clean this up". |
| ~~`notebook_v2_FINAL.ipynb`~~ | ❌ Anti-pattern | Version suffixes on notebooks are a red flag — use Git or experiment numbering instead. |

### Python Source Files (`src/`)

| Pattern | Example | Notes |
|---|---|---|
| `snake_case.py` | `face_dataset.py` | Always snake_case. PEP 8. |
| `[function]_[noun].py` | `augmentation_factory.py` | Name after what it contains. |
| Action verb for CLI scripts | `train.py`, `evaluate.py`, `predict.py` | Run as `python scripts/train.py`. |

### Config Files (`configs/`)

| Pattern | Example | Notes |
|---|---|---|
| `base.yaml` | `base.yaml` | Default hyperparameters. All experiments override from here. |
| `[expNN]_[model]_[tweak].yaml` | `exp03_effnet_b4_mixup.yaml` | Matches the experiment notebook name. Only stores the *delta* from base. |

### Model Checkpoints

| Pattern | Example | Notes |
|---|---|---|
| `best_[metric][score].ckpt` | `best_val_auc0.951.ckpt` | Instantly scannable without opening. |
| `epoch=[N]-step=[N].ckpt` | `epoch=12-step=2340.ckpt` | PyTorch Lightning default — good for resume checkpoints. |
| `[expNN]/best_*.ckpt` | `exp03_effnet_b4_mixup/best_val_auc0.951.ckpt` | Always nest inside experiment folder. Always save `config_snapshot.yaml` alongside. |

### Submissions

| Pattern | Example | Notes |
|---|---|---|
| `sub_[exp]_cv[score]_lb[score].csv` | `sub_exp03_cv0.951_lb0.096.csv` | Best scheme — shows experiment, CV score, and LB score at a glance. |
| `sub_[date]_[exp].csv` | `sub_20250402_exp03.csv` | Alternative if multiple submissions per day. |
| ~~`submission.csv`~~ | ❌ Anti-pattern | You'll have 20 files and won't know which is which in 2 weeks. |

---

## 3. Notebook Splitting Strategy

**Never run the full pipeline in one notebook.** Each pipeline stage becomes its own notebook.
Expensive stages (e.g., preprocessing that takes 7+ hours) are run once and cached — the
notebook becomes a frozen artifact.

### Pipeline Stages

```
[EDA] → [Preprocessing] → [Experiment × N] → [Evaluation] → [Inference]
  NB00        NB01             NB02+              NB03           NB04
```

| Stage | Prefix | Purpose | Reruns? |
|---|---|---|---|
| EDA | `01-eda-` | Understand data, distributions, class balance | Rarely |
| Preprocessing | `02-preprocess-` | Face crop, normalize, split. Saves to `data/processed/` | Once |
| Experiments | `exp01-`, `exp02-` | One notebook per experiment. Loads from `data/processed/` | Each new run |
| Evaluation | `04-eval-` | Compare multiple models, confusion matrix, error analysis | As needed |
| Inference | `05-infer-` | Load best checkpoint, run on test set, generate submission | Per submission |
| Scratch | `scratch-` | Throwaway prototyping. Not part of the pipeline. | Delete freely |

### The Module Import Trick

Add this to the top of every notebook to import from `src/`:

```python
import sys, os
sys.path.insert(0, os.path.abspath('../'))  # adjust depth as needed

# Now you can import from src/
from src.data.dataset import FASDataset
from src.training.metrics import compute_acer
from src.utils.seed import set_seed
```

For modules you actively edit, use `importlib.reload` to avoid restarting the kernel:

```python
from importlib import reload
from src.data import dataset
reload(dataset)
```

> **Rule:** If the same code block appears in more than one notebook, it belongs in `src/`.

---

## 4. Iteration Versioning

Professionals use a **two-level versioning system**: Git for code, experiment IDs for runs.
These serve different purposes and must not be conflated.

### Level 1 — Git (Code Versioning)

Every meaningful change gets committed. Tag commits that produced a submission.

```bash
# Commit convention
git commit -m "exp03: add MixUp augmentation, EfficientNet-B4"
git commit -m "fix: correct label leak in validation split"
git commit -m "refactor: extract FASDataset to src/data/dataset.py"

# Tag submissions with LB score
git tag "lb-0.096-exp03" HEAD
git push origin --tags
```

### Level 2 — Experiment IDs (Run Versioning)

Each training run gets a sequential ID. Never delete or overwrite past experiments — even
bad ones are informative history.

```
exp01 → exp02 → exp03 → exp04 → ...
```

The `experiment_log.md` is the living record (see §7 for template).

### Level 3 — YAML Configs (Hyperparameter Snapshots)

Every experiment has a YAML config that fully specifies it. Store only the *delta* from
`base.yaml`:

```yaml
# configs/experiments/exp03_effnet_b4_mixup.yaml
_base_: ../base.yaml

model:
  name: efficientnet_b4
  pretrained: true

training:
  epochs: 30
  lr: 1.0e-4
  batch_size: 32

augmentation:
  mixup_alpha: 0.4
  use_cutmix: true
```

### Level 4 — Checkpoint Naming

```
models/
├── exp02_effnet_b0/
│   ├── best_val_auc0.934.ckpt    ← metric in filename
│   └── config_snapshot.yaml      ← frozen config for reproducibility
└── exp03_effnet_b4_mixup/
    ├── best_val_auc0.951.ckpt
    └── config_snapshot.yaml
```

Always save a frozen copy of the config alongside the checkpoint. This is what makes
results reproducible months later.

### Anti-Patterns to Eliminate

- ❌ `model_v2.ckpt`, `model_final.ckpt`, `model_new.ckpt`
- ❌ `train_copy.py`, `train_v2.py`
- ❌ `notebook_FINAL_v3.ipynb`
- ❌ Overwriting a previous experiment's checkpoint folder

---

## 5. Experiment Tracking Tools

You need to track: hyperparameters, metrics per epoch, git commit hash, and model artifacts.

| Tool | Use Case | Verdict |
|---|---|---|
| **W&B (wandb)** | Live metric logging, dashboards, artifact storage. ~3 lines to integrate with PyTorch/Lightning. | ✅ Recommended |
| **MLflow** | Open-source, self-hosted, no cloud account. Good for offline/private work. | Optional |
| **experiment_log.md** | Manual markdown table. Zero-friction, human-readable audit trail. Keep this even if you use W&B. | ✅ Always do |
| **Hydra** | YAML config composition, CLI hyperparameter sweeps. Used by the lightning-hydra-template community. | Optional |
| **DVC** | Git for large files (data + model checkpoints). Overkill for solo Kaggle; essential for team/production. | Later |

**Minimum viable setup for a Kaggle competition:**
W&B for live metric logging + `experiment_log.md` for high-level decisions + Git tags on
submission commits. That covers 90% of the value at 10% of the complexity.

### W&B Quick Integration (PyTorch Lightning)

```python
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project="fas-competition", name="exp03-effnet-b4-mixup")
trainer = pl.Trainer(logger=wandb_logger, ...)
```

---

## 6. Scripts vs. Notebooks Philosophy

The most important professional principle in ML engineering. Notebooks and scripts are
fundamentally different tools with **non-overlapping roles**.

### Notebooks are for:
- Exploratory Data Analysis (EDA)
- Quickly prototyping a new idea
- Visualizing results and failure cases
- Creating figures and reports
- Running a single experiment interactively
- Communication (markdown-heavy narratives)

### `src/` scripts are for:
- Reusable `Dataset` / `DataModule` classes
- Model architecture definitions
- Training and evaluation loops
- Loss functions and metrics
- Any code used in 2+ notebooks
- Reproducible, CLI-runnable pipeline execution

### The Refactoring Workflow

```
Prototype in notebook
       ↓
Extract reusable function/class to src/
       ↓
Import it back into the notebook
       ↓
Notebook becomes thin; src/ is the real codebase
```

When a bug is found in `src/data/dataset.py`, it is fixed **once** and all notebooks that
import it are immediately fixed.

> **The classic mistake:** Copying the same preprocessing block across 5 experiment notebooks.
> When a bug appears, you fix it 5 times (or miss some). This is exactly what happens when a
> codebase grows without structure. The fix is extracting to `src/data/preprocessing.py`.

---

## 7. Applied — FAS Competition Template

### Full Directory Tree

```
fas-competition/
│
├── README.md
├── experiment_log.md                # Living lab notebook — update every experiment
├── pyproject.toml
├── .gitignore                       # data/, models/, submissions/, __pycache__, .env
│
├── data/
│   ├── raw/                         # Original competition images — NEVER touch
│   ├── interim/                     # Face crops, normalized images
│   └── processed/                   # Train/val splits
│
├── notebooks/
│   ├── eda/
│   │   ├── 01-eda-class-distribution.ipynb
│   │   └── 02-eda-sample-visualization.ipynb
│   ├── preprocessing/
│   │   └── 03-preprocess-face-crop.ipynb      # Run once → saves to data/processed/
│   ├── experiments/
│   │   ├── exp01-autogluon-automm-baseline.ipynb
│   │   ├── exp02-efficientnet-b0-default-aug.ipynb
│   │   ├── exp03-efficientnet-b4-mixup.ipynb
│   │   └── exp04-ensemble-exp02-exp03.ipynb
│   ├── analysis/
│   │   ├── 04-error-analysis-confusion.ipynb
│   │   └── 05-experiment-comparison.ipynb
│   ├── inference/
│   │   └── 06-inference-generate-submission.ipynb
│   └── scratch/
│       └── scratch-test-augmentation-preview.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── dataset.py               # FASDataset(Dataset) — shared across all exp notebooks
│   │   ├── augmentation.py          # get_train_transforms(), get_val_transforms()
│   │   └── preprocessing.py         # run_face_crop(), split_dataset()
│   ├── models/
│   │   ├── efficientnet.py          # FASEfficientNet(nn.Module)
│   │   └── ensemble.py              # WeightedEnsemble, OOFEnsemble
│   ├── training/
│   │   ├── trainer.py               # Lightning module or custom train loop
│   │   ├── losses.py                # BCE, FocalLoss
│   │   └── metrics.py               # compute_acer(), compute_apcer(), compute_bpcer()
│   └── utils/
│       ├── config.py                # DATA_DIR, MODELS_DIR, SUBMISSION_DIR
│       └── seed.py                  # set_seed(42)
│
├── configs/
│   ├── base.yaml
│   └── experiments/
│       ├── exp01_autogluon.yaml
│       ├── exp02_effnet_b0.yaml
│       └── exp03_effnet_b4_mixup.yaml
│
├── models/                          # gitignored
│   ├── exp02_effnet_b0/
│   │   ├── best_val_auc0.934.ckpt
│   │   └── config_snapshot.yaml
│   └── exp03_effnet_b4_mixup/
│       ├── best_val_auc0.951.ckpt
│       └── config_snapshot.yaml
│
└── submissions/
    ├── sub_exp02_cv0.934_lb0.109.csv
    └── sub_exp03_cv0.951_lb0.096.csv
```

### `experiment_log.md` Template

```markdown
# FAS Competition — Experiment Log

## Active Best: exp03 | CV ACER: 0.049 | LB ACER: 0.096

| ID    | Date       | Model           | Augmentation  | Epochs | CV ACER | LB ACER | Notes                       |
|-------|------------|-----------------|---------------|--------|---------|---------|-----------------------------|
| exp01 | 2025-03-10 | AutoGluon AutoMM| default       | auto   | 0.142   | 0.189   | Baseline. Fast to run.      |
| exp02 | 2025-03-18 | EfficientNet-B0 | flip+crop     | 25     | 0.091   | 0.109   | First custom PyTorch        |
| exp03 | 2025-03-28 | EfficientNet-B4 | +MixUp        | 30     | 0.049   | 0.096   | ✓ Current best              |
| exp04 | planned    | Ensemble E2+E3  | —             | —      | —       | —       | OOF blend                   |

## Next to Try

- [ ] Test-Time Augmentation (TTA) on exp03
- [ ] Freeze backbone for first 5 epochs (gradual unfreezing)
- [ ] CBAM attention on EfficientNet-B4
- [ ] Try DepthwiseNet / binary texture features as auxiliary input

## Key Decisions

- **2025-03-18:** Switched from AutoGluon to custom PyTorch — more control over augmentation pipeline
- **2025-03-28:** MixUp alpha=0.4 gave the biggest single improvement (+0.042 CV ACER)
```

### `.gitignore` Template

```gitignore
# Data
data/

# Models and checkpoints
models/
*.ckpt
*.pt
*.pth
*.pkl

# Submissions
submissions/

# Python
__pycache__/
*.pyc
*.egg-info/
.venv/
.env

# Jupyter
.ipynb_checkpoints/

# Experiment tracking
wandb/
mlruns/
lightning_logs/

# OS
.DS_Store
```

### Migration Strategy

Don't restructure everything at once. Incremental migration is how the pros do it:

1. **Day 1:** Create the folder skeleton above
2. **Day 1:** Move raw data into `data/raw/` — do not modify it
3. **Next experiment:** Save it to `notebooks/experiments/exp0N-...`
4. **Ongoing:** When you write the same block twice, extract it to `src/`
5. **Per submission:** Save to `submissions/sub_expNN_cv_lb.csv` and Git tag it

---

*Last updated: 2025-04-02*
