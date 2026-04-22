"""
config.py — Shared configuration for the FAS (Face Anti-Spoofing) competition.

Auto-detects whether you're running locally or on a remote GPU server.
Import this at the top of every notebook/script:

    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))   # if needed
    from utils.config import *

Directory structure (CCDS standard):
    PROJECT_DIR/
    ├── data/
    │   ├── raw/              ← original competition images (immutable)
    │   ├── interim/          ← manually cleaned images (source for preprocessing)
    │   │   ├── train/
    │   │   │   ├── fake_mannequin/
    │   │   │   ├── fake_mask/
    │   │   │   ├── fake_printed/
    │   │   │   ├── fake_screen/
    │   │   │   ├── fake_unknown/
    │   │   │   └── realperson/
    │   │   └── test/
    │   └── processed/        ← outputs of preprocessing notebook
    │       ├── crops/
    │       │   ├── train/    ← face-cropped 224×224 train images
    │       │   └── test/     ← face-cropped 224×224 test images
    │       ├── train_clean.csv
    │       └── test.csv
    ├── models/               ← trained checkpoints (gitignored)
    ├── oof/                  ← out-of-fold prediction CSVs
    ├── submissions/          ← submission CSVs
    ├── notebooks/
    ├── src/
    ├── configs/
    └── scripts/
"""

from pathlib import Path
import torch

# =============================================================================
# AUTO-DETECT ENVIRONMENT
# =============================================================================
REMOTE_PATHS = [
    Path("/workspace/fas-competition"),   # Vast.ai primary
    Path("/root/fas-competition"),        # Vast.ai alternative
]

LOCAL_PATH = Path.home() / "ML/Competition/FindIT-DAC"

PROJECT_DIR = None
for remote in REMOTE_PATHS:
    try:
        if remote.exists():
            PROJECT_DIR = remote
            ENV_NAME = "remote"
            break
    except (PermissionError, OSError):
        continue

if PROJECT_DIR is None:
    PROJECT_DIR = LOCAL_PATH
    ENV_NAME = "local"

# =============================================================================
# PATHS  (CCDS structure)
# =============================================================================
DATA_DIR      = PROJECT_DIR / "data"
RAW_DIR       = DATA_DIR    / "raw"          # original images — never modify
INTERIM_DIR   = DATA_DIR    / "interim"      # manually cleaned source images
PROCESSED_DIR = DATA_DIR    / "processed"   # all preprocessing outputs

# Source images (input to preprocessing notebook)
TRAIN_INTERIM = INTERIM_DIR / "train"
TEST_INTERIM  = INTERIM_DIR / "test"

# Face-cropped images (output of preprocessing notebook)
CROPS_DIR      = PROCESSED_DIR / "crops"
CROP_TRAIN_DIR = CROPS_DIR / "train"
CROP_TEST_DIR  = CROPS_DIR / "test"

# Output CSVs (output of preprocessing notebook)
TRAIN_CSV = PROCESSED_DIR / "train_clean.csv"
TEST_CSV  = PROCESSED_DIR / "test.csv"

# Other project directories
MODEL_DIR      = PROJECT_DIR / "models"
OOF_DIR        = PROJECT_DIR / "oof"
CONFIGS_DIR    = PROJECT_DIR / "configs"
SUBMISSION_DIR = PROJECT_DIR / "submissions"
REPORTS_DIR    = PROJECT_DIR / "reports"

# Create writable output directories
# (INTERIM_DIR is populated manually — never auto-created here)
for _d in [PROCESSED_DIR, CROP_TRAIN_DIR, CROP_TEST_DIR,
           MODEL_DIR, OOF_DIR, SUBMISSION_DIR, CONFIGS_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CLASSES
# =============================================================================
CLASSES = [
    'fake_mannequin',
    'fake_mask',
    'fake_printed',
    'fake_screen',
    'fake_unknown',
    'realperson',
]
NUM_CLASSES  = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# =============================================================================
# PREPROCESSING CONSTANTS
# =============================================================================
IMG_SIZE  = 224    # target crop resolution (pixels)
EXPANSION = 1.4    # face bounding box expansion factor (40% padding for context)

# =============================================================================
# TRAINING CONSTANTS
# =============================================================================
SEED          = 42
N_FOLDS       = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
# ENVIRONMENT INFO
# =============================================================================
def print_env():
    print(f"Environment  : {ENV_NAME}")
    print(f"Project dir  : {PROJECT_DIR}")
    print(f"  interim    : {INTERIM_DIR}")
    print(f"    train    : {TRAIN_INTERIM}")
    print(f"    test     : {TEST_INTERIM}")
    print(f"  processed  : {PROCESSED_DIR}")
    print(f"    crops    : {CROPS_DIR}")
    print(f"    train csv: {TRAIN_CSV}")
    print(f"    test csv : {TEST_CSV}")
    print(f"Model dir    : {MODEL_DIR}")
    print(f"OOF dir      : {OOF_DIR}")
    print(f"Submissions  : {SUBMISSION_DIR}")
    print(f"Device       : {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU          : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM         : {vram:.1f} GB")


if __name__ == "__main__":
    print_env()
