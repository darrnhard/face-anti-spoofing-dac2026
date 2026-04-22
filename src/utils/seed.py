"""
src/utils/seed.py — Reproducibility utility.

Usage:
    from src.utils.seed import set_seed
    set_seed(42)                        # full determinism
    set_seed(42, deterministic=False)   # faster, non-deterministic (benchmark mode)

Covers all RNG sources used in the training pipeline:
    - Python built-in random
    - NumPy  (augmentations, CutMix, OOF splits)
    - PyTorch CPU
    - PyTorch CUDA (single + multi-GPU)
    - cuDNN backend behaviour
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed all RNG sources for reproducibility.

    Args:
        seed:          Integer seed value. Default: 42.
        deterministic: If True, forces cuDNN into deterministic mode
                       (fully reproducible but ~10-15% slower on RTX 3090).
                       If False, enables cuDNN benchmark mode for speed
                       at the cost of run-to-run variation.
                       Use True for final training runs, False for quick
                       smoke tests or hyperparameter searches.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # covers multi-GPU, harmless on single-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    else:
        # benchmark=True lets cuDNN auto-select the fastest conv algorithm
        # per input shape. Non-deterministic but faster.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark     = True


if __name__ == "__main__":
    set_seed(42)
    print(f"Seed set to 42 (deterministic=True)")
    print(f"  torch.manual_seed   ✓")
    print(f"  torch.cuda seeds    ✓")
    print(f"  numpy.random.seed   ✓")
    print(f"  random.seed         ✓")
    print(f"  cudnn.deterministic = {torch.backends.cudnn.deterministic}")
    print(f"  cudnn.benchmark     = {torch.backends.cudnn.benchmark}")
