"""
src/inference/predict.py — Test-time inference with TTA.

Runs trained checkpoints on the test set with optional Test-Time Augmentation
(TTA). Returns per-image softmax probability arrays that are later ensembled
and threshold-tuned in the inference notebook.

Usage:
    from src.inference.predict import predict_test, INFER_BATCH
    probs = predict_test('convnext', fold=0, test_df=test_df, exp_id='exp03')
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.amp import autocast
from torch.utils.data import DataLoader

from src.data.augmentation import IMAGENET_MEAN, IMAGENET_STD
from src.data.dataset import FaceSpoofDataset
from src.models.loader import load_model_from_checkpoint
from src.models.registry import ARCH_CONFIGS
from src.utils.config import DEVICE, NUM_CLASSES


# ── Inference batch sizes (tuned for GTX 1050 Ti, 4 GB VRAM) ─────────────────
# These are deliberately conservative — OOM on inference stalls the whole run.
# RTX 3090 users can increase these; the inference code does not assume a size.
INFER_BATCH = {
    'convnext' : 4,
    'eva02'    : 4,
    'dinov2'   : 4,
    'effnet_b4': 8,
    'swinv2'   : 4,
}


def _build_tta_transforms(img_size: int):
    """
    Returns [original_transform, hflip_transform] for 2-view TTA.

    Why HFlip is the only TTA we use:
        Horizontal flip is the only geometric augmentation that is
        truly symmetric for face images — faces look realistic when
        flipped. Rotations, crops, and vertical flips introduce
        unrealistic artifacts that reduce the signal-to-noise ratio
        of the averaged probability vector.

    Why the hflip is applied BEFORE Resize+ToTensor:
        torchvision applies transforms left-to-right. Flipping a PIL
        image before converting to a tensor is equivalent to flipping
        the pixel array and is slightly faster than tensor-level flip.
        The transform order is: HFlip → Resize → ToTensor → Normalize,
        which is consistent with how val_transform is structured in
        src/data/augmentation.py.

    Parameters
    ----------
    img_size : int — target resolution for Resize

    Returns
    -------
    transforms : list[torchvision.transforms.Compose] — length 2
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

    return [base_tf, hflip_tf]


def predict_test(model_key: str, fold: int, test_df,
                 exp_id: str = 'exp03',
                 tta_views: int = 2,
                 model_dir=None) -> np.ndarray:
    """
    Run TTA inference on the full test set for one (model_key, fold) checkpoint.

    Process
    -------
    1. Load the best checkpoint for (model_key, fold, exp_id) from disk
    2. For each TTA view, run a full forward pass over the test set
    3. Average softmax probabilities across TTA views
    4. Delete the model and free GPU memory before returning

    Why we delete the model after each call:
        The GTX 1050 Ti has only 4 GB VRAM. Holding 20 models in memory
        simultaneously (4 models × 5 folds) would require ~400 GB. Loading
        one at a time and immediately freeing is the only practical approach.

    Parameters
    ----------
    model_key  : str          — key into ARCH_CONFIGS
    fold       : int          — fold index (0–4)
    test_df    : pd.DataFrame — must have 'crop_path' column
    exp_id     : str          — experiment ID used when saving checkpoints
    tta_views  : int          — 2 = original + hflip; 1 = no TTA
    model_dir  : Path | None  — defaults to MODEL_DIR from config

    Returns
    -------
    probs : np.ndarray shape (N, NUM_CLASSES)
        Averaged softmax probabilities across all TTA views.
    """
    cfg       = ARCH_CONFIGS[model_key]
    img_size  = cfg['img_size']
    batch     = INFER_BATCH[model_key]

    transforms = _build_tta_transforms(img_size)[:tta_views]

    model     = load_model_from_checkpoint(model_key, fold, exp_id, model_dir)
    all_probs = np.zeros((len(test_df), NUM_CLASSES), dtype=np.float32)

    for tf in transforms:
        loader = DataLoader(
            FaceSpoofDataset(test_df, transform=tf, is_test=True),
            batch_size=batch, shuffle=False, num_workers=2, pin_memory=True,
        )

        # Collect per-batch probs and concatenate — avoids index arithmetic
        batch_probs = []
        with torch.no_grad():
            for images in loader:
                images = images.to(DEVICE)
                with autocast("cuda"):
                    logits = model(images)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                batch_probs.append(probs)

        all_probs += np.concatenate(batch_probs, axis=0)

    all_probs /= len(transforms)   # average over TTA views

    del model
    torch.cuda.empty_cache()
    return all_probs
