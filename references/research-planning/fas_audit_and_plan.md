# FAS Competition — Comprehensive Audit & Iteration Plan v2

## Part 1: Adherence Audit — Initial Research Plan vs Implementation

### ✅ Fully Implemented
| Plan Item | Implementation Status |
|---|---|
| **Phase 0: Deduplication** | Done. MD5 hashing found 161 duplicate groups, removed 310 images (1652→1342) |
| **Phase 0: Cross-label noise removal** | Done. 132 cross-label groups removed entirely |
| **Phase 1: ConvNeXt-Base (IN-22k)** | Done. Best model: 0.8587 mean CV |
| **Phase 1: EfficientNetV2-S** | Done. 0.7476 mean CV |
| **Phase 2: Face detection + cropping** | Done. MTCNN with 1.4× expansion, 224×224 |
| **Phase 3: Mandatory augmentations** | Done. HFlip, RandomResizedCrop, ColorJitter, GaussianBlur, RandomErasing |
| **Phase 3: FAS-specific augmentations** | Done. JPEG compression, DownscaleUpscale |
| **Phase 3: CutMix** | Done. α=1.0, 50% probability |
| **Phase 4: Focal Loss with class weights** | Done. γ=2.0, inverse-frequency α weights |
| **Phase 4: AdamW with discriminative LR** | Done. Backbone 1e-4, head 1e-3 |
| **Phase 4: Cosine annealing + warmup** | Partially. Cosine annealing done, but no explicit warmup — Stage 1 acts as implicit warmup |
| **Phase 4: Two-stage fine-tuning** | Done. 5 epochs head-only, then full fine-tuning |
| **Phase 4: EMA decay=0.999** | Done. Validated and used for best checkpoint |
| **Phase 4: Batch size 32** | Done |
| **Phase 5: Stratified 5-fold CV** | Done |
| **Phase 5: Per-class threshold optimization** | Done. OOF F1 improved 0.8713→0.8856 |
| **Phase 5: TTA (horizontal flip)** | Done |
| **Phase 5: Leaked label exploitation** | Done. 75 test images matched |

### ⚠️ Partially Implemented or Deviated
| Plan Item | What Happened |
|---|---|
| **Phase 1: CLIP ViT-B/16** | ATTEMPTED but failed catastrophically (0.1424 CV). Replaced with ResNet-50 and EfficientNet-B4 |
| **Phase 1: 3 architectures × 5 folds = 15 models** | Ended up with 4 architectures × 5 folds = 20 models, but 2 architectures (CLIP, Swin) failed completely |
| **Phase 4: Class-balanced sampling** | Coded but NOT USED — all folds trained with `use_sampler=False` |
| **Phase 3: Moiré pattern overlay** | NOT implemented — this was listed as highest-impact FAS-specific augmentation |
| **Phase 3: Color gamut reduction** | NOT implemented |
| **Phase 3: TrivialAugment** | NOT implemented |

### ❌ Skipped Entirely
| Plan Item | Impact Assessment |
|---|---|
| **Phase 0: Cleanlab noisy label detection** | HIGH IMPACT. Was described as "single most reliable method." With 1342 images, even 2-5% mislabeled = 27-67 images poisoning training |
| **Phase 0: Perceptual hashing / embedding similarity** | MEDIUM. MD5 only catches exact duplicates, not near-duplicates or same-subject-different-angle |
| **Phase 1: Auxiliary binary real/fake head** | MEDIUM. Could help model learn hierarchical structure |
| **Phase 2: Nonlinear brightness/contrast +35%** | LOW-MEDIUM |
| **Phase 3: Frequency-domain auxiliary supervision** | LOW-MEDIUM |
| **Phase 5: Pseudo-labeling on test set** | MEDIUM. Could leverage 404 test images |
| **Phase 5: Progressive resizing (224→320)** | MEDIUM. ConvNeXt at higher resolution could help |
| **Error analysis + targeted fixes** | HIGH. The plan explicitly called for this |

---

## Part 2: Deep Technical Audit — Critical Issues Found

### 🔴 CRITICAL BUG: ConvNeXt Backbone/Head Split is INVERTED

This is the single most important finding. Look at the training logs:

```
ConvNeXt-Base:
  Backbone: 3.7M params (lr=1e-4)
  Head:     83853.4K params (lr=1e-3)    ← 83.8M params as "head"?!
```

Compare with EfficientNetV2-S:
```
  Backbone: 19.8M params (lr=1e-4)
  Head:     335.4K params (lr=1e-3)      ← This is correct
```

ConvNeXt-Base has 87.6M total parameters. The classifier head for a 6-class problem should be ~6K parameters (1024×6 + bias), not 83.8M. **The parameter split logic is checking for `'classifier', 'head', 'fc'` in parameter names, but ConvNeXt's architecture in timm uses `head.fc` for the final linear layer while ALL the backbone stages also contain `head` in their parameter paths** (specifically `head.norm`, `head.fc` — the `head` module in ConvNeXt includes a LayerNorm + FC layer, but the string matching `'head'` catches parameters that shouldn't be in the head group).

**What this means**: The vast majority of ConvNeXt's backbone was trained with lr=1e-3 (10× too high) while only 3.7M params got the conservative lr=1e-4. This explains why:
- ConvNeXt still outperformed everything else — it's a strong architecture with IN-22k pretraining
- But it likely UNDERPERFORMED what it could have achieved — the high LR on backbone params causes more aggressive feature disruption

**The same bug likely affected Swin-Base** (also uses `head` in naming), which would explain why Swin completely failed to train (0.16 CV). The Swin training logs show:
```
  Backbone: 30.8M params (lr=1e-4)  
  Head:     55903.0K params (lr=1e-3)   ← 55.9M as "head" — clearly wrong
```

### 🔴 CRITICAL: Leaked Labels Are Poisonous

The submission scores reveal a damning pattern:

| Submission | Description | LB Score |
|---|---|---|
| submission_ensemble_v1.csv | Full ensemble + thresholds + 75 leaked labels | **0.68434** |
| submission_ensemble_v1_pure.csv | Full ensemble + thresholds, NO leaked labels | **0.71238** |
| sub_raw_no_thresh_no_leaked.csv | Full ensemble, NO thresholds, NO leaked labels | **0.68807** |
| sub_convnext_thresh_no_leaked.csv | ConvNeXt only + thresholds, NO leaked labels | **0.66743** |

**Key insight**: Adding leaked labels DECREASED the score by ~0.03 (0.71238 → 0.68434). This means a significant fraction of those 75 "leaked" labels are WRONG.

Why? Your data prep removed cross-label duplicates from training but the test-to-train matching was done against the cleaned set. However, the original dataset had images appearing under MULTIPLE labels (e.g., same image in both `fake_mannequin` and `fake_screen`). If a test image matches a hash that was in the CLEANED train set but the "correct" label was actually the OTHER label (from the removed duplicate), you're assigning wrong labels.

More critically: some of those 87 original train-test overlap images came from cross-label duplicate groups. For example, the EDA showed: `mannequin_042.jpg` AND `screen_088.jpg` both hash-matched `test_220.jpg`. After cleaning, you kept only one, but the test image's true label could be either class.

### 🔴 CRITICAL: Massive CV-LB Gap (0.8856 vs 0.71)

The 17-point gap between OOF CV (0.8856) and best LB score (0.71238) is catastrophic and indicates severe overfitting to the validation distribution. Contributing factors:

1. **Threshold overfitting**: The per-class threshold optimization was tuned on OOF predictions. With only 1342 samples and 6 thresholds, this optimization can easily overfit to the validation set's class distribution. Evidence: raw argmax ensemble (0.68807 LB) is VERY close to threshold-tuned (0.66743 for ConvNeXt-only), meaning thresholds provided minimal real improvement and may have hurt.

2. **Near-duplicate leakage in CV folds**: You used StratifiedKFold but NOT StratifiedGroupKFold. If near-duplicate images (same subject, similar angle but not exact MD5 match) ended up in both train and validation folds, CV scores are inflated. The plan specifically warned about this.

3. **Train-test distribution shift**: The test set likely contains attack types, imaging conditions, or subjects not well-represented in training. This is the fundamental challenge of FAS — cross-domain generalization. The research literature consistently reports 30-40% HTER degradation in cross-dataset testing.

4. **Model confidence calibration**: The ensemble produces overconfident predictions that happen to work on the training distribution but fail on shifted test data.

### 🟡 IMPORTANT: EfficientNetV2-S Loss Values Are Abnormally High

During Stage 1 head-only training, EfficientNetV2-S shows:
```
Epoch 1/5 | Train: 5.2127 | Val: 7.4800
```

For a 6-class problem with focal loss (γ=2.0), the maximum possible cross-entropy is -log(1/6) ≈ 1.79. Even with class weights, losses above ~3-4 suggest something is wrong. The focal loss modulates CE by (1-pt)^γ, which should REDUCE loss, not inflate it.

The issue: class weights are quite extreme (fake_printed gets weight 2.98) and focal loss multiplies CE by these weights. The combination of α-weighted CE × focal modulation × high initial uncertainty produces these inflated values. While technically not a bug, it makes monitoring difficult and the loss landscape more unstable.

In contrast, ConvNeXt shows reasonable losses (~1.3 at start) because the backbone/head split bug means most params get lr=1e-3 from the start, so the model adapts faster — ironically, the bug helps ConvNeXt converge better during Stage 1.

### 🟡 IMPORTANT: CLIP ViT and Swin Failures Were Diagnosable

Both CLIP ViT-B/16 and Swin-Base completely failed (0.14 and 0.16 CV respectively). The training logs show they never learned beyond random-chance predictions. This is almost certainly caused by the backbone/head parameter split bug:

- For ViT architectures, the parameter naming convention differs. `vit_base_patch16_clip_224` uses `head` for its classifier, but the string matching catches other parameters too.
- The solution in the plan was to use CLIP "as a feature extractor with a linear probe first, then full fine-tuning" — this was not followed. A simple linear probe on frozen CLIP features would have worked.

### 🟡 IMPORTANT: WeightedRandomSampler Was Never Used

The code implements `get_sampler()` but all training calls use `use_sampler=False`. The plan explicitly recommended combining WeightedRandomSampler with focal loss. For the most imbalanced class (fake_printed: 75 images vs realperson: 385 images, a 5.1× ratio), this matters significantly.

### 🟡 IMPORTANT: Validation Transform Missing Resize for 224×224

The validation transform is:
```python
val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

Since cropped images are already 224×224, this works. But there's no explicit resize as a safety measure. If any image has a different size (e.g., center-cropped fallbacks with different aspect ratios), this silently produces wrong-sized tensors. The EfficientNet-B4 correctly adds Resize(380) for its larger input.

### 🟡 NOTABLE: 102 Training Images Had No Face Detected

7.6% of training images (102/1342) fell back to center-crop. For classes like fake_unknown (which includes diverse/unusual attacks), center-cropping might capture irrelevant background instead of spoofing clues. These should have been flagged for manual review.

---

## Part 3: Submission Score Analysis

### Score Decomposition

| Component | Effect on LB Score |
|---|---|
| ConvNeXt-only with thresholds | 0.66743 (baseline single architecture) |
| 4-model ensemble without thresholds | 0.68807 (+0.02 from ensembling) |
| 4-model ensemble with thresholds | 0.71238 (+0.024 from thresholds over raw) |
| Adding leaked labels | -0.028 (HURTS performance) |

### What This Tells Us

1. **Ensembling helps modestly** (+2 points). This is lower than expected (plan predicted +2-5%), suggesting the weaker models (EfficientNetV2-S at 0.75, ResNet-50 at 0.78) are adding noise rather than diversity. The high-variance folds (ResNet-50 fold 4 at 0.60, EfficientNet-B4 fold 4 at 0.71) are particularly damaging.

2. **Threshold tuning on OOF is overfitting to CV**. The thresholds improved OOF from 0.8713→0.8856 (+1.4%) but only improved LB from 0.68807→0.71238 (+2.4%). The fact that the LB improvement is higher than CV improvement is suspicious and may be noise from the 30% public LB sample.

3. **Leaked labels are catastrophically wrong**. The -0.028 drop means roughly 5-10 of the 75 leaked labels that fall in the public 30% test split are wrong. Extrapolating to all 75, perhaps 15-25 are mislabeled — coming from those cross-label duplicates.

4. **The real model performance on unseen data is ~0.69-0.71**. This means ~30% of test predictions are wrong. For Macro F1, this likely means 1-2 classes are performing very poorly (likely fake_printed with only 75 training samples, and fake_unknown with its diverse/ambiguous nature).

---

## Part 4: Second Iteration Plan

### Priority Ranking (by expected LB impact)

#### 🔴 P0 — Fix Immediately (Expected: +5-10% LB)

**1. Fix the backbone/head parameter split for ALL architectures**

```python
def get_param_groups(model, model_key):
    """Correct parameter grouping for each architecture."""
    head_params, backbone_params = [], []
    
    if model_key == 'convnext':
        # ConvNeXt: only head.fc is the classifier
        for name, param in model.named_parameters():
            if name.startswith('head.fc'):
                head_params.append(param)
            else:
                backbone_params.append(param)
    elif model_key in ['clip_vit', 'swin']:
        # ViT/Swin: only 'head' linear layer
        for name, param in model.named_parameters():
            if name.startswith('head.') and 'head.norm' not in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
    else:
        # EfficientNet/ResNet: standard 'classifier'/'fc' naming
        for name, param in model.named_parameters():
            if any(k in name for k in ['classifier', 'fc']):
                head_params.append(param)
            else:
                backbone_params.append(param)
    
    return backbone_params, head_params
```

**Verify** by printing param counts — head should be ~6K for 6-class, backbone should be ~87M for ConvNeXt-Base.

**2. Remove ALL leaked label overrides from submissions**

Based on the evidence, leaked labels are hurting. Remove them entirely. If you want to use them, you MUST first verify each one:
- Re-examine all 87 original train-test overlaps (before cleaning)
- For any test image that hash-matches a cross-label duplicate group, mark it as UNKNOWN (use model prediction instead)
- Only use leaked labels where the train image had a single, unambiguous label

**3. Run Cleanlab on OOF predictions**

You already have OOF predictions from all 4 models. Use them:

```python
from cleanlab.filter import find_label_issues

# Use ConvNeXt OOF probs (best model)
oof_probs = convnext_oof[prob_cols].values
oof_labels = convnext_oof['label_idx'].values

issues = find_label_issues(
    labels=oof_labels,
    pred_probs=oof_probs,
    return_indices_ranked_by='self_confidence'
)
# Manually review top 50 flagged images
```

This is essentially free — you already have the predictions. Review flagged images manually (feasible at 30-75 images).

#### 🟠 P1 — High Impact Improvements (Expected: +3-7% LB)

**4. Use StratifiedGroupKFold to prevent near-duplicate leakage**

Compute perceptual hashes (pHash) or CLIP embeddings for all images, cluster similar images, and assign group IDs:

```python
import imagehash
from sklearn.model_selection import StratifiedGroupKFold

# Compute perceptual hashes
def get_phash(path):
    return str(imagehash.phash(Image.open(path), hash_size=16))

train_df['phash'] = train_df['crop_path'].apply(get_phash)

# Group images with similar hashes  
# (hamming distance < threshold = same group)
from collections import defaultdict
groups = {}
group_id = 0
for idx, row in train_df.iterrows():
    h = imagehash.hex_to_hash(row['phash'])
    found = False
    for gid, ghash in groups.items():
        if h - ghash < 8:  # hamming distance threshold
            train_df.at[idx, 'group'] = gid
            found = True
            break
    if not found:
        groups[group_id] = h
        train_df.at[idx, 'group'] = group_id
        group_id += 1

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_df, train_df['label'], train_df['group'])):
    train_df.loc[train_df.index[val_idx], 'fold'] = fold
```

**5. Retrain ConvNeXt-Base and CLIP ViT-B/16 with correct parameter split**

With the fix from P0-1:
- ConvNeXt-Base should improve from 0.8587 to potentially 0.88-0.90 CV
- CLIP ViT-B/16 should now actually work. Use lower LR (3e-5 backbone, 1e-3 head) and longer Stage 1 (10 epochs)
- This gives you the architecture diversity the plan called for (CNN + ViT)

**6. Implement proper threshold optimization with cross-validation**

The current approach optimizes thresholds on ALL OOF predictions, which overfits. Instead:

```python
# Nested CV: optimize thresholds on inner folds, evaluate on outer fold
from sklearn.model_selection import KFold

def robust_threshold_optimization(oof_probs, oof_labels, n_inner=5):
    """Optimize thresholds with nested CV to prevent overfitting."""
    kf = KFold(n_splits=n_inner, shuffle=True, random_state=42)
    all_thresh = []
    
    for train_idx, val_idx in kf.split(oof_probs):
        best_f1, best_t = 0, np.ones(6)
        for _ in range(20):
            init = np.random.uniform(0.7, 1.3, 6)
            res = minimize(macro_f1_with_thresholds, init,
                          args=(oof_probs[train_idx], oof_labels[train_idx]),
                          method='Nelder-Mead')
            if -res.fun > best_f1:
                best_f1 = -res.fun
                best_t = res.x
        all_thresh.append(best_t)
    
    # Average thresholds across folds (more robust)
    return np.mean(all_thresh, axis=0)
```

**7. Increase input resolution for ConvNeXt to 288×288 or 320×320**

ConvNeXt-Base's native test resolution is 288×288. The plan suggested progressive resizing. Re-crop all images at 320×320 (or just resize the 224 crops) and fine-tune the best ConvNeXt checkpoints for 10-15 more epochs. This is especially impactful for screen and print attacks where fine texture details matter.

#### 🟢 P2 — Moderate Impact (Expected: +1-3% LB)

**8. Enable WeightedRandomSampler for fake_printed**

fake_printed has only 75 samples. Enable the sampler:

```python
f1, preds, labels, probs = train_fold(
    fold=fold, model_key='convnext', num_epochs=50, batch_size=32,
    use_sampler=True, use_cutmix=True  # CHANGE: use_sampler=True
)
```

**9. Add moiré pattern simulation augmentation**

This was the #1 recommended FAS-specific augmentation from the plan:

```python
class MoirePattern:
    """Simulates screen moiré artifacts on real/non-screen images."""
    def __init__(self, freq_range=(5, 30), alpha_range=(0.05, 0.15)):
        self.freq_range = freq_range
        self.alpha_range = alpha_range
    
    def __call__(self, img):
        w, h = img.size
        arr = np.array(img, dtype=np.float32)
        freq = np.random.uniform(*self.freq_range)
        alpha = np.random.uniform(*self.alpha_range)
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        angle = np.random.uniform(0, np.pi)
        pattern = np.sin(2 * np.pi * freq * (xx * np.cos(angle) + yy * np.sin(angle)) / max(w, h))
        pattern = (pattern * 127.5 + 127.5).astype(np.float32)
        pattern = np.stack([pattern]*3, axis=-1)
        blended = arr * (1 - alpha) + pattern * alpha
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
```

**10. Implement pseudo-labeling on high-confidence test predictions**

Use the ensemble's predictions on test data to create pseudo-labels for samples where the model is highly confident:

```python
# Get ensemble test predictions
max_probs = test_probs.max(axis=1)
confident_mask = max_probs > 0.95  # only use very confident predictions
pseudo_df = test_df[confident_mask].copy()
pseudo_df['label'] = [IDX_TO_CLASS[p] for p in test_probs[confident_mask].argmax(axis=1)]
pseudo_df['label_idx'] = test_probs[confident_mask].argmax(axis=1)

# Add to training with lower sample weight (or just use for additional epoch)
```

**11. Better ensemble weighting**

Instead of equal-weight averaging, weight by OOF performance:

```python
# Weight inversely proportional to OOF error
weights = {
    'convnext': 0.8587,    # best
    'resnet50': 0.7793,
    'effnet_b4': 0.7806,
    'effnet': 0.7476,
}
total = sum(weights.values())
norm_weights = {k: v/total for k, v in weights.items()}

# Or better: optimize weights on OOF (with nested CV to avoid overfitting)
```

#### 🔵 P3 — Nice to Have (Expected: +0.5-1% LB)

**12. Add more TTA views**: 5-crop (center + 4 corners) + hflip = 10 views

**13. Calibrate model probabilities**: Use temperature scaling on OOF predictions before ensembling

**14. Try label smoothing (0.05-0.1)** as alternative to CutMix on some folds for ensemble diversity

**15. Review the 102 no-face-detected images**: Manually verify these got reasonable crops; consider using a different face detector (RetinaFace) as backup

---

## Execution Roadmap (2 Weeks)

### Week 1: Critical Fixes + Retraining

| Day | Task | GPU | Expected Impact |
|---|---|---|---|
| 1 | Fix backbone/head split, verify param counts for all architectures | CPU | Foundation for everything |
| 1 | Remove leaked labels, investigate which are wrong | CPU | +2-3% LB |
| 1 | Run Cleanlab on existing OOF predictions, review flagged images | CPU | +1-2% LB |
| 2 | Implement StratifiedGroupKFold with perceptual hashing | CPU | Prevents CV inflation |
| 2-3 | Retrain ConvNeXt-Base 5-fold with FIXED param split | RTX 3090, ~5h | +3-5% LB |
| 3-4 | Train CLIP ViT-B/16 5-fold with FIXED param split + linear probe first | RTX 3090, ~5h | +1-2% via ensemble |
| 4 | Implement robust threshold optimization (nested CV) | CPU | +0.5-1% LB |
| 5 | Submit and validate improvements | — | Checkpoint |

### Week 2: Optimization + Polish

| Day | Task | GPU | Expected Impact |
|---|---|---|---|
| 6 | Enable WeightedRandomSampler, add moiré augmentation | — | Setup |
| 6-7 | Retrain best 2 architectures with sampler + moiré | RTX 3090, ~8h | +1-2% LB |
| 8 | Higher resolution fine-tuning (ConvNeXt at 320×320) | RTX 3090, ~3h | +0.5-1% LB |
| 9 | Pseudo-labeling + ensemble weight optimization | RTX 3090, ~2h | +0.5-1% LB |
| 10 | Extended TTA, temperature scaling | GTX 1050Ti | +0.5% LB |
| 11-12 | Error analysis on wrong predictions, targeted fixes | Both | +1-2% LB |
| 13-14 | Final submission tuning, select best 2 submissions for private LB | — | — |

### Target Scores

| Metric | Current | Week 1 Target | Week 2 Target |
|---|---|---|---|
| OOF CV (honest) | 0.8856 (inflated) | 0.85-0.87 (honest) | 0.87-0.90 |
| Public LB | 0.71238 | 0.76-0.80 | 0.80-0.85 |

The CV will likely DROP initially when you fix the group-aware splitting and remove noise, but the CV-LB gap should shrink dramatically. A smaller, honest gap means your model generalizes better.

---

## Key Principles for Iteration 2

1. **Trust LB over CV when they disagree**. The 17-point gap means CV is lying. Every decision should be validated against LB.

2. **Fix data before fixing models**. Cleanlab + leaked label investigation + group-aware splits will have more impact than any architecture change.

3. **Simpler is often better on small data**. The ConvNeXt-only submission (0.667) vs full ensemble (0.688) shows only +2 points from ensembling 20 models. A well-tuned single architecture might beat a noisy ensemble.

4. **Don't over-optimize post-processing**. Threshold tuning and leaked labels both backfired or provided minimal gain. Focus on making the model fundamentally better, not squeezing marginal gains from brittle post-processing.

5. **The backbone/head bug fix is the single highest-ROI change**. It affects ConvNeXt (your best model), Swin (failed), and CLIP (failed). Fixing it could unlock 2 architectures and improve your best one.
