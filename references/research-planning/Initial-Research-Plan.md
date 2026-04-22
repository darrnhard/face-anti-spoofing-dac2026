# Winning a 6-class face anti-spoofing competition with 1,500 images

**The single highest-impact strategy is domain-specific transfer learning combined with simulated spoofing augmentation, stratified K-fold ensembling, and per-class threshold optimization for Macro F1.** This combination addresses every constraint simultaneously: small data (transfer learning), class imbalance (focal loss + threshold tuning), noisy labels (EMA + cleanlab), and limited compute (efficient architectures on T4). The approach mirrors how every recent FAS competition winner — including the CVPR 2024 1st-place MTFace team — won: not through exotic architectures, but through intelligent augmentation, robust validation, and disciplined ensembling. Below is a complete, priority-ranked playbook built from the latest FAS research (2023–2026), small-dataset deep learning best practices, and documented competition-winning solutions.

---

## Phase 0: Data cleaning and deduplication come first

Before any modeling, **fix the data**. With only ~1,652 images (1,474 unique), even a handful of mislabeled or duplicated samples can corrupt 1–2% of training signal — equivalent to hundreds of images in a larger dataset.

**Deduplication** should use perceptual hashing (via the `imagehash` library) or cosine similarity of pretrained embeddings. Extract features with a frozen CLIP ViT-B/16 or EfficientNet, compute pairwise similarity, and flag pairs above 0.95 similarity with conflicting labels. Cleanlab's `Datalab` module can automate this. For the ~178 duplicates (1,652 minus 1,474 unique), decide whether to keep the majority-voted label or remove the sample entirely.

**Noisy label detection** with Cleanlab is the single most reliable method. Train a baseline model, collect out-of-fold predicted probabilities via 5-fold CV, then run `cleanlab.find_label_issues()`. This requires zero hyperparameters and has been shown to find label errors even in ImageNet. On a dataset this small, expect to flag 2–5% of samples. Review flagged samples manually — at ~30–75 images, this is feasible in under an hour. **EMA (exponential moving average with decay=0.999) provides implicit robustness to remaining noise** throughout training, as demonstrated in a 2024 study showing EMA alone beats many specialized noisy-label methods on CIFAR-10N/100N benchmarks with ~40% human annotation noise.

---

## Phase 1: Architecture and backbone selection for maximum transfer

**The backbone choice matters more than any other single decision** for a small dataset. Research consistently shows three top-tier options for face anti-spoofing on limited data:

**Tier 1 — ConvNeXt-Base (ImageNet-21k pretrained)** ranks as the best general-purpose fine-tuning backbone in the "Battle of the Backbones" benchmark. Its inductive biases (locality, translation equivariance) help with small data, while its modern design competes with ViTs. The CVPR 2024 FAS challenge 2nd-place solution (SeaRecluse) used ConvNeXt V2 successfully.

**Tier 1 — EfficientNetV2-S (ImageNet-21k or Noisy Student pretrained)** was the architecture of choice for every DFDC top-5 finisher. Selim Seferbekov's 1st-place DFDC solution used EfficientNet-B7 with Noisy Student pretraining. For a T4 GPU with 16GB VRAM, EfficientNetV2-S at **224×224 or 260×260** fits comfortably with batch size 32. EfficientNet-B4 is also viable at 380×380.

**Tier 2 — CLIP ViT-B/16** offers the best cross-domain generalization. The FLIP paper (2023) demonstrated that CLIP initialization alone dramatically improves face anti-spoofing generalizability. For a 6-class task with unusual categories like "fake_mannequin" and "fake_unknown," CLIP's semantic understanding of diverse concepts may provide an edge. Use it as a feature extractor with a linear probe first, then full fine-tuning.

**Practical recommendation:** Train all three as part of a 5-fold CV ensemble. On a T4 GPU, each fold of EfficientNetV2-S trains in ~15 minutes for 30 epochs at 224×224. Budget approximately **5 hours total** for training all folds of all three architectures — well within the 30hr/week limit.

For the classification head, use a simple structure: `GlobalAvgPool → Dropout(0.3) → Linear(features, 6)`. Adding a hidden layer of 512 units with BatchNorm can marginally help. For multi-task learning, optionally add an auxiliary **binary real/fake head** — this forces the model to learn a hierarchical representation (first real vs fake, then attack type), which is better aligned with the underlying data structure.

---

## Phase 2: Preprocessing pipeline built on FAS domain knowledge

**Face detection and tight cropping with context padding** is the single most important preprocessing step. Every successful FAS system crops faces before classification. Use **RetinaFace** (more accurate) or **MTCNN** (faster, used by DFDC 1st place) for detection, then expand the bounding box by **1.3–1.5× on each side** to capture critical context: paper edges for print attacks, screen bezels for replay attacks, mask boundaries for 3D masks. The DFDC 1st-place solution added 30% of face crop size from each side. Silent-Face-Anti-Spoofing uses 1.5× expansion.

**Resize to 224×224** for EfficientNetV2-S and ConvNeXt, or **260×260** for slightly better performance. Higher resolutions (384×384) provide marginal gains but substantially increase memory and training time on T4.

**Color space: stay with RGB.** While older literature advocated YCbCr and HSV, modern CNNs with sufficient pretraining implicitly learn relevant color transformations. Research by Sun et al. confirmed RGB is superior to HSV and YCbCr for CNN-based FAS. However, consider **nonlinear brightness/contrast adjustment (+35%)** as preprocessing, which amplifies visual differences between real and spoof faces — particularly print artifacts and screen moiré patterns.

**Optional frequency-domain auxiliary input:** The Silent-Face-Anti-Spoofing system uses Fourier spectrum as auxiliary supervision during training (not at inference). Compute the FFT magnitude spectrum of face crops and use it as a secondary training signal. This teaches the model to recognize frequency differences between attack types — print attacks lack high-frequency detail, screen attacks exhibit pixel-grid moiré patterns, while real faces and 3D masks have rich high-frequency skin texture.

---

## Phase 3: Domain-specific augmentation is the #1 competition differentiator

The CVPR 2024 FAS challenge 1st-place solution won specifically because of **Simulated Physical Spoofing Clues (SPSC)** and **Simulated Digital Spoofing Clues (SDSC)** — augmentations that generate realistic spoofing artifacts on live images. This is the technique with the highest impact-to-effort ratio.

**Mandatory augmentations (apply to all classes):**

- Horizontal flip (probability 0.5)
- RandomResizedCrop (scale 0.8–1.0)
- ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) — the CVPR 2024 winner specifically noted ColorJitter simulates print attack clues
- Light Gaussian noise (σ=0.01–0.03) — simulates sensor differences between capture devices
- Random JPEG compression (quality 30–95) — critical because many spoof images undergo compression
- TrivialAugment as the automated policy (zero hyperparameters, outperforms RandAugment on small data)

**FAS-specific augmentations (high-impact):**

- **Moiré pattern overlay** on real images to simulate screen replay artifacts (SPSC approach)
- **Color gamut reduction** to simulate print attack color loss
- **Gaussian blur** (σ=0.5–2.0) to simulate loss of fine texture in printed/screen photos
- **Downscale then upscale** (factor 2–4×) to simulate resolution degradation in attack media
- **Random rectangular patch overlay** (CutOut) — the DFDC 1st-place solution used structured dropout of face parts during training

**CutMix (α=1.0, probability 0.5)** is the strongest regularizing augmentation for this setting. It outperforms MixUp for classification and was used by the CVPR 2024 FAS 2nd-place solution. **Do not combine CutMix with label smoothing** — both modify labels and create conflicting training signals. Choose one. For noisy data, CutMix is the better choice because it preserves local spatial features.

**Augmentations to avoid:** Excessive geometric distortions (large rotations >30°, aggressive perspective transforms) that destroy subtle texture patterns. Also avoid augmentations that simulate the exact artifacts you're trying to detect — e.g., don't add moiré patterns to fake_screen images, as this confuses the model about what constitutes a screen attack.

---

## Phase 4: Training configuration for maximum Macro F1

**Loss function: Focal Loss with per-class weights.** Focal loss (γ=2.0) down-weights easy/well-classified examples and focuses on hard ones — critical for a 6-class problem where "realperson" likely dominates. Set α weights inversely proportional to class frequency. The CVPR 2024 FAS winner used protocol-specific loss weight ratios of up to **5:1** (live:fake).

**Optimizer: AdamW** with weight_decay=1e-4 and base learning rate=1e-4 for backbone, 10× (1e-3) for the classification head. This discriminative learning rate strategy prevents catastrophic forgetting of pretrained features while allowing the head to adapt quickly.

**Learning rate schedule: Cosine annealing with linear warmup** (warmup for 3–5 epochs, then cosine decay over remaining epochs). This is the dominant schedule across all competition winners surveyed. Total training: 30–50 epochs with early stopping (patience=10–15, monitoring validation Macro F1).

**Two-stage fine-tuning protocol:**
1. **Stage 1 (5–10 epochs):** Freeze backbone, train only classification head with lr=1e-3
2. **Stage 2 (20–40 epochs):** Unfreeze all layers, discriminative learning rates (backbone 1e-5 to 1e-4, head 1e-3), cosine annealing

**EMA with decay=0.999** throughout training. This is essentially free (+negligible compute) and provides **0.5–2% improvement** plus robustness to noisy labels. PyTorch implementation: `torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))`.

**Batch size: 32** with gradient accumulation if needed. The T4's 16GB VRAM handles EfficientNetV2-S at 224×224 with batch size 32 easily.

**Class-balanced sampling:** Use PyTorch's `WeightedRandomSampler` to ensure each batch is approximately class-balanced. This complements focal loss — the sampler ensures minority classes appear often enough, while focal loss ensures the model focuses on hard examples within each class.

---

## Phase 5: Validation, ensembling, and inference optimization

**Stratified 5-fold cross-validation is non-negotiable** for a dataset this small. Each fold gives ~1,180 training and ~295 validation images. Use `StratifiedKFold(n_splits=5, shuffle=True)` from scikit-learn. If near-duplicates remain after deduplication, use `StratifiedGroupKFold` to prevent the same subject/scene from appearing in both train and validation. This is the approach used by the Iceberg Classifier winner (1,604 images — nearly identical dataset size) and every DFDC top-5 solution.

**Ensemble strategy — the approach that consistently wins:**

1. Train 3 architectures × 5 folds = **15 models total**
2. For each test image, collect 15 softmax probability vectors
3. Average all 15 vectors (arithmetic mean)
4. Apply per-class threshold optimization (see below)
5. Expected improvement from ensembling: **2–5% Macro F1** over best single model

For additional diversity without additional architectures, vary the random seed, augmentation policy, or input resolution across folds. The CVPR 2024 FAS 3rd-place solution used **self-supervised pretraining (simMIM)** on training data before fine-tuning — this adds another axis of diversity if time permits.

**Per-class threshold optimization is critical for Macro F1.** After obtaining averaged softmax probabilities from the ensemble, Macro F1 is threshold-dependent and the default argmax is rarely optimal. The procedure:

1. On validation data, collect out-of-fold predictions from all 15 models
2. For each class independently, sweep the decision threshold from 0.1 to 0.9 in steps of 0.01
3. Choose thresholds that maximize per-class F1, subject to the constraint that every sample gets exactly one predicted class
4. In practice, use `scipy.optimize.minimize` with Nelder-Mead to jointly optimize all 6 thresholds

The MTFace team (CVPR 2024 FAS 1st place) and multiple Kaggle winners emphasize that **post-processing threshold optimization often provides 1–3% Macro F1 improvement** over naive argmax — a massive gain at the top of a leaderboard.

**Test-Time Augmentation (TTA):** Apply horizontal flip + original image (2 views), average softmax outputs. This reliably provides **1–3% improvement at zero training cost**. Use the `ttach` library for easy implementation. More aggressive TTA (adding 5-crop or rotations) provides diminishing returns but is worth testing.

---

## The 12-day implementation roadmap

Given the T4 GPU constraint (30hrs/week, ~12 days remaining), here is the exact execution order ranked by impact per hour invested:

| Day | Task | GPU hours | Expected Macro F1 gain |
|-----|------|-----------|----------------------|
| 1 | Data cleaning: dedup, cleanlab, manual review of flagged samples | 1 | +2–5% |
| 1–2 | Face detection + cropping pipeline (RetinaFace, 1.4× expansion) | 1 | +5–10% |
| 2–3 | Baseline: EfficientNetV2-S, 5-fold CV, basic augmentation | 3 | Establish baseline (~0.70–0.80) |
| 3–4 | Add focal loss + class-balanced sampling + EMA | 0.5 | +2–4% |
| 4–5 | Add FAS-specific augmentations (moiré, blur, JPEG, CutMix) | 3 | +3–6% |
| 5–6 | Train ConvNeXt-Base, 5-fold CV, same pipeline | 4 | +1–3% (via ensemble) |
| 6–7 | Train CLIP ViT-B/16, 5-fold CV, same pipeline | 4 | +1–2% (via ensemble) |
| 7–8 | Ensemble all 15 models + per-class threshold optimization | 1 | +2–4% |
| 8–9 | TTA implementation + pseudo-labeling on test set (optional) | 2 | +1–3% |
| 9–10 | Progressive resizing experiments (train at 224, finetune at 320) | 3 | +0.5–1% |
| 11–12 | Error analysis, targeted fixes, final submission tuning | 3 | +1–2% |

**Total estimated GPU hours: ~25.5** (within 30hr/week budget). **Target final Macro F1: 0.85–0.92** depending on data quality and test set difficulty.

---

## What distinguishes attack types at the feature level

Understanding the physical differences between attack types informs both augmentation and architecture choices:

| Attack type | Key discriminative features | Best detection approach |
|---|---|---|
| **fake_printed** | Paper texture, halftone dots, reduced color gamut, flat depth, paper edges visible | Frequency analysis (missing high-freq), color histogram analysis |
| **fake_screen** | Moiré patterns, pixel grid artifacts, screen glare/reflections, flat depth, bezel edges | Frequency domain (periodic moiré peaks in FFT), specular reflection detection |
| **fake_mask** | 3D depth present but unnatural skin texture, mask boundary edges, rigid expression | Texture micro-analysis, edge detection at mask boundaries |
| **fake_mannequin** | 3D depth present, smooth plastic/resin surface, no skin micro-texture, unnatural eyes | Skin texture analysis, eye region features |
| **fake_unknown** | Variable — may include partial attacks, paper cutouts, or novel methods | Ensemble diversity, broad feature learning |

The key insight: **print and screen attacks are "flat" (no depth) while mask and mannequin attacks are "3D" (have depth).** A two-level hierarchy — first classify flat vs 3D vs real, then subclassify — may improve accuracy for the hardest confusions. FaceShield (2025) found frequent confusion between flexible/rigid masks and between print/replay attacks, confirming this hierarchical structure.

---

## Conclusion: three techniques that matter most

After synthesizing the latest FAS research, small-dataset best practices, and competition-winning strategies, three techniques stand far above the rest in expected impact for this specific competition:

**First, domain-specific augmentation that simulates spoofing artifacts** — the CVPR 2024 FAS winner won specifically because SPSC and SDSC generated realistic attack simulations from live images, expanding effective training set diversity by an order of magnitude. This matters more than architecture choice.

**Second, aggressive ensembling of diverse architectures with per-class threshold optimization** — combining CNNs (ConvNeXt, EfficientNet) with a ViT (CLIP) across 5-fold CV yields 15 models whose averaged predictions are substantially more robust than any single model. Threshold tuning on top of this directly optimizes the competition metric.

**Third, data cleaning before anything else** — with only 1,474 unique images, fixing even 30 mislabeled samples (2%) has the same impact as adding 30 perfectly labeled images. Cleanlab makes this nearly automatic. Combined with EMA for residual noise robustness, this ensures the model learns from correct signal rather than memorizing errors.