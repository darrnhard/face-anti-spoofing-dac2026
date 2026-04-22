# FAS Training Deep-Dive: Research-Backed Plan for training.ipynb v2

*Companion to fas_audit_and_plan.md — focused exclusively on model training improvements*

---

## 1. CDCN Analysis: Should We Add It?

### What CDCN Is
CDCN (Central Difference Convolutional Network) replaces standard convolutions with Central Difference Convolutions that capture both intensity AND gradient information. It was purpose-built for FAS and won 1st place at ChaLearn Multi-Modal FAS Challenge @CVPR2020. The key innovation is the θ parameter (default 0.7) that controls the blend between vanilla convolution and gradient-based features.

### Verdict: **YES, but with caveats**

**Arguments FOR adding CDCN:**
- It is the only FAS-domain-specific architecture in consideration — all others are general-purpose ImageNet backbones
- CDCN captures fine-grained texture patterns (paper grain, screen moiré, mask boundaries) that generic CNNs learn less efficiently
- It provides **maximum ensemble diversity** — CDCN learns fundamentally different features (gradient-based) vs ConvNeXt (intensity-based) vs CLIP (semantic)
- Small model (~2M params) so trains fast, and its depth-map prediction paradigm adds a different inductive bias
- Published PyTorch implementations exist (github.com/ZitongYu/CDCN)

**Arguments AGAINST / Caveats:**
- CDCN was designed for binary FAS (live vs spoof). Your task is 6-class. You'll need to modify the architecture: replace the depth-map regression head with a 6-class classification head, or use CDCN as a feature extractor + linear classifier
- CDCN expects 256×256 input and outputs a 32×32 depth map — you'd need to adapt for your 224×224 pipeline
- It has NO ImageNet pretraining. On 1342 images, training from scratch is risky. Solution: use it as an **auxiliary feature extractor** alongside pretrained models, not as a standalone
- The original implementation requires depth map ground truth for supervision, which you don't have. Use binary depth (1.0 for real, 0.0 for fake) as the simplified supervision signal

**Recommended implementation:**
```python
# Approach: CDCN as auxiliary feature extractor in ensemble
# 1. Train CDCN with simplified binary depth supervision
# 2. For 6-class, add a classification head on CDCN features
# 3. Weight it lower in ensemble (0.5-0.7× weight vs ConvNeXt)
```

**Expected impact:** +1-2% ensemble F1 from diversity, minimal training cost (~30 min per fold on RTX 3090)

---

## 2. Architecture Selection for Maximum Ensemble Diversity

### Current Architecture Inventory (6 attempted, 4 working)

| Architecture | Type | Pretrain | Mean CV | Status | Feature Type |
|---|---|---|---|---|---|
| ConvNeXt-Base | Modern CNN | IN-22k | 0.8587 | ✅ Best | Local + mid-range spatial |
| EfficientNetV2-S | Efficient CNN | IN-1k | 0.7476 | ✅ Weak | Multi-scale efficient |
| EfficientNet-B4 | Efficient CNN | IN-1k | 0.7806 | ✅ OK | Multi-scale, higher res |
| ResNet-50 | Classic CNN | IN-1k | 0.7793 | ✅ OK | Residual hierarchical |
| CLIP ViT-B/16 | Transformer | CLIP 400M | 0.1424 | ❌ Failed (bug) | Global semantic |
| Swin-Base | Transformer | IN-22k | 0.1622 | ❌ Failed (bug) | Shifted window attention |

### The Diversity Problem
Your current working ensemble has **near-zero architectural diversity** — all 4 models are standard CNNs with similar inductive biases. The research is clear: ensemble gains come primarily from **model diversity**, not model count.

The "Battle of the Backbones" (2023) benchmark and "Which Backbone to Use" (TMLR 2025) papers both confirm:
- ConvNeXt-Base (IN-22k) is the best general fine-tuning backbone
- Transformers underperform CNNs on small datasets UNLESS properly configured
- The biggest ensemble gains come from mixing CNN + Transformer architectures

### Recommended Architecture Selection (5 models for ensemble)

**Slot 1: ConvNeXt-Base (IN-22k)** — KEEP, your best model. Fix the param split bug.
- `convnext_base.fb_in22k` via timm
- Expected post-fix CV: 0.88-0.92

**Slot 2: EVA-02-Base (CLIP pretrained)** — NEW, replaces failed CLIP ViT
- `eva02_base_patch14_224.mim_in22k` via timm
- Why: EVA-02 combines masked image modeling + CLIP initialization. It's specifically designed for robust transfer learning and addresses the exact failure mode of your CLIP ViT attempt. EVA-02 has better fine-tuning stability than raw CLIP ViT.
- Key config: Use layer-wise learning rate decay (LLRD) with factor 0.75, lower base LR (5e-5 backbone, 5e-4 head), and longer Stage 1 (10 epochs)
- Expected CV: 0.84-0.88

**Slot 3: DINOv2-Base (with registers)** — NEW, self-supervised ViT
- `vit_base_patch14_reg4_dinov2.lvd142m` via timm
- Why: A recent ICCV2025 FAS workshop paper (Feng et al., 2025) specifically demonstrated DINOv2 with registers for face anti-spoofing, showing it captures fine-grained differences between live and spoofed faces. DINOv2's self-supervised features are complementary to both supervised CNNs and CLIP-style models.
- Key config: Use as frozen feature extractor first (linear probe), then cautious fine-tuning with LLRD 0.8
- Expected CV: 0.82-0.87

**Slot 4: EfficientNet-B4 (380px)** — KEEP but lower priority
- Provides resolution diversity (380 vs 224)
- Expected CV: 0.80-0.85 with fixed pipeline

**Slot 5: CDCN** — NEW, domain-specific
- Trained from scratch with binary depth + 6-class head
- Provides fundamentally different feature type (gradient-based)
- Expected CV: 0.70-0.78 standalone, but adds unique signal to ensemble

### Why Drop EfficientNetV2-S and ResNet-50?

Both overlap heavily with ConvNeXt in feature space (all standard IN-1k CNNs) and perform significantly worse. The "Battle of the Backbones" paper shows that adding more models of the same type provides diminishing returns. Replacing them with a ViT (EVA-02) and self-supervised model (DINOv2) adds far more ensemble diversity.

### Architecture Diversity Matrix

| | Texture/Local | Semantic/Global | Gradient/Freq | Multi-Scale |
|---|---|---|---|---|
| ConvNeXt-Base | ✅✅✅ | ✅ | ✅ | ✅✅ |
| EVA-02-Base | ✅ | ✅✅✅ | ✅ | ✅ |
| DINOv2-Base | ✅✅ | ✅✅ | ✅ | ✅✅ |
| EfficientNet-B4 | ✅✅ | ✅ | ✅ | ✅✅✅ |
| CDCN | ✅✅✅ | ❌ | ✅✅✅ | ✅ |

This gives excellent coverage across feature types.

---

## 3. Data Augmentation Improvements

### Current Augmentation Pipeline (what you have)

```
RandomHorizontalFlip(0.5) → RandomResizedCrop(224, 0.8-1.0) →
ColorJitter(0.2,0.2,0.2,0.1) → GaussianBlur(0.2) →
JPEGCompression(0.3) → DownscaleUpscale(0.2) →
ToTensor → Normalize → RandomErasing(0.2)
+ CutMix at batch level (50%)
```

### What's Missing (from research)

**A) Moiré Pattern Simulation (CRITICAL — SPSC approach from CVPR 2024 1st place)**

The CVPR 2024 FAS Challenge 1st place team (MTFace/He et al.) won specifically because of SPSC augmentation. Their paper shows that simulating moiré patterns on live images dramatically improves detection of screen replay attacks.

```python
class MoirePattern:
    """SPSC-style moiré simulation from CVPR2024 FAS 1st place."""
    def __init__(self, freq_range=(5, 40), alpha_range=(0.03, 0.15)):
        self.freq_range = freq_range
        self.alpha_range = alpha_range

    def __call__(self, img):
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        freq = np.random.uniform(*self.freq_range)
        alpha = np.random.uniform(*self.alpha_range)
        angle = np.random.uniform(0, np.pi)

        y, x = np.mgrid[0:h, 0:w]
        pattern = np.sin(2 * np.pi * freq *
                        (x * np.cos(angle) + y * np.sin(angle)) / max(h, w))
        pattern = ((pattern + 1) / 2 * 255).astype(np.float32)
        pattern = np.stack([pattern] * 3, axis=-1)

        result = arr * (1 - alpha) + pattern * alpha
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
```

Apply with `p=0.2` on non-screen classes only.

**B) Color Gamut Reduction (simulates print attack clues)**

```python
class ColorGamutReduction:
    """Simulates the color loss from printing."""
    def __init__(self, saturation_range=(0.3, 0.7)):
        self.saturation_range = saturation_range

    def __call__(self, img):
        factor = np.random.uniform(*self.saturation_range)
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(factor)
```

Apply with `p=0.15` on non-print classes.

**C) Screen Bezel Overlay (simulates visible phone/screen edges)**

```python
class ScreenBezelOverlay:
    """Adds dark border simulating a phone screen bezel."""
    def __init__(self, width_range=(5, 20)):
        self.width_range = width_range

    def __call__(self, img):
        arr = np.array(img)
        w = np.random.randint(*self.width_range)
        arr[:w, :] = arr[:w, :] * 0.3  # top
        arr[-w:, :] = arr[-w:, :] * 0.3  # bottom
        arr[:, :w] = arr[:, :w] * 0.3  # left
        arr[:, -w:] = arr[:, -w:] * 0.3  # right
        return Image.fromarray(arr.astype(np.uint8))
```

Apply with `p=0.1` on non-screen classes.

**D) Improved CutMix Strategy**

Current: CutMix with 50% probability on every batch.
Research finding: The CVPR 2024 2nd place solution noted that CutMix should NOT be combined with label smoothing. However, you should also consider **class-aware CutMix** — avoid mixing within the same superclass (e.g., don't mix fake_printed with fake_screen since both are "flat" attacks).

**E) AugMax / Adversarial Augmentation (for robustness)**

Recent FAS work shows that adversarial training-style augmentations improve cross-domain generalization. A lightweight version:

```python
# After standard augmentations, randomly select the "worst" aug
# from a pool for each image (the one that maximizes loss)
# This forces the model to be robust to the hardest transformations
```

### Recommended v2 Augmentation Pipeline

```python
train_transform_v2 = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0))], p=0.2),
    T.RandomApply([JPEGCompression(quality_range=(20, 95))], p=0.3),
    T.RandomApply([DownscaleUpscale(scale_range=(2, 5))], p=0.2),
    T.RandomApply([MoirePattern()], p=0.15),           # NEW
    T.RandomApply([ColorGamutReduction()], p=0.1),     # NEW
    T.RandomApply([ScreenBezelOverlay()], p=0.08),     # NEW
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    T.RandomErasing(p=0.25, scale=(0.02, 0.2)),       # slightly more aggressive
])
```

---

## 4. Fine-Tuning Strategy Improvements

### Current Approach Analysis

Your current 2-stage approach:
- Stage 1: Freeze backbone, train head only (5 epochs, no CutMix)
- Stage 2: Unfreeze all, discriminative LR (backbone 1e-4, head 1e-3), cosine annealing

**Problems identified from training logs:**

1. **Stage 1 is too short and losses are abnormally high.** EfficientNetV2-S Stage 1 ends with val loss ~6.0 and val F1 ~0.19. The head hasn't converged at all before Stage 2 begins. This means Stage 2 starts with a badly initialized head, forcing both backbone and head to co-adapt simultaneously.

2. **The transition from Stage 1 to Stage 2 is too abrupt.** Going from fully frozen to fully unfrozen causes instability. The training logs show val F1 fluctuating wildly in early Stage 2 epochs.

3. **No learning rate warmup in Stage 2.** The cosine annealing starts at peak LR immediately. Research consistently shows warmup is critical for stability.

4. **The backbone/head param split bug** (already covered in audit) means ConvNeXt and ViTs get wrong LRs.

### Recommended: 3-Stage Gradual Unfreezing Protocol

```python
def train_fold_v3(fold, model_key, num_epochs=60, batch_size=32):
    """
    3-stage training with gradual unfreezing.

    Stage 1 (10 epochs): Head only, higher LR, no CutMix
      - Goal: get a well-calibrated classification head
      - LR: 1e-3 with cosine annealing within stage

    Stage 2 (15 epochs): Unfreeze last 1/3 of backbone + head
      - Goal: adapt high-level features to FAS domain
      - LR: backbone_top 5e-5, head 5e-4, with 3-epoch warmup
      - CutMix ON at 30% probability

    Stage 3 (remaining epochs): Full fine-tuning
      - Goal: end-to-end optimization
      - LR: layer-wise decay (LLRD 0.85 for CNNs, 0.75 for ViTs)
      - Base LR: 3e-5, head: 3e-4
      - CutMix ON at 50% probability
      - 5-epoch warmup, then cosine decay
    """
```

### Layer-Wise Learning Rate Decay (LLRD)

For ConvNeXt-Base, implement proper LLRD instead of 2-group discriminative LR:

```python
def get_convnext_param_groups(model, base_lr=3e-5, head_lr=3e-4, decay=0.85):
    """Layer-wise LR decay for ConvNeXt-Base."""
    param_groups = []

    # ConvNeXt has 4 stages: stages.0, stages.1, stages.2, stages.3
    # + stem, downsample layers, and head
    num_stages = 4
    for i in range(num_stages):
        stage_lr = base_lr * (decay ** (num_stages - 1 - i))
        for name, param in model.named_parameters():
            if f'stages.{i}.' in name:
                param_groups.append({
                    'params': param, 'lr': stage_lr,
                    'weight_decay': 1e-4
                })

    # Stem gets lowest LR
    stem_lr = base_lr * (decay ** num_stages)
    for name, param in model.named_parameters():
        if 'stem.' in name:
            param_groups.append({'params': param, 'lr': stem_lr})

    # Head gets highest LR
    for name, param in model.named_parameters():
        if name.startswith('head.fc'):
            param_groups.append({'params': param, 'lr': head_lr})

    return param_groups
```

For ViT-based models (EVA-02, DINOv2), use a more aggressive decay (0.75):

```python
def get_vit_param_groups(model, base_lr=2e-5, head_lr=5e-4, decay=0.75):
    """LLRD for ViT. Research shows ViTs benefit more from LLRD than CNNs."""
    # ViT-Base has 12 blocks
    num_layers = 12
    param_groups = []
    for i in range(num_layers):
        layer_lr = base_lr * (decay ** (num_layers - 1 - i))
        for name, param in model.named_parameters():
            if f'blocks.{i}.' in name:
                param_groups.append({'params': param, 'lr': layer_lr})

    # Patch embed + cls token + pos embed get lowest LR
    for name, param in model.named_parameters():
        if any(k in name for k in ['patch_embed', 'cls_token', 'pos_embed']):
            param_groups.append({
                'params': param,
                'lr': base_lr * (decay ** num_layers)
            })

    # Head gets highest LR
    for name, param in model.named_parameters():
        if 'head' in name and 'blocks' not in name:
            param_groups.append({'params': param, 'lr': head_lr})

    return param_groups
```

### BatchNorm Handling

Critical detail from Keras/TF docs and best practices: when fine-tuning, BatchNorm layers should stay in eval mode to preserve pretrained statistics. Your current code doesn't handle this:

```python
# Add to training loop for fine-tuning stages:
def set_bn_eval(module):
    """Keep BatchNorm in eval mode during fine-tuning."""
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        module.eval()

# During training:
model.train()
model.apply(set_bn_eval)  # Override BN to stay in eval
```

Note: ConvNeXt uses LayerNorm, not BatchNorm, so this mainly helps EfficientNet and ResNet.

---

## 5. Hyperparameter Optimization

### Learning Rate

**Current:** backbone 1e-4, head 1e-3 (flat across all architectures)

**Problem:** One LR doesn't fit all. Research shows:
- ConvNeXt with IN-22k pretrain: optimal fine-tune LR is 5e-5 to 1e-4 for backbone
- CLIP-initialized ViTs: need much lower LR, 1e-5 to 5e-5 backbone (CLIP fine-tuning paper shows this)
- DINOv2: 2e-5 to 5e-5 backbone (frozen feature extraction often beats full fine-tuning on small data)

**Recommended per-architecture LRs:**

| Architecture | Backbone LR | Head LR | LLRD factor | Warmup epochs |
|---|---|---|---|---|
| ConvNeXt-Base | 5e-5 | 5e-4 | 0.85 | 3 |
| EVA-02-Base | 2e-5 | 5e-4 | 0.75 | 5 |
| DINOv2-Base | 1e-5 | 1e-3 | 0.80 | 5 |
| EfficientNet-B4 | 1e-4 | 1e-3 | N/A (2-group) | 3 |
| CDCN | 1e-3 (from scratch) | — | N/A | 5 |

### Epochs and Early Stopping

**Current:** 50 epochs, patience 10

**Analysis from training logs:** Many models hit best F1 at epoch 38-50 — they're still improving when training ends. But the val loss keeps decreasing while val F1 fluctuates, suggesting the model is becoming more confident on easy samples without improving on hard ones.

**Recommended:**
- **Increase to 60-80 epochs** for ConvNeXt (your best model is still improving at epoch 50)
- **Patience 15** — too aggressive early stopping killed ResNet-50 fold 4 at epoch 18
- **Monitor val Macro F1, not val loss** — these diverge on imbalanced datasets
- Use **cosine annealing with warm restarts** (CosineAnnealingWarmRestarts with T_0=20, T_mult=2) instead of single cosine decay. This gives the model multiple chances to escape local minima.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-7
)
```

### Batch Size

**Current:** 32 for most models, 16 for EfficientNet-B4

**Analysis:** With RTX 3090 24GB, you can go larger:
- ConvNeXt-Base at 224: batch_size=64 (uses ~14GB)
- EfficientNet-B4 at 380: batch_size=24 (uses ~18GB)
- EVA-02-Base at 224: batch_size=48 (uses ~16GB)

Larger batch sizes with the same LR typically help on small datasets by reducing gradient noise. If you increase batch size, scale LR linearly (batch 64 → multiply LR by 2).

### Weight Decay

**Current:** 1e-4 for all

**Recommended:** Keep 1e-4 for backbone, but use **0** or **1e-5** for head parameters. The newly initialized head needs freedom to learn; weight decay constrains it unnecessarily during initial training.

### Focal Loss γ Parameter

**Current:** γ=2.0 (default from paper)

**Analysis:** γ=2.0 is aggressive — it strongly down-weights "easy" examples. On a 6-class problem with only 75 samples in the smallest class, you need the model to learn from EVERY example, including "easy" ones. Consider **γ=1.0** which is a gentler version that still helps with class imbalance but doesn't discard signal.

Alternatively, try **Poly-1 loss** which has been shown to outperform both CE and focal loss on small datasets:

```python
class Poly1CrossEntropyLoss(nn.Module):
    """Poly-1 loss: CE + epsilon * (1 - pt)"""
    def __init__(self, num_classes=6, epsilon=1.0, weight=None):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.weight = weight

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, weight=self.weight, reduction='none')
        pt = F.one_hot(labels, self.num_classes).float()
        pt = (pt * F.softmax(logits, dim=-1)).sum(dim=-1)
        poly1 = ce + self.epsilon * (1 - pt)
        return poly1.mean()
```

### EMA Decay

**Current:** 0.999

**This is fine** for most settings. However, with 1342 images and batch 32, you get ~42 steps per epoch. Over 50 epochs that's ~2100 steps. At decay=0.999, the EMA only "sees" the last ~1000 steps significantly. Consider **0.9995** for longer effective memory, or use **a schedule** that starts at 0.99 and increases to 0.9999.

### Gradient Clipping

**Missing from your code entirely.** Add gradient clipping to prevent explosive gradients, especially important when unfreezing:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Stochastic Weight Averaging (SWA)

Not currently used. For the last 10-20% of training, SWA can improve generalization by averaging weights across the flat region of the loss landscape:

```python
from torch.optim.swa_utils import AveragedModel, SWALR

# After epoch 40 (out of 60):
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=1e-5)

for epoch in range(40, 60):
    train_one_epoch(...)
    swa_model.update_parameters(model)
    swa_scheduler.step()

# Update batch norm statistics
torch.optim.swa_utils.update_bn(train_loader, swa_model)
```

---

## 6. Additional Research-Backed Improvements

### A) Supervised Contrastive Learning (SupCon) Pre-training

The CVPR 2024 Snapshot Spectral FAS Challenge 1st place solution used supervised contrastive learning. This creates a feature space where same-class images cluster together and different-class images separate, BEFORE the classification head is trained:

```python
# Stage 0 (NEW): SupCon pre-training (5-10 epochs)
# - Use SupCon loss on augmented pairs
# - This creates better feature representations
# - Then add classification head and proceed with normal training
```

This is especially powerful for the `fake_unknown` class which is inherently diverse.

### B) Test-Time Training (TTT)

Recent FAS research (Zhou et al., CVPR 2024) proposes test-time domain generalization for FAS. The idea: during inference, briefly adapt the model to each test sample using self-supervised objectives. This directly addresses the train-test distribution shift.

### C) Mixup Between Specific Class Pairs

Instead of random CutMix, create synthetic samples specifically for confused class pairs. From the confusion matrix:
- `fake_printed ↔ fake_unknown` (high confusion)
- `fake_mask ↔ fake_unknown` (high confusion)
- `realperson ↔ fake_printed` (concerning confusion)

Train additional epochs with mixup samples from these specific pairs.

### D) Multi-Scale Training

Currently all models see 224×224. For the final training:
1. Random resize: randomly pick from [192, 224, 256, 288] per batch
2. This forces the model to learn scale-invariant features

### E) Knowledge Distillation from Ensemble to Single Model

After building the 5-model ensemble, distill its predictions into a single ConvNeXt-Base:
```python
# Train ConvNeXt-Base using soft labels from ensemble
# This captures the ensemble's "knowledge" in a single model
# Often performs better than any single model in the ensemble
```

---

## Summary: Priority-Ranked Training Changes

| Priority | Change | Expected Impact | Effort |
|---|---|---|---|
| 🔴 P0 | Fix backbone/head param split | +5-10% for ConvNeXt, enables ViTs | 30 min |
| 🔴 P0 | Implement proper LLRD per architecture | +1-3% | 1 hour |
| 🔴 P0 | Add 3-epoch warmup to Stage 2 | +0.5-1% stability | 15 min |
| 🔴 P0 | Add gradient clipping (max_norm=1.0) | Prevents instability | 5 min |
| 🟠 P1 | 3-stage gradual unfreezing | +1-2% | 2 hours |
| 🟠 P1 | Add moiré + color gamut augmentations | +1-3% | 1 hour |
| 🟠 P1 | Replace CLIP ViT with EVA-02-Base | +2-4% ensemble | 4 hours training |
| 🟠 P1 | Add DINOv2-Base to ensemble | +1-3% ensemble | 4 hours training |
| 🟠 P1 | Increase epochs to 60-80, patience to 15 | +0.5-1% | Minimal |
| 🟠 P1 | Use CosineAnnealingWarmRestarts | +0.5-1% | 10 min |
| 🟢 P2 | Add CDCN to ensemble | +1-2% diversity | 3 hours |
| 🟢 P2 | Try Poly-1 loss or focal with γ=1.0 | +0.5-1% | 30 min |
| 🟢 P2 | Per-architecture batch size optimization | +0.5% | Minimal |
| 🟢 P2 | SWA for last 20% of training | +0.5-1% | 30 min |
| 🟢 P2 | Knowledge distillation from ensemble | +1-2% single model | 2 hours |
| 🔵 P3 | SupCon pre-training stage | +1-2% | 3 hours |
| 🔵 P3 | Multi-scale training | +0.5-1% | 1 hour |
| 🔵 P3 | Class-pair targeted mixup | +0.5% | 1 hour |
