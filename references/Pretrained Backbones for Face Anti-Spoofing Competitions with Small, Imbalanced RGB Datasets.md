# Pretrained Backbones for Face Anti-Spoofing Competitions with Small, Imbalanced RGB Datasets

## Executive Summary

Recent face anti-spoofing (FAS) challenges and cross-dataset studies show a clear trend: transformer-based foundation models adapted with lightweight parameter-efficient fine-tuning now outperform classic CNN backbones (ResNet, EfficientNet, MobileNet) on diverse attacks and low-data regimes. Among these, CLIP ViT backbones with LoRA adaptation (FoundPAD), self-supervised face-security-specific ViT backbones (FSFM), and ViT backbones with statistical adapters (S-Adapter) achieve state-of-the-art generalization on multiple public PAD/FAS benchmarks, especially under single-source (low-data) protocols similar to small competition datasets.[^1][^2][^3][^4][^5]

For a 6-class RGB FAS competition with only 1.4k training images and moderate class imbalance, the most evidence-backed and practical choice is a **ViT-B/16 backbone from CLIP or FSFM, adapted using a parameter-efficient method (LoRA or S-Adapter) and trained with a 6-way classifier head**. CNN backbones such as ResNet-34/50 or MobileNetV3 remain strong, easier-to-train baselines but no longer represent the best achievable performance on modern FAS benchmarks.[^6][^7][^8]

The recommended ranking below balances (1) measured performance on public FAS benchmarks and challenges, (2) robustness to attack diversity and domain shift, (3) performance in low-data regimes, and (4) training practicality.


## 1. Problem Setting and Requirements

### 1.1 Competition constraints

- Dataset: ~1464 training images, 404 test images (RGB).
- Task: 6-class classification (realperson, fake_printed, fake_screen, fake_mask, fake_mannequin, fake_unknown).
- Metric: macro F1 over 6 classes (thus minority classes matter).
- Imbalance: majority/minority ratio ≈ 3.9×.

These properties imply:

- **Low-data regime** → strong need for transfer learning and good inductive biases to avoid overfitting.
- **Multi-class spoof types** → the model must capture fine-grained cues for different attacks, not only live vs spoof.
- **Class imbalance + macro F1** → backbone must support stable optimization under re-weighting/oversampling; no special architectural requirement here.

Consequently, the best backbone should:

- Be pretrained on very large image or face datasets.
- Have demonstrated strong cross-dataset FAS/PAD performance.
- Work well under single-source or few-shot regimes.
- Be trainable with only a small number of updated parameters (adapters/LoRA) to mitigate overfitting.


## 2. Evidence from Surveys and Benchmarks

### 2.1 Surveys: shift from handcrafted to deep and then to transformers

Multiple recent surveys document the evolution of FAS from LBP-like handcrafted features to CNNs and now transformer/foundation-model approaches.[^9][^10][^11][^12][^13]

Key points:

- Deep CNNs (ResNet, MobileNet, EfficientNet) dominated 2017–2020, especially under intra-dataset protocols.[^14][^13]
- Pixel-wise auxiliary supervision (depth/reflectance/mask maps) and patch-based CNNs (e.g., CDCN, Auto-FAS, LGSC) improved robustness but still struggled with cross-dataset generalization.[^15][^8][^16]
- Recent works emphasize **cross-domain FAS**, **unified physical/digital attacks**, and **foundation-model-style backbones** (ViT, Swin, CLIP, FSFM) as the new frontier.[^17][^18][^7][^3][^4][^1]

These surveys consistently report that generalization to new domains/attacks is the main challenge; this is critical for small competitions where the test distribution may differ from train.


### 2.2 Challenge winners: CVPR 2020–2024 FAS challenges

#### CelebA-Spoof Challenge 2020 (ECCV 2020)

- Dataset: large-scale CelebA-Spoof (625k images, 10k subjects).[^14]
- Top-3 solutions all used **ensembles of CNN-based models** (AENet, ResNet variants, CDCN-style networks), along with patch-based training and sophisticated fusion.[^19]
- First-place solution (ZOLOZ) combined:
  - A multi-task FOCUS framework (dual ResNet18-CDC encoder + U-Net style decoder with pixel-wise spoof cues).[^19]
  - AENet (multi-task CNN), a binary ResNet-18 specialized on hard cases, an attack-type classifier, and a noise-print camera-type model.[^19]

These results show that **ResNet-style backbones with task-specific designs were competitive in 2020**, but they required heavy ensembles and specialized components, and they predate the transformer boom.


#### Wild Face Anti-Spoofing Challenge 2023 (CVPR 2023)

- Dataset: WFAS, a large-scale in-the-wild FAS dataset with 1.38M images, 469k subjects and 17 2D/3D PAs.[^1]
- Baseline comparison (Protocol 1: known-type, face bounding box input):
  - CNN baselines: ResNet-50, PatchNet, MobileNet-style, CDCN, DC-CDN, etc.[^1]
  - Best CNN baseline: **MaxViT** (hybrid CNN/ViT) with ACER ≈ 6.58%, better than ResNet-50 (7.71%) and CDCN++ variants (≈ 8–11%).[^1]
- Top-3 challenge solutions (relaxed face scale) all use **transformer architectures**:
  - **China Telecom (1st, ACER ≈ 1.60%)**: two-stage ViT pipeline: DINO-style self-supervised pretraining, then supervised fine-tuning on FAS; heavy augmentations and AdamW + cosine LR.[^1]
  - **Meituan (2nd)**: self-supervised MoCoV2 on unlabeled data, then SwinV2-Huge/Tiny fine-tuning and distillation (SwinV2-H → SwinV2-T).[^1]
  - **NetEase (3rd)**: ConvNeXt trained first, then MaxViT trained on soft labels with focal+triplet loss.[^1]

Key takeaway: On a very diverse in-the-wild FAS dataset, **ViT/Swin/MaxViT backbones substantially beat CNN baselines**, even when the CNNs include sophisticated pixel-wise supervision.[^1]


#### Face Anti-Spoofing Challenge 2024 (CVPR 2024, Unified Physical-Digital)

- Track uses UniAttackData with both physical (print, display, masks) and digital (deepfake, adversarial) attacks.[^7][^20]
- One published solution evaluated **MobileNetV3-Spoof** (MobileNetV3 backbone with custom spoof head) and systematically studied pre-processing (face detection/alignment) and pretraining; they found that carefully tuned MobileNetV3 with proper alignment and augmentation can achieve strong ACER while remaining lightweight.[^20][^7]
- Another top workshop paper proposes **simulated spoofing clues augmentation (SPSC/SDSC)** around **ResNet-50 and SwinV2 backbones**, and provides public checkpoints (full_resnet50, full_swin_v2_base, mobilenet_v3_small, shufflenetv2).[^21]

This challenge confirms that **both transformer backbones (SwinV2) and strong CNNs (ResNet-50, MobileNetV3) remain competitive**, but the best generalization tends to come from transformer-based or hybrid approaches.


## 3. Foundation-Model and Transformer Backbones for FAS

### 3.1 FSFM: Face Security Foundation Model (ViT-based)

FSFM is a self-supervised **face security foundation model** that pretrains a vanilla ViT on large collections of real face images using a combination of masked image modeling and instance discrimination ("3C" objectives).[^22][^4][^5]

- Backbone: vanilla ViT (e.g., ViT-B) serving as a universal face-security encoder.[^4][^5]
- Training: CRFR-P masking emphasizes intra-region consistency and inter-region coherency on faces, while an ID head enforces local-to-global correspondence.[^5][^4]
- Downstream: authors evaluate on **cross-domain face anti-spoofing**, deepfake detection, and diffusion-based facial forgery.
- Reported results: FSFM transfers better than supervised ImageNet pretraining, generic visual/facial self-supervised methods, and even task-specialized SOTA FAS/forgery detectors on 10 public datasets.[^22][^4][^5]

Implications for the competition:

- FSFM is explicitly optimized to capture **real-face semantics and spoof cues**, then reused across FAS tasks.
- It is particularly suited to cross-domain settings where training data is limited, mirroring a small competition dataset.
- The backbone is a standard ViT, so adding a 6-class head and fine-tuning via partial freezing or adapter-based updates is straightforward.

Limitations:

- FSFM is relatively recent (CVPR 2025); available code and models are outside mainstream libraries but the project site offers models and masking demos.[^5]
- Most reported experiments focus on binary live vs spoof; adaptation to 6-class spoof categorization will require additional training and careful loss design.


### 3.2 FoundPAD: CLIP ViT + LoRA for PAD

FoundPAD adapts CLIP’s ViT-B/L image encoder to PAD using **rank-stabilized LoRA** and a binary classifier.[^2]

Design:

- Backbone: CLIP ViT-B (86M params, patch 16) and ViT-L (≈300M params).[^2]
- Adaptation: rsLoRA applied to query and value projection matrices in MSA layers, with original CLIP weights frozen; LoRA rank r=8, α=8, dropout 0.4.[^2]
- Header: simple fully-connected binary classifier trained with cross-entropy.[^2]
- Training data: multiple public PAD datasets (MSU-MFSD, CASIA-FASD, Replay-Attack, OULU-NPU, CelebA-Spoof) and synthetic SynthASpoof.[^2]
- Setting: cross-dataset PAD with **triple-source, double-source, single-source (low data) protocols**, plus synthetic→authentic.

Key results (HTER↓, AUC↑):[^2]

- **Triple-source (e.g., O&C&I→M, etc.)**: FoundPAD ViT-L achieves avg HTER ≈ 9.67% and AUC ≈ 96.60%, improving over the second-best method CF-PAD (HTER ≈ 11.57%, AUC ≈ 94.68) and over ViT-FS/FE baselines by 4–14 AUC points depending on architecture.[^2]
- **Double-source (M&I→C/O)**: FoundPAD ViT-L reaches HTER ≈ 4.67% / AUC ≈ 99.22% on CASIA-FASD, surpassing CNN-based CF-PAD and CIFAS as well as ViT-FS/FE; similarly strong on OULU-NPU.[^2]
- **Single-source (hardest, small data)**: For ViT-L, FoundPAD improves average HTER by about 6.54 percentage points versus the best prior method, reducing worst-case HTER from ~34% to ~24%.[^2]
- **SynthASpoof→authentic**: FoundPAD ViT-L is competitive with recent synthetic-data PAD methods and close to the winner of SynFacePAD-2023 on average HTER, while requiring only synthetic training.[^2]

Relevance to small competition:

- FoundPAD is explicitly designed for **low data availability** and **cross-dataset generalization**, demonstrating large gains in single-source settings most similar to a small competition dataset.[^2]
- It uses **parameter-efficient fine-tuning (LoRA)**, which is ideal when the dataset is small; only adapters and the head are updated, reducing overfitting.
- CLIP weights are widely available via PyTorch/timm/HuggingFace, making implementation practical.

Challenges:

- FoundPAD is formulated as binary PAD; for 6-class FAS, the classification head must be extended to 6 outputs and the loss redefined (e.g., class-weighted cross-entropy or focal loss), while still keeping LoRA adaptation.
- CLIP’s pretraining is generic rather than face-specific; FSFM may have an advantage in purely face-centric tasks, but FoundPAD empirically shows strong PAD generalization.


### 3.3 S-Adapter: ViT + Statistical Adapters for FAS

S-Adapter inserts **spoof-aware statistical adapters** into a pre-trained ViT backbone and only trains these adapters plus the classifier, under the Efficient Parameter Transfer Learning paradigm.[^3][^23]

Principal ideas:[^23][^3]

- Use a standard pre-trained ViT (e.g., ViT-B/16) as backbone.
- Add S-Adapter modules to gather local discriminative and statistical information via localized token histograms (texture-inspired, similar to LBP).[^3][^23]
- Introduce Token Style Regularization (TSR) to reduce domain style variance by regularizing Gram matrices across tokens from different domains.[^23][^3]
- Keep main ViT weights frozen; train adapters and classifier only.

Reported results:

- On multiple FAS benchmarks (e.g., SiW, OULU-NPU, CASIA-SURF CeFA, CelebA-Spoof), S-Adapter + ViT **outperforms previous SOTA**, including CNNs, vanilla ViT fine-tuning, and other adapter-based baselines, in both zero-shot and few-shot cross-domain settings.[^24][^3][^23]
- The design explicitly addresses domain shift via statistical tokens rather than only high-level semantics.[^3][^23]

Relevance:

- S-Adapter is well-suited to **few-shot or limited-sample** training because only small adapter modules are learned.
- The texture-like statistical tokens are aligned with what is known about spoof cues (moiré, printing noise, edge artifacts).[^13][^3]

Limitations:

- Code is not as widely used as CLIP/LoRA; adapting and debugging under competition time pressure may be harder.
- Original experiments focus on binary PAD, although multi-class adaptation is straightforward at the classifier level.


## 4. CNN and Lightweight Backbones

### 4.1 ResNet family (18/34/50) and CDCN

ResNet backbones have been extensively used for FAS.

- A ResNet34 model with ImageNet pretraining, fine-tuned for FAS, achieved ≈99.8% accuracy and very low EER (0.2%) on the NUAA dataset using transfer learning, demonstrating that even moderate-depth ResNets can nearly saturate simpler datasets.[^6]
- CDCN/CDCNet, built upon ResNet-like blocks with **central difference convolution** (CDC), achieved state-of-the-art performance on several FAS benchmarks (CASIA-FASD, OULU-NPU, SiW) via pixel-wise depth supervision and patch-level augmentation.[^15]
- The CelebA-Spoof challenge winners use ResNet-18 and ResNet-like encoders for FOCUS and other components.[^19]
- A 2025 adaptive fusion FAS network still uses ResNet-18 as backbone, combined with dynamic convolution, bottleneck attention, and domain-adversarial alignment to improve cross-domain performance.[^25]

Overall, ResNets remain strong and relatively simple to train; however, recent FAS challenges and cross-dataset studies show that **transformers and foundation models are surpassing ResNet-based architectures** when powerful pretraining and adaptation are applied.[^3][^1][^2]


### 4.2 MobileNetV3-Spoof and other lightweight CNNs

MobileNet-family backbones are attractive for edge deployment and small datasets due to their low parameter count.

- The CVPR 2024 FAS workshop paper "Assessing the Performance of Efficient FAS Detection" evaluates **MobileNetV3-Spoof** (MobileNetV3 backbone + spoof head) on UniAttackData, showing that **proper pre-processing (face detection/alignment) significantly improves detection of complex physical and digital attacks**.[^7][^20]
- A 2024 paper integrates CBAM attention and CDC into an improved MobileNetV3, achieving >97% accuracy on NUAA and Replay-Attack with minimal computational cost.[^26][^27]
- Auto-FAS uses NAS to design lightweight pixel-wise supervised FAS networks optimized for mobile constraints, outperforming manually designed lightweight CNNs on several benchmarks.[^8][^16][^28]

These results show that **MobileNetV3/Auto-FAS-style backbones can be very competitive at much lower FLOPs**, which is relevant if the competition has strict runtime limits. However, in terms of absolute state-of-the-art performance on large and diverse FAS benchmarks, they typically lag behind transformer/foundation-model backbones.


## 5. Ranking of Backbones for the Given Competition

The ranking below assumes:

- You can use standard GPUs (e.g., 12–24 GB) and common libraries (PyTorch, timm, HuggingFace).
- The main constraint is **data scarcity**, not FLOPs.
- You can invest some engineering effort into adapters/LoRA, but not full-blown multi-model ensembles.

If your hardware or time is very limited, Section 6 discusses how to simplify.

### 5.1 Top tier (Tier A): Foundation-model ViT backbones with parameter-efficient adaptation

**1) CLIP ViT-B/16 or ViT-L/14 with LoRA, following FoundPAD-style training**

Justification:

- FoundPAD shows that **adapting CLIP’s ViT image encoder with LoRA yields strong cross-dataset PAD performance, especially in single-source (low-data) settings**, improving average HTER by up to 6.54 percentage points over prior SOTA and 4–9 points over other ViT baselines.[^2]
- LoRA adaptation is specifically designed to make FMs usable under low data, updating only small matrices while preserving the generalization learned from massive pretraining.[^2]
- CLIP weights and code are easy to obtain, and LoRA implementations (e.g., PEFT, loralib) are mature.

For your competition:

- Use CLIP ViT-B/16 if GPU memory is modest; ViT-L/14 if you have strong hardware and can tolerate slower training.
- Freeze the main backbone; train LoRA adapters + a 6-class classifier head.
- Apply strong data augmentation (color jitter, blur, cutout/mixup) and class-weighted/focal loss to handle imbalance, but do not rely on heavy pixel-wise supervision since you only have static RGB images.

Given the evidence and practicality, **CLIP ViT-B/16 + LoRA is the most balanced “best backbone” choice for your small, imbalanced 6-class FAS competition.**[^2]


**2) FSFM ViT-B (Face Security Foundation Model)**

Justification:

- FSFM uses self-supervised pretraining on large-scale face data to build a **face-specific foundation model**, and experiments show it transfers better than supervised/other self-supervised pretraining for cross-domain FAS and related tasks.[^4][^22][^5]
- The backbone is a vanilla ViT, making it simple to add a classifier for 6 classes and fine-tune via partial freezing or small adapters.

For your competition:

- FSFM may capture more nuanced live-face cues (e.g., skin detail, 3D structure) than CLIP’s generic pretraining, which could help separate realperson from various spoof types.
- However, tooling and documentation are less mainstream than CLIP; integrating FSFM might take more time.

If you can quickly integrate FSFM weights, **FSFM ViT-B with a shallow 6-class head (and possibly adapters) is a very strong alternative, likely on par with or better than CLIP ViT-B in FAS-specific generalization**, but with higher integration risk.[^4][^5]


**3) ViT-B/16 with S-Adapter (statistical tokens)**

Justification:

- S-Adapter explicitly targets FAS domain shift by modeling texture statistics via token histograms and style regularization on top of a pre-trained ViT, while freezing backbone weights.[^23][^3]
- Experimental results show **superior cross-domain and few-shot FAS performance compared to CNNs, vanilla ViT fine-tuning, and other adapter baselines**.[^23][^3]

For your competition:

- Architecturally attractive for a small dataset; only S-Adapter layers + classifier are trained.
- Implementation is more involved than LoRA; the model is less “off-the-shelf” than CLIP+LoRA.

Overall, S-Adapter ranks just behind CLIP+LoRA and FSFM for this competition because of engineering complexity, not because of lack of potential.


### 5.2 Strong tier (Tier B): Transformer/CNN hybrids and strong CNNs

**4) SwinV2-Tiny/Base pre-trained on ImageNet-21k**

- SwinV2 is a hierarchical transformer; Meituan’s CVPR 2023 WFAS 2nd-place solution used self-supervised MoCoV2 + SwinV2-Huge/Tiny with distillation, achieving ACER ≈ 2.22% on a very challenging in-the-wild dataset.[^1]
- The CVPR 2024 solution "Joint Physical-Digital Facial Attack Detection" offers public models for SwinV2 and ResNet-50 backbones used on UniAttackData, including full-image and face-crop variants.[^21]

SwinV2 pretraining and architecture yield strong generalization, but without face-specific FSFM or CLIP-level massive pretraining, they are marginally less attractive than Tier A for low-data 6-class FAS.


**5) ConvNeXt / MaxViT pre-trained on ImageNet-21k**

- NetEase’s CVPR 2023 WFAS solution uses ConvNeXt + MaxViT in a two-stage training strategy (ConvNeXt generates soft labels; MaxViT trained with focal+triplet on those labels) and ranks 3rd with ACER ≈ 2.55%.[^1]
- Baseline comparisons show MaxViT clearly outperforming ResNet-50 and CDCN++ on WFAS under consistent settings.[^1]

These architectures are strong but somewhat heavy and more complex to tune; for a small dataset, they may overfit without careful regularization and are less straightforward than CLIP+LoRA.


### 5.3 Solid baselines (Tier C): Classic CNNs and lightweight models

**6) ResNet-34/50 with ImageNet pretraining (optionally with CDC)**

- Proven workhorse for FAS; strong results on CASIA-FASD, OULU-NPU, NUAA, etc.[^25][^6][^19]
- Many competition entries still include ResNet-18/34/50 models as part of ensembles (CelebA-Spoof, UniAttackData).[^21][^19]

For your task, a simple ResNet-34/50 with strong augmentation, class-balanced loss, and possibly additional features (central difference convolution, attention) will provide a robust baseline but will likely not surpass Tier A/B transformer-based solutions on diverse attacks and unseen test conditions.


**7) MobileNetV3-Spoof / Auto-FAS-type lightweight models**

- Excellent when FLOPs/latency are constrained; CVPR 2024 studies demonstrate MobileNetV3-Spoof can perform surprisingly well when combined with careful pre-processing and training.[^20][^7]
- Auto-FAS and later lightweight FAS networks searched by NAS show that lightweight architectures can reach competitive ACER with much lower compute.[^16][^28][^8]

For your competition, unless there is a strict runtime limit on the evaluation server, these models are better treated as additional ensemble members rather than primary backbones.


## 6. Practical Recommendations for Your Setup

### 6.1 If you want a single main model

1. **Use CLIP ViT-B/16 as backbone, with LoRA + 6-class classifier (FoundPAD-style).**[^2]
   - Initialize from `openai/clip-vit-base-patch16` or equivalent.
   - Insert LoRA modules into attention (Q/V) and optionally MLP layers.
   - Train only LoRA params + classifier; freeze main backbone.
   - Loss: class-weighted cross-entropy or focal loss to emphasize minority attack types; optionally add label smoothing.
   - Use strong augmentation (RandomResizedCrop, horizontal flip, ColorJitter, Gaussian blur, slight affine) to simulate diverse PAs.

2. If time and effort allow, **try FSFM ViT-B as a drop-in replacement for CLIP**, keeping the same LoRA + head structure.

Given current evidence and practicality, this is the highest-value approach for your competition.


### 6.2 If you can train 2–3 models and ensemble

- Primary: **CLIP ViT-B/16 + LoRA + 6-way head**.
- Secondary: **ResNet-34 or ResNet-50 (ImageNet pre-trained) with CDC or simple 6-way head**, heavy augmentation.
- Optional third: **MobileNetV3-Spoof** or **SwinV2-Tiny** if compute allows.

Fusion can be done by simple average or rank-based weighting (as used in CelebA-Spoof winners), optionally tuned on a validation split if available.[^19]


### 6.3 Training details that matter more than backbone differences

Across challenges and papers, some training tricks often contribute as much as backbone choice:[^8][^7][^19][^1]

- **Patch/region-based training**: cropping facial regions or patches to force the network to focus on spoof artifacts (print edges, moiré, mask seams).
- **Data augmentation oriented to spoof cues**: ColorJitter for print attacks, moiré augmentation for screen attacks, cutouts and local blur for occlusions.[^21][^1]
- **Balanced sampling / WeightedRandomSampler** to compensate class imbalance within each batch.[^7]
- **Margin-based or focal losses** to enhance minority classes and hard samples.
- **Face detection and alignment** preprocessing, which consistently improves MobileNet and other lightweight backbones on complicated datasets.[^20][^7]

These techniques should be applied regardless of backbone.


## 7. Limitations and Uncertainties

- Most state-of-the-art works (FSFM, FoundPAD, S-Adapter) report results on **binary PAD/FAS**; while their backbones are suitable for multi-class heads, there is no published benchmark directly on 6-class FAS classification.
- There is no direct head-to-head comparison between FSFM and FoundPAD on the same PAD protocols; the ranking between them relies on the strength of each paper’s reported improvements and on practicality rather than a single common leaderboard.[^4][^2]
- Competition-specific constraints (image resolution, hidden test distribution, runtime limits) may slightly change the optimal choice; for example, very low resolutions may reduce the benefit of large ViT models, and strict latency limits may favor MobileNet/Auto-FAS.

Despite these caveats, the convergence of challenge results and recent literature strongly supports **ViT-based foundation-model backbones with parameter-efficient adaptation (LoRA/adapters) as the best starting point** for your competition setting.[^3][^4][^1][^2]

---

## References

1. [Wild Face Anti-Spoofing Challenge 2023: Benchmark and Results](https://arxiv.org/abs/2304.05753) - ...commercial sensors, including 17
presentation attacks (PAs) that encompass both 2D and 3D forms. ...

2. [FoundPAD: Foundation Models Reloaded for Face Presentation Attack
  Detection](http://arxiv.org/pdf/2501.02892.pdf) - Although face recognition systems have seen a massive performance enhancement
in recent years, they ...

3. [S-Adapter: Generalizing Vision Transformer for Face Anti-Spoofing with Statistical Tokens](https://arxiv.org/abs/2309.04038v1) - Face Anti-Spoofing (FAS) aims to detect malicious attempts to invade a face recognition system by pr...

4. [FSFM: A Generalizable Face Security Foundation Model via Self ...](https://arxiv.org/abs/2412.12032) - Abstract page for arXiv paper 2412.12032: FSFM: A Generalizable Face Security Foundation Model via S...

5. [FSFM: A Generalizable Face Security Foundation Model via Self ...](https://fsfm-3c.github.io) - FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learn...

6. [Face Anti-spoofing based on Deep Residual Network](https://www.ijrar.org/papers/IJRAR22D2786.pdf)

7. [CVPR 2024 Open Access Repository](https://openaccess.thecvf.com/content/CVPR2024W/FAS2024/html/Luevano_Assessing_the_Performance_of_Efficient_Face_Anti-Spoofing_Detection_Against_Physical_CVPRW_2024_paper.html)

8. [AUTO-FAS: SEARCHING LIGHTWEIGHT NETWORKS FOR FACE ANTI-SPOOFING](https://oulurepo.oulu.fi/bitstream/handle/10024/27690/nbnfi-fe2020112092121.pdf;jsessionid=3CE85BEB6CEDE8BAF927B1B3D05F4A68?sequence=1)

9. [Deep Learning for Face Anti-Spoofing: A Survey](https://arxiv.org/abs/2106.14948) - Face anti-spoofing (FAS) has lately attracted increasing attention due to its
vital role in securing...

10. [A Survey on Anti-Spoofing Methods for Facial Recognition with RGB Cameras of Generic Consumer Devices](https://www.mdpi.com/2313-433X/6/12/139/pdf) - The widespread deployment of facial recognition-based biometric systems has made facial presentation...

11. [A Survey on Deep Learning-based Face Anti-Spoofing](https://www.nowpublishers.com/article/Details/SIP-20240053) - Publishers of Foundations and Trends, making research accessible

12. [A Comprehensive Survey on the Evolution of Face Anti‐spoofing ...](https://www.sciencedirect.com/science/article/abs/pii/S0925231224017636) - This paper provides a comprehensive review of the state-of-the-art works published over the past dec...

13. [A Survey on Anti-Spoofing Methods for Facial Recognition with RGB ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321190/) - This survey thoroughly investigates facial Presentation Attack Detection (PAD) methods that only req...

14. [[2106.14948] Deep Learning for Face Anti-Spoofing: A Survey - ar5iv](https://ar5iv.labs.arxiv.org/html/2106.14948) - This paper covers the most recent and advanced progress of deep learning on four practical FAS proto...

15. [Dual-Cross Central Difference Network for Face Anti-Spoofing](https://arxiv.org/pdf/2105.01290.pdf) - ...and local detailed representation enhancement.
Furthermore, a novel Patch Exchange (PE) augmentat...

16. [AUTO-FAS: SEARCHING LIGHTWEIGHT NETWORKS FOR FACE ANTI-SPOOFING](http://jultika.oulu.fi/files/nbnfi-fe2020112092121.pdf)

17. [Advancing Cross-Domain Generalizability in Face Anti-Spoofing: Insights,
  Design, and Metrics](http://arxiv.org/pdf/2406.12258.pdf) - ...also
leverages the advantages of measuring uncertainty, allowing for enhanced
sampling during tra...

18. [Unified Physical-Digital Attack Detection Challenge](https://arxiv.org/html/2404.06211v1) - Face Anti-Spoofing (FAS) is crucial to safeguard Face Recognition (FR)
Systems. In real-world scenar...

19. [arXiv:2102.12642v2 [cs.CV] 26 Feb 2021](https://arxiv.org/pdf/2102.12642.pdf)

20. [GitHub - Inria-CENATAV-Tec/Assessing-Efficient-FAS-CVPR2024: Code for the paper "Assessing the Performance of Efficient Face Anti-Spoofing Detection Against Physical and Digital Presentation Attacks" at the 5th Face Anti-Spoofing Challenge and Workshop @ CVPR2024](https://github.com/Inria-CENATAV-Tec/Assessing-Efficient-FAS-CVPR2024) - Code for the paper "Assessing the Performance of Efficient Face Anti-Spoofing Detection Against Phys...

21. [GitHub - Xianhua-He/cvpr2024-face-anti-spoofing-challenge: Accepted by CVPR Workshop 2024](https://github.com/Xianhua-He/cvpr2024-face-anti-spoofing-challenge) - Accepted by CVPR Workshop 2024. Contribute to Xianhua-He/cvpr2024-face-anti-spoofing-challenge devel...

22. [FSFM: A Generalizable Face Security Foundation Model via Self-Supervised
  Facial Representation Learning](https://arxiv.org/html/2412.12032v2) - ...distillation. These
three learning objectives, namely 3C, empower encoding both local features an...

23. [S-Adapter: Generalizing Vision Transformer for Face Anti-Spoofing ...](https://arxiv.org/html/2309.04038v2) - We propose a novel Statistical Adapter (S-Adapter) that gathers local discriminative and statistical...

24. [S-Adapter: Generalizing Vision Transformer for Face Anti-Spoofing ...](https://www.semanticscholar.org/paper/e04770a7d487c2e850f36623e220906a05cc72fd) - S-Adapter: Generalizing Vision Transformer for Face Anti-Spoofing With Statistical Tokens ... Domain...

25. [A high-performance adaptive fusion network for face anti-spoofing ...](https://www.nature.com/articles/s41598-025-21461-0) - This paper takes ResNet-18 as the backbone network and innovatively introduces the bottleneck attent...

26. [Lightweight Face Anti-spoofing for Improved MobileNetV3](https://www.clausiuspress.com/article/14608.html)

27. [Lightweight Face Anti-spoofing for Improved MobileNetV3](https://www.clausiuspress.com/assets/default/article/2024/12/26/article_1735230967.pdf)

28. [[PDF] auto-fas: searching lightweight networks for face anti-spoofing](https://oulurepo.oulu.fi/bitstream/10024/27690/1/nbnfi-fe2020112092121.pdf) - In this paper, we propose a neural architecture search. (NAS) based method called Auto-FAS, intendin...

