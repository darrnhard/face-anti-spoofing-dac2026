# Session Log — 01_data_preparation.ipynb
**Competition:** DAC Find IT 2026 — Face Anti-Spoofing (6-class, Macro F1)
**Session date:** 2026-04-01
**Notebook:** `01_data_preparation.ipynb`
**Status:** ✅ Complete

---

## Context Going Into This Session

- First iteration already complete: 20 models trained (4 architectures × 5 folds), best LB 0.71238
- Large CV-LB gap: OOF CV 0.8856 vs LB 0.71238 — audit identified this as the core problem
- Audit file (`fas_audit_and_plan.md`) identified multiple critical issues
- This session focused entirely on fixing data preparation before retraining

---

## What We Did This Session

### 1. Reviewed All Context Files
Read `data_preparation.md`, `fas_audit_and_plan.md`, `Initial-Research-Plan.md` in full before touching any code.

### 2. Removed Leaked Label Logic
**What:** Deleted the entire `### Find Leaked Labels` section from the notebook.

**Why:** Audit showed leaked labels were *hurting* LB by -0.028 (0.71238 pure → 0.68434 with leaked). Root cause: some of the 75 leaked labels may be wrong because the competition organizers assigned different labels in the test set than we assigned in training.

**Decision:** Moved leaked label logic to `03_inference.ipynb` where it belongs — it's a prediction override, not a data prep step. We'll re-examine it there with more careful verification before next submission.

### 3. Confirmed MD5 Deduplication Logic is Correct
Darren questioned whether the deduplication already handled the cross-label ambiguity problem. Confirmed yes — the cleaning step guarantees every hash in `train_df` maps to exactly one label unambiguously. No additional verification cell was needed.

**Key insight:** MD5 only catches exact byte-for-byte duplicates. Near-duplicates (same subject, different angle, different JPEG quality) pass through — which is what Step 5 addresses.

### 4. Deferred Cleanlab
**Why deferred:** Cleanlab requires trustworthy OOF predictions as input. Current OOF files come from a model trained with the backbone/head split bug (ConvNeXt was accidentally trained with most of its backbone at lr=1e-3, 10× too high). Using buggy predictions for label cleaning would produce false flags.

**Plan:** Run Cleanlab after we fix the training pipeline and get clean OOF predictions from the retrained models.

**OOF files confirmed available locally:**
```
/work/oof_convnext_fold{0-4}.csv
/work/oof_effnet_b4_fold{0-4}.csv
/work/oof_effnet_fold{0-4}.csv
/work/oof_resnet50_fold{0-4}.csv
```

### 5. Replaced StratifiedKFold with StratifiedGroupKFold + Face Identity Clustering
**What:** Replaced the entire `### Stratified 5-Fold Split` section.

**Why:** `StratifiedKFold` doesn't know about image similarity. If two photos of the same person (one real, one printed) end up in train and val respectively, the model recognizes the face — not the attack type. This inflates CV scores.

**Approach chosen:** Face identity clustering using `InceptionResnetV1` (VGGFace2 pretrained) + DBSCAN.
- Extract 512-dim face embeddings for all 1342 training images
- Cluster with DBSCAN (eps=0.2, cosine distance) — images within distance 0.2 = same person
- 102 no-face images → each gets a unique singleton group
- Pass group IDs to `StratifiedGroupKFold` — groups never split across train/val

**Why eps=0.2:** First tried eps=0.4 → largest group had 469 images (35% of dataset), completely broken fold balance. eps=0.2 gave largest group = 14, perfect balance.

**Results:**
```
Total images  : 1342
Unique groups : 970
Largest group : 14 images
Fold 0: 269 images ✓ CLEAN
Fold 1: 269 images ✓ CLEAN
Fold 2: 268 images ✓ CLEAN
Fold 3: 268 images ✓ CLEAN
Fold 4: 268 images ✓ CLEAN
```

**Key finding:** 372 images share identity with at least one other image — significantly more than the 11 near-duplicates pHash caught. This means there was meaningful identity leakage in the old CV splits.

### 6. Inspected 102 No-Face Images
Visualized all 102 images that MTCNN failed to detect a face on. Class breakdown:
```
fake_unknown    46
realperson      21
fake_screen     15
fake_printed    12
fake_mannequin   4
fake_mask        4
```

**Decision: Keep all 102.** Only 4 images have truly zero useful signal (screen_097, screen_103, unknown_110, unknown_104), but 4/1342 = 0.3% — not worth adding deletion logic complexity. MTCNN simply failed on legitimate images; center crop fallback still captures spoofing clues.

**Notable finding:** `real_197.jpg` (B&W printed photo labeled realperson) and `real_283.jpg` (blue painted face labeled realperson) appear mislabeled. These will be caught by Cleanlab after retraining — no manual deletion.

### 7. Updated Data Preparation Summary Cell
Removed leaked label line, added identity groups line.

### 8. Re-ran Full Notebook Top to Bottom
All cells ran clean. Final outputs verified correct.

---

## Final State of Outputs

| File | Status | Notes |
|---|---|---|
| `work/train_clean.csv` | ✅ Updated | 1342 images, 5 folds, 970 identity groups, `crop_path` column |
| `work/test.csv` | ✅ Updated | 404 images, `crop_path` column |
| `work/cropped/train/` | ✅ Updated | 1342 face-cropped 224×224 images |
| `work/cropped/test/` | ✅ Updated | 404 face-cropped 224×224 images |
| `work/leaked_labels.csv` | ⚠️ Stale | No longer generated here — move logic to inference notebook |

**New columns in `train_clean.csv`:**
- `group` — integer identity group ID (0 to 969)
- `fold` — updated fold assignments (0-4), now subject-independent

---

## Key Decisions & Rationale

| Decision | Rationale |
|---|---|
| Use face identity (InceptionResnetV1) over pHash for grouping | FAS-specific: risk is model recognizing a face, not visual similarity |
| eps=0.2 for DBSCAN | eps=0.4 created a giant 469-image cluster breaking fold balance |
| Keep all 102 no-face images | <0.3% of dataset, center crop still captures spoofing signal |
| Defer Cleanlab | Needs trustworthy OOF — wait for retrained model |
| Remove leaked labels from data prep | It's a prediction override, belongs in inference notebook |

---

## What's Deferred / Still To Do

| Item | Priority | Blocker |
|---|---|---|
| Cleanlab noisy label detection | P0 after retraining | Needs clean OOF from fixed model |
| Fix leaked label logic in inference notebook | P0 | Needs careful per-label verification |
| Higher-res re-crop (320×320) | P2 | Wait until ready to train ConvNeXt at higher res |

---

## Next Session: `02_training.ipynb`

**Primary goal:** Fix the backbone/head parameter split bug — the single highest ROI change.

**The bug:** ConvNeXt's parameter split logic uses string matching for `'head'` but ConvNeXt's architecture names ALL its final stages with `head` in the path. Result: 83.8M params were treated as "head" (lr=1e-3) while only 3.7M got the conservative backbone lr (1e-4). The backbone was trained 10× too aggressively.

**Fix to implement:**
```python
def get_param_groups(model, model_key):
    head_params, backbone_params = [], []
    if model_key == 'convnext':
        for name, param in model.named_parameters():
            if name.startswith('head.fc'):
                head_params.append(param)
            else:
                backbone_params.append(param)
    # ... other architectures
    return backbone_params, head_params
```

**Verify:** After fix, ConvNeXt head should be ~6K params, backbone ~87M params.

**Then:** Retrain all folds with fixed split. New OOF predictions → Cleanlab → final clean dataset.

---

## Environment Reference

- Local: Arch Linux, GTX 1050 Ti, micromamba `findit-dac` env
- Remote: Vast.ai RTX 3090, VS Code Remote-SSH
- Notebook structure: `01_data_preparation.ipynb` (local) → `02_training.ipynb` (remote GPU) → `03_inference.ipynb` (local)
- Key paths:
  - Dataset: `/home/darrnhard/ML/Competition/FindIT-DAC/dataset`
  - Work: `/home/darrnhard/ML/Competition/FindIT-DAC/work`
  - Crops: `/home/darrnhard/ML/Competition/FindIT-DAC/work/cropped`
