## Import


```python
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

PROJECT_ROOT = Path().resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Project-wide constants and paths ─────────────────────────────────────────
from utils.config import *

# ── Standard library ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from tqdm import tqdm

print_env()
print(f"\nProject root resolved to: {PROJECT_ROOT}")
print(f"\nInterim train exists : {TRAIN_INTERIM.exists()}")
print(f"Interim test exists  : {TEST_INTERIM.exists()}")
print(f"Processed dir exists : {PROCESSED_DIR.exists()}")
print(f"Crops train exists   : {CROP_TRAIN_DIR.exists()}")
```

    Environment  : local
    Project dir  : /home/darrnhard/ML/Competition/FindIT-DAC
      interim    : /home/darrnhard/ML/Competition/FindIT-DAC/data/interim
        train    : /home/darrnhard/ML/Competition/FindIT-DAC/data/interim/train
        test     : /home/darrnhard/ML/Competition/FindIT-DAC/data/interim/test
      processed  : /home/darrnhard/ML/Competition/FindIT-DAC/data/processed
        crops    : /home/darrnhard/ML/Competition/FindIT-DAC/data/processed/crops
        train csv: /home/darrnhard/ML/Competition/FindIT-DAC/data/processed/train_clean.csv
        test csv : /home/darrnhard/ML/Competition/FindIT-DAC/data/processed/test.csv
    Model dir    : /home/darrnhard/ML/Competition/FindIT-DAC/models
    OOF dir      : /home/darrnhard/ML/Competition/FindIT-DAC/oof
    Submissions  : /home/darrnhard/ML/Competition/FindIT-DAC/submissions
    Device       : cuda
    GPU          : NVIDIA GeForce GTX 1050 Ti
    VRAM         : 4.2 GB
    
    Project root resolved to: /home/darrnhard/ML/Competition/FindIT-DAC
    
    Interim train exists : True
    Interim test exists  : True
    Processed dir exists : True
    Crops train exists   : True


## Build train and test DataFrame


```python
# ── Train ─────────────────────────────────────────────────────────────────────
records = []
for cls in CLASSES:
    cls_dir = TRAIN_INTERIM / cls
    for img_path in sorted(cls_dir.glob("*")):
        if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            records.append({
                "path"      : str(img_path),
                "filename"  : img_path.name,
                "label"     : cls,
                "label_idx" : CLASS_TO_IDX[cls],
            })

train_df = pd.DataFrame(records)
print(f"Train images found: {len(train_df)}")
print(f"\nClass distribution:")
print(train_df['label'].value_counts().to_string())

# ── Test ──────────────────────────────────────────────────────────────────────
test_records = []
for img_path in sorted(TEST_INTERIM.glob("*")):
    if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
        test_records.append({
            "path" : str(img_path),
            "id"   : img_path.stem,
        })

test_df = pd.DataFrame(test_records)
print(f"\nTest images found: {len(test_df)}")
```

    Train images found: 1642
    
    Class distribution:
    label
    realperson        448
    fake_unknown      346
    fake_mask         283
    fake_mannequin    229
    fake_screen       197
    fake_printed      139
    
    Test images found: 404


## MD5 Deduplication


```python
from data.preprocessing import file_hash

print("Computing MD5 hashes... (may take ~30 seconds)")
train_df['hash'] = train_df['path'].apply(file_hash)

# ── Analyse duplicates ────────────────────────────────────────────────────────
hash_groups = train_df.groupby('hash').agg(
    count    = ('path',  'size'),
    labels   = ('label', lambda x: sorted(set(x))),
    n_labels = ('label', lambda x: len(set(x))),
).reset_index()

same_label_dupes  = hash_groups[(hash_groups['count'] > 1) & (hash_groups['n_labels'] == 1)]
cross_label_dupes = hash_groups[(hash_groups['count'] > 1) & (hash_groups['n_labels'] > 1)]

print(f"\nTotal unique hashes    : {len(hash_groups)}")
print(f"Same-label duplicates  : {len(same_label_dupes)} groups  → keep one copy")
print(f"Cross-label duplicates : {len(cross_label_dupes)} groups → remove entirely (ambiguous label)")

if len(cross_label_dupes) > 0:
    print(f"\nCross-label examples:")
    for _, row in cross_label_dupes.head(5).iterrows():
        print(f"  {row['labels']}  ({row['count']} copies)")
```

    Computing MD5 hashes... (may take ~30 seconds)
    
    Total unique hashes    : 1464
    Same-label duplicates  : 161 groups  → keep one copy
    Cross-label duplicates : 0 groups → remove entirely (ambiguous label)


### Apply Deduplication


```python
before = len(train_df)

# No cross-label dupes to remove.
# Same-label dupes: keep first occurrence of each hash.
train_df = train_df.drop_duplicates(subset='hash', keep='first').reset_index(drop=True)

after = len(train_df)
print(f"Before : {before} images")
print(f"After  : {after} images  (removed {before - after} same-label duplicates)")
print(f"\nClean class distribution:")
print(train_df['label'].value_counts().to_string())
```

    Before : 1642 images
    After  : 1464 images  (removed 178 same-label duplicates)
    
    Clean class distribution:
    label
    realperson        403
    fake_unknown      307
    fake_mask         266
    fake_mannequin    193
    fake_screen       191
    fake_printed      104


## Initialize InsightFace + Extract Embeddings


```python
import cv2
from insightface.app import FaceAnalysis
from data.preprocessing import get_embedding, assign_identity_groups

# ── Initialize InsightFace Buffalo-L (ArcFace R100) ──────────────────────────
# Downloads model weights on first run (~500MB), cached afterwards
app = FaceAnalysis(
    name='buffalo_l',
    providers=['CPUExecutionProvider']
)
app.prepare(ctx_id=-1, det_size=(640, 640))
print("InsightFace Buffalo-L ready")

# ── Extract embeddings for all training images ────────────────────────────────
print(f"\nExtracting embeddings for {len(train_df)} images...")
paths = train_df['path'].tolist()
embeddings = [get_embedding(p, app) for p in tqdm(paths)]

valid_idx   = [i for i, e in enumerate(embeddings) if e is not None]
no_face_idx = [i for i, e in enumerate(embeddings) if e is None]

print(f"\nFace detected : {len(valid_idx)}")
print(f"No face       : {len(no_face_idx)}")
if no_face_idx:
    print(f"\nNo-face images:")
    for i in no_face_idx[:10]:
        print(f"  [{train_df.iloc[i]['label']}]  {train_df.iloc[i]['filename']}")
    if len(no_face_idx) > 10:
        print(f"  ... and {len(no_face_idx) - 10} more")
```

    Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
    find model: /home/darrnhard/.insightface/models/buffalo_l/1k3d68.onnx landmark_3d_68 ['None', 3, 192, 192] 0.0 1.0
    Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
    find model: /home/darrnhard/.insightface/models/buffalo_l/2d106det.onnx landmark_2d_106 ['None', 3, 192, 192] 0.0 1.0
    Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
    find model: /home/darrnhard/.insightface/models/buffalo_l/det_10g.onnx detection [1, 3, '?', '?'] 127.5 128.0
    Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
    find model: /home/darrnhard/.insightface/models/buffalo_l/genderage.onnx genderage ['None', 3, 96, 96] 0.0 1.0
    Applied providers: ['CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}}
    find model: /home/darrnhard/.insightface/models/buffalo_l/w600k_r50.onnx recognition ['None', 3, 112, 112] 127.5 127.5
    set det-size: (640, 640)
    InsightFace Buffalo-L ready
    
    Extracting embeddings for 1464 images...


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1464/1464 [11:36<00:00,  2.10it/s]

    
    Face detected : 1238
    No face       : 226
    
    No-face images:
      [fake_mannequin]  mannequin_009.jpg
      [fake_mannequin]  mannequin_012.jpeg
      [fake_mannequin]  mannequin_022.jpg
      [fake_mannequin]  mannequin_024.jpg
      [fake_mannequin]  mannequin_034.jpg
      [fake_mannequin]  mannequin_037.jpg
      [fake_mannequin]  mannequin_060.jpeg
      [fake_mannequin]  mannequin_081.jpeg
      [fake_mannequin]  mannequin_098.jpg
      [fake_mannequin]  mannequin_107.jpg
      ... and 216 more


    


## HDBSCAN Clustering + Group Assignment


```python
from data.preprocessing import assign_identity_groups

group_col, valid_idx, no_face_idx = assign_identity_groups(embeddings)

train_df['group'] = group_col

n_clusters = len(set(group_col[group_col >= 0])) - (1 if -1 in group_col else 0)
print(f"Total images  : {len(train_df)}")
print(f"Unique groups : {train_df['group'].nunique()}")
print(f"Largest group : {train_df.groupby('group').size().max()} images")
print(f"\nGroup size distribution:")
print(train_df.groupby('group').size().value_counts().sort_index().head(10))
```

    Total images  : 1464
    Unique groups : 807
    Largest group : 16 images
    
    Group size distribution:
    1     509
    2     155
    3      69
    4      37
    5      17
    6       6
    7       2
    8       2
    9       1
    13      2
    Name: count, dtype: int64


## StratifiedGroupKFold Assignment + Quality Verification


```python
from sklearn.model_selection import StratifiedGroupKFold

# ── Assign folds ──────────────────────────────────────────────────────────────
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
train_df['fold'] = -1

fold_col = train_df.columns.get_loc('fold')
for fold, (train_idx, val_idx) in enumerate(
    sgkf.split(train_df, train_df['label'], train_df['group'])
):
    # Use iloc (positional) — safe regardless of whether index is a clean RangeIndex
    train_df.iloc[val_idx, fold_col] = fold

# ── Leakage check ─────────────────────────────────────────────────────────────
print("Group leakage check:")
for fold in range(N_FOLDS):
    val_groups   = set(train_df[train_df['fold'] == fold]['group'])
    train_groups = set(train_df[train_df['fold'] != fold]['group'])
    overlap      = val_groups & train_groups
    print(f"  Fold {fold}: {'✓ CLEAN' if len(overlap) == 0 else f'✗ LEAK — {len(overlap)} groups'}")

# ── Class distribution per fold ───────────────────────────────────────────────
print(f"\nSamples per fold:")
for fold in range(N_FOLDS):
    print(f"  Fold {fold}: {len(train_df[train_df['fold'] == fold])} images")

print(f"\nClass distribution per fold (validation set counts):")
print(train_df.groupby(['fold', 'label']).size().unstack(fill_value=0).to_string())

# ── CV% quality check (research requirement: all classes < 15%) ───────────────
print(f"\nCV% of class counts across folds (target: all < 15%):")
all_pass = True
for cls in sorted(CLASSES):
    counts = [len(train_df[(train_df['fold'] == f) & (train_df['label'] == cls)])
              for f in range(N_FOLDS)]
    cv = np.std(counts) / np.mean(counts) * 100
    status = '✓' if cv < 15 else ('⚠' if cv < 25 else '✗ REGENERATE')
    if cv >= 25:
        all_pass = False
    print(f"  {cls:<20}: {cv:5.1f}%  {status}   counts={counts}")

if all_pass:
    print(f"\n✓ All classes within acceptable range — fold assignment is valid")
else:
    print(f"\n✗ Some classes above 25% — consider rerunning with a different random_state")
```

## Save Cleaned DataFrames


```python
train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print(f"Saved:")
print(f"  {TRAIN_CSV}")
print(f"    {len(train_df)} images | {N_FOLDS} folds | {train_df['group'].nunique()} identity groups")
print(f"\n  {TEST_CSV}")
print(f"    {len(test_df)} images")
print(f"\nClass weights (inverse frequency — for training notebook):")
class_counts = train_df['label'].value_counts()
for cls in CLASSES:
    weight = len(train_df) / (NUM_CLASSES * class_counts[cls])
    print(f"  {cls:<20}: {class_counts[cls]:>4d} images   weight = {weight:.4f}")
```

    Saved:
      /home/darrnhard/ML/Competition/FindIT-DAC/data/processed/train_clean.csv
        1464 images | 5 folds | 807 identity groups
    
      /home/darrnhard/ML/Competition/FindIT-DAC/data/processed/test.csv
        404 images
    
    Class weights (inverse frequency — for training notebook):
      fake_mannequin      :  193 images   weight = 1.2642
      fake_mask           :  266 images   weight = 0.9173
      fake_printed        :  104 images   weight = 2.3462
      fake_screen         :  191 images   weight = 1.2775
      fake_unknown        :  307 images   weight = 0.7948
      realperson          :  403 images   weight = 0.6055


## Face Detection Setup


```python
import torch
from facenet_pytorch import MTCNN
from data.preprocessing import crop_face

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

mtcnn = MTCNN(
    keep_all=False,
    device=device,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
)
print("MTCNN ready")
```

    Using device: cuda
    MTCNN ready


## Crop All Train Images


```python
face_found_count = 0
no_face_count = 0

print(f"Cropping {len(train_df)} training images → {CROP_TRAIN_DIR}")

for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
    cropped, found = crop_face(row['path'], mtcnn, expansion=EXPANSION, target_size=IMG_SIZE)

    # Save as: label_originalfilename.jpg for easy traceability
    save_name = f"{row['label']}_{Path(row['path']).stem}.jpg"
    save_path  = CROP_TRAIN_DIR / save_name
    cropped.save(save_path, quality=95)

    train_df.at[idx, 'crop_path'] = str(save_path)

    if found:
        face_found_count += 1
    else:
        no_face_count += 1

print(f"\nDone.")
print(f"  Face detected    : {face_found_count}")
print(f"  Fallback (center): {no_face_count}")
print(f"  Total saved      : {face_found_count + no_face_count}")
```

    Cropping 1464 training images → /home/darrnhard/ML/Competition/FindIT-DAC/data/processed/crops/train


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1464/1464 [03:01<00:00,  8.06it/s]

    
    Done.
      Face detected    : 1359
      Fallback (center): 105
      Total saved      : 1464


    


## Crop All Test Images


```python
face_found_test = 0
no_face_test = 0

print(f"Cropping {len(test_df)} test images → {CROP_TEST_DIR}")

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    cropped, found = crop_face(row['path'], mtcnn, expansion=EXPANSION, target_size=IMG_SIZE)

    save_name = f"{row['id']}.jpg"
    save_path  = CROP_TEST_DIR / save_name
    cropped.save(save_path, quality=95)

    test_df.at[idx, 'crop_path'] = str(save_path)

    if found:
        face_found_test += 1
    else:
        no_face_test += 1

print(f"\nDone.")
print(f"  Face detected    : {face_found_test}")
print(f"  Fallback (center): {no_face_test}")
print(f"  Total saved      : {face_found_test + no_face_test}")
```

    Cropping 404 test images → /home/darrnhard/ML/Competition/FindIT-DAC/data/processed/crops/test


    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 404/404 [01:03<00:00,  6.32it/s]

    
    Done.
      Face detected    : 382
      Fallback (center): 22
      Total saved      : 404


    


## Save Final CSVs + Summary


```python
# Re-save with crop_path column added
train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

# ── Final summary ─────────────────────────────────────────────────────────────
print("=" * 55)
print("DATA PREPARATION v3 — COMPLETE")
print("=" * 55)

print(f"\n[ Source ]")
print(f"  Interim train    : {len(records)} images (raw scan)")
print(f"  After dedup      : {len(train_df)} images (removed {len(records) - len(train_df)} same-label dupes)")
print(f"  Cross-label dupes: 0 (resolved by manual cleaning)")

print(f"\n[ Identity Grouping ]")
print(f"  Method           : InsightFace Buffalo-L (ArcFace R100) + HDBSCAN")
print(f"  Face detected    : {len(valid_idx)} / {len(train_df)} images")
print(f"  No face          : {len(no_face_idx)} images (singleton groups)")
print(f"  Unique groups    : {train_df['group'].nunique()}")
print(f"  Largest group    : {train_df.groupby('group').size().max()} images")

print(f"\n[ Folds ]")
print(f"  Method           : StratifiedGroupKFold (n=5, seed=42)")
print(f"  Leakage          : 0 groups shared across train/val")
print(f"  Max class CV%    : {max([np.std([len(train_df[(train_df['fold']==f) & (train_df['label']==cls)]) for f in range(N_FOLDS)]) / np.mean([len(train_df[(train_df['fold']==f) & (train_df['label']==cls)]) for f in range(N_FOLDS)]) * 100 for cls in CLASSES]):.1f}%  (threshold: 15%)")

print(f"\n[ Face Cropping ]")
print(f"  Expansion        : {EXPANSION}x  |  Target size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Train — detected : {face_found_count}  |  fallback: {no_face_count}")
print(f"  Test  — detected : {face_found_test}   |  fallback: {no_face_test}")

print(f"\n[ Outputs → data/processed/ ]")
print(f"  {TRAIN_CSV.name:<25}: {len(train_df)} rows, {len(train_df.columns)} columns")
print(f"  {TEST_CSV.name:<25}: {len(test_df)} rows, {len(test_df.columns)} columns")
print(f"  crops/train/     : {face_found_count + no_face_count} images ({IMG_SIZE}x{IMG_SIZE} JPG)")
print(f"  crops/test/      : {face_found_test + no_face_test} images ({IMG_SIZE}x{IMG_SIZE} JPG)")

print(f"\n[ Class Distribution ]")
print(f"  {'Class':<20} {'Count':>6}   {'Weight':>7}   {'Per fold (val)':>15}")
class_counts = train_df['label'].value_counts()
for cls in CLASSES:
    cnt    = class_counts[cls]
    weight = len(train_df) / (NUM_CLASSES * cnt)
    per_fold = cnt // N_FOLDS
    print(f"  {cls:<20} {cnt:>6}   {weight:>7.4f}   ~{per_fold:>3} per fold")

print(f"\n→ Next: open training notebook on GPU server")
```

    =======================================================
    DATA PREPARATION v3 — COMPLETE
    =======================================================
    
    [ Source ]
      Interim train    : 1642 images (raw scan)
      After dedup      : 1464 images (removed 178 same-label dupes)
      Cross-label dupes: 0 (resolved by manual cleaning)
    
    [ Identity Grouping ]
      Method           : InsightFace Buffalo-L (ArcFace R100) + HDBSCAN
      Face detected    : 1238 / 1464 images
      No face          : 226 images (singleton groups)
      Unique groups    : 807
      Largest group    : 16 images
    
    [ Folds ]
      Method           : StratifiedGroupKFold (n=5, seed=42)
      Leakage          : 0 groups shared across train/val
      Max class CV%    : 1.9%  (threshold: 15%)
    
    [ Face Cropping ]
      Expansion        : 1.4x  |  Target size: 224x224
      Train — detected : 1359  |  fallback: 105
      Test  — detected : 382   |  fallback: 22
    
    [ Outputs → data/processed/ ]
      train_clean.csv          : 1464 rows, 8 columns
      test.csv                 : 404 rows, 3 columns
      crops/train/     : 1464 images (224x224 JPG)
      crops/test/      : 404 images (224x224 JPG)
    
    [ Class Distribution ]
      Class                 Count    Weight    Per fold (val)
      fake_mannequin          193    1.2642   ~ 38 per fold
      fake_mask               266    0.9173   ~ 53 per fold
      fake_printed            104    2.3462   ~ 20 per fold
      fake_screen             191    1.2775   ~ 38 per fold
      fake_unknown            307    0.7948   ~ 61 per fold
      realperson              403    0.6055   ~ 80 per fold
    
    → Next: open training notebook on GPU server



```python

```


```python

```
