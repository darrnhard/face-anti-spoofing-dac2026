# 12 — Resubmission Strategy — exp07 (LB-Informed Ensemble Weights)

**Purpose:** Use the 3 remaining submission slots to search for a better ensemble weighting
than `all5_argmax` (current best LB = **0.79207**).

**No re-inference needed.** All 5 exp07 per-arch test probability arrays are already saved
at `oof/exp07/test_probs_{arch}.npy`. This notebook is pure numpy — runs in <1 minute on CPU.

---

### LB Evidence Summary (from 11-inference-exp07.ipynb)

| Submission | OOF F1 | LB Score | Interpretation |
|---|---|---|---|
| `pruned_mega_argmax` (exp06×5 + exp07×3) | 0.9349 | 0.78555 | exp06 models add noise on test set |
| `top4_argmax` (drop convnext) | 0.9306 | 0.78673 | convnext adds LB value |
| `all5_argmax` (equal weight) | 0.9252 | **0.79207** ← current best | All 5 needed; equal weights best so far |

**Key fact:** OOF ranking was exactly inverted on LB. This means:
1. Convnext is underweighted at equal (0.20) — it contributed +0.00534 to LB vs dropping it
2. Exp06 models introduced test-set noise — never add them again
3. OOF-optimized strategies (pruned mega) can be actively harmful on this test distribution

**Gap to close:** 0.800 − 0.79207 = **0.00793**

---

### 3 Submission Plan

| Slot | Name | Strategy | Rationale |
|---|---|---|---|
| 1 | `scipy_opt_argmax` | scipy.minimize on OOF Macro F1 | Most principled weight search |
| 2 | `convnext_2x_argmax` | convnext w=0.333, others w=0.167 | Directly from LB: convnext worth 2× |
| 3 | `convnext_1p5x_argmax` | convnext w=0.250, others w=0.1875 | Bracket midpoint between 1× and 2× |

**Why not threshold tuning:** nested CV OOB = 0.9216 < argmax 0.9252 → overfitting signal
= +0.0108, expected LB Δ = −0.0037. The data predicts thresholds will hurt; a slot is too
valuable to spend on a strategy the analysis already ruled out.


## 1. Setup


```python
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.metrics import f1_score
from scipy.optimize import minimize

PROJECT_ROOT = Path().resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    CLASSES, NUM_CLASSES, OOF_DIR, SUBMISSION_DIR,
    CROP_TRAIN_DIR, CROP_TEST_DIR, TRAIN_CSV, TEST_CSV,
    N_FOLDS, MODEL_DIR,
)
from src.inference.submission import make_submission

# ── Experiment constants ───────────────────────────────────────────────────────
EXP_ID    = 'exp07'
ALL_ARCHS = ['convnext', 'eva02', 'dinov2', 'effnet_b4', 'swinv2']
prob_cols = [f'prob_{cls}' for cls in CLASSES]

SUBMISSION_DIR.mkdir(exist_ok=True)
print(f'EXP_ID    : {EXP_ID}')
print(f'ALL_ARCHS : {ALL_ARCHS}')
print(f'OOF dir   : {OOF_DIR / EXP_ID}')
print(f'Submit dir: {SUBMISSION_DIR}')

# ── Locked LB results from 11-inference-exp07.ipynb ───────────────────────────
LB_RESULTS = {
    'all5_argmax'         : 0.79207,   # current best ← convnext included, equal weight
    'top4_argmax'         : 0.78673,   # drop convnext
    'pruned_mega_argmax'  : 0.78555,   # exp06 models hurt
}
print()
print('Locked LB results:')
for name, lb in sorted(LB_RESULTS.items(), key=lambda x: -x[1]):
    print(f'  {name:<30}: LB={lb:.5f}')
print(f'  Gap to top (0.800)          : {0.800 - max(LB_RESULTS.values()):.5f}')

```

    EXP_ID    : exp07
    ALL_ARCHS : ['convnext', 'eva02', 'dinov2', 'effnet_b4', 'swinv2']
    OOF dir   : /home/darrnhard/ML/Competition/FindIT-DAC/oof/exp07
    Submit dir: /home/darrnhard/ML/Competition/FindIT-DAC/submissions
    
    Locked LB results:
      all5_argmax                   : LB=0.79207
      top4_argmax                   : LB=0.78673
      pruned_mega_argmax            : LB=0.78555
      Gap to top (0.800)          : 0.00793



```python
import pandas as pd
from pathlib import Path

test_df  = pd.read_csv(TEST_CSV)
test_df['crop_path'] = test_df['crop_path'].apply(
    lambda p: str(CROP_TEST_DIR / Path(p).name)
)
print(f'Test set: {len(test_df)} images')
assert Path(test_df['crop_path'].iloc[0]).exists(), \
    f'Test crops not found — check CROP_TEST_DIR: {CROP_TEST_DIR}'
print('✅ test_df ready')

```

    Test set: 404 images
    ✅ test_df ready


## 2. Load Saved Probabilities

Load the per-arch test probability arrays saved during 11-inference-exp07.ipynb,
and the OOF arrays needed for scipy weight optimization.

This is CPU-only. No checkpoints loaded.



```python
probs_dir = OOF_DIR / EXP_ID

# ── Test probability arrays (from inference) ───────────────────────────────────
print('Loading test probability arrays...')
test_probs = {}
for arch in ALL_ARCHS:
    path = probs_dir / f'test_probs_{arch}.npy'
    if not path.exists():
        raise FileNotFoundError(
            f'Missing: {path}\n'
            f'Run 11-inference-exp07.ipynb first — it saves these files.'
        )
    arr = np.load(path)
    row_sum = arr.sum(axis=1).mean()
    test_probs[arch] = arr
    print(f'  {arch:<14}: shape={arr.shape}  row_sum={row_sum:.4f} ✓')

print()

# ── OOF probability arrays (for weight optimization) ──────────────────────────
# The OOF CSVs contain per-image softmax probabilities for each arch.
# We align them to a common sort order (by crop_path) and build numpy arrays.
print('Loading OOF probability arrays...')
oof_base = OOF_DIR / EXP_ID
oof_data = {}
for arch in ALL_ARCHS:
    fold_dfs = []
    for fold in range(N_FOLDS):
        path = oof_base / f'oof_{arch}_fold{fold}.csv'
        df = pd.read_csv(path)
        df['crop_path'] = df['crop_path'].apply(
            lambda p: str(CROP_TRAIN_DIR / Path(p).name)
        )
        fold_dfs.append(df)
    oof_data[arch] = pd.concat(fold_dfs, ignore_index=True)
    print(f'  {arch:<14}: {len(oof_data[arch])} OOF rows ✓')

# ── Align all OOF arrays to a single sorted order ─────────────────────────────
base_df     = oof_data['convnext'].sort_values('crop_path').reset_index(drop=True)
true_labels = base_df['label_idx'].values

oof_prbs = {}
for arch in ALL_ARCHS:
    aligned       = oof_data[arch].set_index('crop_path').loc[base_df['crop_path']]
    oof_prbs[arch] = aligned[prob_cols].values.astype('float32')

print()

# ── Quick sanity: all5 equal OOF and test row sums ────────────────────────────
oof_all5  = np.mean(list(oof_prbs.values()), axis=0)
f1_check  = f1_score(true_labels, oof_all5.argmax(axis=1), average='macro')
test_all5 = np.mean(list(test_probs.values()), axis=0)

print(f'all5 equal OOF F1 (should be ~0.9252): {f1_check:.4f}  {"✅" if abs(f1_check - 0.9252) < 0.002 else "⚠️ MISMATCH"}')
print(f'all5 equal test row_sum (should be ~1): {test_all5.sum(axis=1).mean():.4f}')

```

    Loading test probability arrays...
      convnext      : shape=(404, 6)  row_sum=1.0000 ✓
      eva02         : shape=(404, 6)  row_sum=1.0000 ✓
      dinov2        : shape=(404, 6)  row_sum=1.0000 ✓
      effnet_b4     : shape=(404, 6)  row_sum=1.0000 ✓
      swinv2        : shape=(404, 6)  row_sum=1.0000 ✓
    
    Loading OOF probability arrays...
      convnext      : 1464 OOF rows ✓
      eva02         : 1464 OOF rows ✓
      dinov2        : 1464 OOF rows ✓
      effnet_b4     : 1464 OOF rows ✓
      swinv2        : 1464 OOF rows ✓
    
    all5 equal OOF F1 (should be ~0.9252): 0.9252  ✅
    all5 equal test row_sum (should be ~1): 1.0000


## 3. Scipy-Optimized Ensemble Weights

**Method:** `scipy.optimize.minimize` with Nelder-Mead, optimizing for **OOF Macro F1** over
the 5 architecture weight vector. Weights are constrained to sum to 1 and all be ≥ 0 via
a softmax reparameterization (unconstrained logits → softmax → weights).

**Why softmax reparameterization?** It automatically enforces the simplex constraint
(weights ≥ 0, sum = 1) without needing scipy's constrained optimizer (SLSQP), which is
slower and more sensitive to initialization.

**Multiple restarts (n_restarts=50):** We run from 50 random starting points and take the
best result. This reduces the risk of getting stuck in a local minimum. More restarts than
exp06's analysis (which used 30) since this is our last attempt.

**What to expect:** In exp06, scipy weights found OOF +0.0043. In exp07, EVA-02 (best OOF)
and DINOv2 are expected to dominate. The key question is whether convnext gets a non-trivial
weight, since LB evidence shows it contributes value beyond what OOF suggests.



```python
np.random.seed(42)
N_RESTARTS = 50

def weighted_f1(logits):
    """
    Objective for scipy.minimize — minimize negative OOF Macro F1.

    logits : unconstrained vector of length n_archs
        Converted via softmax to weights that sum to 1 and are all > 0.
        This avoids the need for constrained optimization while guaranteeing
        a valid probability simplex.
    """
    # Softmax: convert logits to weights (always sums to 1, always > 0)
    w = np.exp(logits - logits.max())
    w = w / w.sum()

    # Weighted average of OOF probability arrays
    avg = sum(w[i] * oof_prbs[ALL_ARCHS[i]] for i in range(len(ALL_ARCHS)))
    f1  = f1_score(true_labels, avg.argmax(axis=1), average='macro')
    return -f1   # minimize negative F1 = maximize F1


print(f'Running scipy Nelder-Mead with {N_RESTARTS} random restarts...')
print('(This runs on CPU — typically <30 seconds)\n')

best_f1  = -np.inf
best_w   = None
best_res = None

for restart in range(N_RESTARTS):
    x0  = np.random.randn(len(ALL_ARCHS))
    res = minimize(weighted_f1, x0, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-7})
    # Recover weights from optimized logits
    w_raw = np.exp(res.x - res.x.max())
    w     = w_raw / w_raw.sum()
    f1    = -res.fun

    if f1 > best_f1:
        best_f1  = f1
        best_w   = w
        best_res = res

print(f'Best OOF Macro F1 found : {best_f1:.6f}')
print(f'Equal-weight baseline   : {f1_check:.6f}')
print(f'OOF gain vs equal       : {best_f1 - f1_check:+.6f}')
print()
print('Optimized weights:')
for arch, w in zip(ALL_ARCHS, best_w):
    bar  = '█' * int(w * 80)
    flag = '  ← LB evidence agrees' if arch == 'convnext' and w > 0.20 else ''
    print(f'  {arch:<14}: {w:.4f}  {bar}{flag}')

print()
print(f'Weight sum (sanity): {best_w.sum():.6f}  (should be 1.0)')

```

    Running scipy Nelder-Mead with 50 random restarts...
    (This runs on CPU — typically <30 seconds)
    
    Best OOF Macro F1 found : 0.932666
    Equal-weight baseline   : 0.925226
    OOF gain vs equal       : +0.007439
    
    Optimized weights:
      convnext      : 0.0209  █
      eva02         : 0.2506  ████████████████████
      dinov2        : 0.2803  ██████████████████████
      effnet_b4     : 0.2348  ██████████████████
      swinv2        : 0.2133  █████████████████
    
    Weight sum (sanity): 1.000000  (should be 1.0)



```python
# ── Build scipy-optimized test ensemble ───────────────────────────────────────
test_probs_scipy = sum(
    best_w[i] * test_probs[ALL_ARCHS[i]] for i in range(len(ALL_ARCHS))
)

# Verify row sums
assert abs(test_probs_scipy.sum(axis=1).mean() - 1.0) < 1e-4, 'Row sums broken'

# Prediction agreement with all5_argmax (current best)
preds_scipy = test_probs_scipy.argmax(axis=1)
preds_all5  = test_all5.argmax(axis=1)
n_disagree  = (preds_scipy != preds_all5).sum()
print(f'scipy_opt vs all5_argmax:')
print(f'  Disagreements : {n_disagree} / {len(preds_all5)} samples ({n_disagree/len(preds_all5)*100:.1f}%)')
print(f'  Agreement     : {(preds_scipy == preds_all5).mean():.4f}')
print()

# Class distribution
print('Class distribution — scipy_opt vs all5:')
print(f'  {"Class":<22}  {"all5":>6}  {"scipy":>6}  {"Δ":>4}')
for i, cls in enumerate(CLASSES):
    n_all5  = int((preds_all5  == i).sum())
    n_scipy = int((preds_scipy == i).sum())
    print(f'  {cls:<22}  {n_all5:>6}  {n_scipy:>6}  {n_scipy - n_all5:>+4}')

```

    scipy_opt vs all5_argmax:
      Disagreements : 4 / 404 samples (1.0%)
      Agreement     : 0.9901
    
    Class distribution — scipy_opt vs all5:
      Class                     all5   scipy     Δ
      fake_mannequin              52      51    -1
      fake_mask                   71      72    +1
      fake_printed                55      54    -1
      fake_screen                 66      67    +1
      fake_unknown                52      53    +1
      realperson                 108     107    -1


## 4. Convnext-Upweighted Ensembles

**Motivation:** Direct LB evidence.
- `all5_argmax` (convnext w=0.20) → LB 0.79207
- `top4_argmax` (convnext w=0.00) → LB 0.78673
- Delta = +0.00534 from going 0→0.20 for convnext

If the marginal value of convnext is positive and roughly linear in weight, increasing
convnext's weight beyond 0.20 should continue to improve LB — up to some saturation point.
We test two bracket points:

- **2× (w=0.333):** Other 4 archs each get (1−0.333)/4 = 0.167
- **1.5× (w=0.250):** Other 4 archs each get (1−0.250)/4 = 0.1875

The spacing gives us 3 LB data points {0.20, 0.25, 0.33} which bracket the optimal
convnext weight for the test distribution.

⚠️ **Important caveat:** We are making an assumption that convnext's value is monotonically
increasing with weight. This is not guaranteed — it could be that convnext makes unique correct
predictions that are already captured at w=0.20 and further upweighting just overrides the
other models on samples they get right. The bracket approach mitigates this risk.



```python
# ── Build the two convnext-upweighted ensembles ───────────────────────────────
N_ARCHS = len(ALL_ARCHS)   # 5

def convnext_weighted_probs(convnext_weight: float) -> np.ndarray:
    """
    Build a weighted ensemble with convnext at `convnext_weight` and
    the remaining 4 archs sharing the rest equally.

    Parameters
    ----------
    convnext_weight : float in (0, 1)
        Weight assigned to convnext. e.g. 0.333 → 2× upweight vs equal (0.20).

    Returns
    -------
    np.ndarray shape (N_TEST, NUM_CLASSES) — weighted average of test probabilities
    """
    other_archs  = [a for a in ALL_ARCHS if a != 'convnext']
    other_weight = (1.0 - convnext_weight) / len(other_archs)

    weights = {}
    weights['convnext'] = convnext_weight
    for a in other_archs:
        weights[a] = other_weight

    avg = sum(weights[a] * test_probs[a] for a in ALL_ARCHS)
    return avg, weights


# ── 2× convnext (w=0.333) ─────────────────────────────────────────────────────
test_probs_cx2, weights_cx2 = convnext_weighted_probs(1/3)

# ── 1.5× convnext (w=0.250) ───────────────────────────────────────────────────
test_probs_cx1p5, weights_cx1p5 = convnext_weighted_probs(0.25)

# ── Print weight tables ───────────────────────────────────────────────────────
print('Ensemble weight comparison:')
print(f'  {"Arch":<14}  {"equal":>8}  {"1.5×cx":>8}  {"2×cx":>8}')
print('  ' + '-' * 48)
for arch in ALL_ARCHS:
    w_eq   = 1/5
    w_1p5  = weights_cx1p5[arch]
    w_2x   = weights_cx2[arch]
    marker = '  ← upweighted' if arch == 'convnext' else ''
    print(f'  {arch:<14}  {w_eq:>8.4f}  {w_1p5:>8.4f}  {w_2x:>8.4f}{marker}')
print(f'  {"SUM":<14}  {1.0:>8.4f}  {sum(weights_cx1p5.values()):>8.4f}  {sum(weights_cx2.values()):>8.4f}')

# ── Compute OOF F1 for both variants ─────────────────────────────────────────
# Note: this OOF F1 is expected to be LOWER than equal weight because OOF
# says convnext hurts. But LB says it helps. We compute anyway for the record.
print()
for label, ws in [('1.5× convnext', weights_cx1p5), ('2× convnext', weights_cx2)]:
    oof_w = sum(ws[a] * oof_prbs[a] for a in ALL_ARCHS)
    f1_w  = f1_score(true_labels, oof_w.argmax(axis=1), average='macro')
    print(f'  {label} OOF F1: {f1_w:.4f}  (equal={f1_check:.4f}, Δ={f1_w - f1_check:+.4f})')
    print(f'    ↳ OOF likely lower because convnext regressed in exp07 training.')
    print(f'    ↳ LB evidence supersedes OOF here — proceed anyway.')
print()

# ── Row sum sanity ────────────────────────────────────────────────────────────
for name, arr in [('cx_1p5', test_probs_cx1p5), ('cx_2x', test_probs_cx2)]:
    assert abs(arr.sum(axis=1).mean() - 1.0) < 1e-4, f'{name}: row sums not ≈ 1!'
print('✅ Row sums verified for all variants')

```

    Ensemble weight comparison:
      Arch               equal    1.5×cx      2×cx
      ------------------------------------------------
      convnext          0.2000    0.2500    0.3333  ← upweighted
      eva02             0.2000    0.1875    0.1667
      dinov2            0.2000    0.1875    0.1667
      effnet_b4         0.2000    0.1875    0.1667
      swinv2            0.2000    0.1875    0.1667
      SUM               1.0000    1.0000    1.0000
    
      1.5× convnext OOF F1: 0.9221  (equal=0.9252, Δ=-0.0031)
        ↳ OOF likely lower because convnext regressed in exp07 training.
        ↳ LB evidence supersedes OOF here — proceed anyway.
      2× convnext OOF F1: 0.9210  (equal=0.9252, Δ=-0.0042)
        ↳ OOF likely lower because convnext regressed in exp07 training.
        ↳ LB evidence supersedes OOF here — proceed anyway.
    
    ✅ Row sums verified for all variants



```python
# ── Agreement analysis vs all5_argmax (current best LB) ──────────────────────
print('Prediction agreement vs all5_argmax (LB=0.79207):')
print(f'  {"Variant":<25}  {"Agree":>8}  {"Disagree":>10}  Note')
print('  ' + '-' * 65)

for label, arr in [
    ('scipy_opt  (slot 1)', test_probs_scipy),
    ('cx_1p5     (slot 3)', test_probs_cx1p5),
    ('cx_2x      (slot 2)', test_probs_cx2),
]:
    preds = arr.argmax(axis=1)
    agree = (preds == preds_all5).mean()
    ndiff = (preds != preds_all5).sum()
    print(f'  {label:<25}  {agree:>8.4f}  {ndiff:>10}  samples')

print()
print('Interpretation:')
print('  Higher disagreement = more distinct predictions = more information to LB.')
print('  But disagreement on samples the model GETS RIGHT hurts — we cannot know which.')
print('  These numbers give a sense of how much each variant diverges from current best.')

```

    Prediction agreement vs all5_argmax (LB=0.79207):
      Variant                       Agree    Disagree  Note
      -----------------------------------------------------------------
      scipy_opt  (slot 1)          0.9901           4  samples
      cx_1p5     (slot 3)          0.9950           2  samples
      cx_2x      (slot 2)          0.9876           5  samples
    
    Interpretation:
      Higher disagreement = more distinct predictions = more information to LB.
      But disagreement on samples the model GETS RIGHT hurts — we cannot know which.
      These numbers give a sense of how much each variant diverges from current best.


## 5. Generate Submissions

Three files, one per remaining slot.

**Submission order recommendation:**
1. `scipy_opt_argmax` first — most principled, independent of LB evidence
2. `convnext_2x_argmax` second — sharpest test of the convnext hypothesis
3. `convnext_1p5x_argmax` third — bracket; if 2× hurts, this tells us the optimal is between 1× and 2×

> **Before submitting:** Confirm `all5_argmax` is still selected as your final submission
> in case these don't beat it on the private LB.



```python
submissions = {}

# 1. Scipy-optimized weights
submissions['scipy_opt_argmax'] = make_submission(
    test_probs_scipy, test_df if 'test_df' in dir() else None,
    thresholds=None,
    name='scipy_opt_argmax',
)

# 2. Convnext 2× upweighted
submissions['convnext_2x_argmax'] = make_submission(
    test_probs_cx2, test_df if 'test_df' in dir() else None,
    thresholds=None,
    name='convnext_2x_argmax',
)

# 3. Convnext 1.5× upweighted
submissions['convnext_1p5x_argmax'] = make_submission(
    test_probs_cx1p5, test_df if 'test_df' in dir() else None,
    thresholds=None,
    name='convnext_1p5x_argmax',
)

print('Submission files generated:')
for name, result in submissions.items():
    sub_df, _ = result
    dist  = dict(sub_df['label'].value_counts().sort_index())
    saved = '✅' if (SUBMISSION_DIR / f'{name}.csv').exists() else '⚠️ not found'
    print(f'\n  {name}:')
    print(f'    Rows        : {len(sub_df)}')
    print(f'    Distribution: {dist}')
    print(f'    Saved       : {saved}')

```

    Submission files generated:
    
      scipy_opt_argmax:
        Rows        : 404
        Distribution: {'fake_mannequin': 51, 'fake_mask': 72, 'fake_printed': 54, 'fake_screen': 67, 'fake_unknown': 53, 'realperson': 107}
        Saved       : ✅
    
      convnext_2x_argmax:
        Rows        : 404
        Distribution: {'fake_mannequin': 52, 'fake_mask': 75, 'fake_printed': 55, 'fake_screen': 65, 'fake_unknown': 52, 'realperson': 105}
        Saved       : ✅
    
      convnext_1p5x_argmax:
        Rows        : 404
        Distribution: {'fake_mannequin': 52, 'fake_mask': 72, 'fake_printed': 55, 'fake_screen': 65, 'fake_unknown': 52, 'realperson': 108}
        Saved       : ✅


## 6. Summary & Submission Order


```python
print('=' * 72)
print('RESUBMISSION STRATEGY — exp07 (LB-Informed)')
print('=' * 72)
print()
print('PREVIOUS LB RESULTS (from 11-inference-exp07):')
for name, lb in sorted(LB_RESULTS.items(), key=lambda x: -x[1]):
    marker = '  ← current best' if name == 'all5_argmax' else ''
    print(f'  {name:<30}: LB={lb:.5f}{marker}')
print()
print('NEW SUBMISSIONS:')
slot_info = {
    'scipy_opt_argmax'     : ('Slot 1', 'scipy weights, OOF-optimized'),
    'convnext_2x_argmax'   : ('Slot 2', f'convnext w=0.333, others w=0.167'),
    'convnext_1p5x_argmax' : ('Slot 3', f'convnext w=0.250, others w=0.188'),
}
for name, (slot, desc) in slot_info.items():
    saved = '✅' if (SUBMISSION_DIR / f'{name}.csv').exists() else '⚠️'
    print(f'  {slot}: {name:<30} {saved}')
    print(f'         {desc}')
print()
print('WEIGHT SUMMARY:')
print(f'  {"Arch":<14}  {"equal (LB=0.792)":>18}  {"1.5× cx":>10}  {"2× cx":>8}  {"scipy":>8}')
print('  ' + '-' * 68)
for i, arch in enumerate(ALL_ARCHS):
    w_eq  = 0.20
    w_1p5 = weights_cx1p5[arch]
    w_2x  = weights_cx2[arch]
    w_sc  = best_w[i]
    print(f'  {arch:<14}  {w_eq:>18.4f}  {w_1p5:>10.4f}  {w_2x:>8.4f}  {w_sc:>8.4f}')
print()
print('SUBMISSION ORDER:')
print('  1st → scipy_opt_argmax     (principled OOF-maximizing weights)')
print('  2nd → convnext_2x_argmax   (sharpest test of convnext-up hypothesis)')
print('  3rd → convnext_1p5x_argmax (bracket between 1× and 2×)')
print()
print('⚠️  IMPORTANT: Keep all5_argmax selected as your final submission')
print('   unless one of these beats 0.79207 on the public LB.')
print()
print('INTERPRETATION GUIDE (after seeing results):')
print('  scipy > all5        → OOF-weight direction works; trust OOF more for future')
print('  cx_2x > all5        → convnext even more valuable than LB gap suggested')
print('  cx_1p5 > all5 > cx_2x → optimal convnext weight is between 0.25 and 0.333')
print('  all5 > everything   → equal weights are optimal; convnext effect saturates at 0.20')
print(f'{"="*72}')

```

    ========================================================================
    RESUBMISSION STRATEGY — exp07 (LB-Informed)
    ========================================================================
    
    PREVIOUS LB RESULTS (from 11-inference-exp07):
      all5_argmax                   : LB=0.79207  ← current best
      top4_argmax                   : LB=0.78673
      pruned_mega_argmax            : LB=0.78555
    
    NEW SUBMISSIONS:
      Slot 1: scipy_opt_argmax               ✅
             scipy weights, OOF-optimized
      Slot 2: convnext_2x_argmax             ✅
             convnext w=0.333, others w=0.167
      Slot 3: convnext_1p5x_argmax           ✅
             convnext w=0.250, others w=0.188
    
    WEIGHT SUMMARY:
      Arch              equal (LB=0.792)     1.5× cx     2× cx     scipy
      --------------------------------------------------------------------
      convnext                    0.2000      0.2500    0.3333    0.0209
      eva02                       0.2000      0.1875    0.1667    0.2506
      dinov2                      0.2000      0.1875    0.1667    0.2803
      effnet_b4                   0.2000      0.1875    0.1667    0.2348
      swinv2                      0.2000      0.1875    0.1667    0.2133
    
    SUBMISSION ORDER:
      1st → scipy_opt_argmax     (principled OOF-maximizing weights)
      2nd → convnext_2x_argmax   (sharpest test of convnext-up hypothesis)
      3rd → convnext_1p5x_argmax (bracket between 1× and 2×)
    
    ⚠️  IMPORTANT: Keep all5_argmax selected as your final submission
       unless one of these beats 0.79207 on the public LB.
    
    INTERPRETATION GUIDE (after seeing results):
      scipy > all5        → OOF-weight direction works; trust OOF more for future
      cx_2x > all5        → convnext even more valuable than LB gap suggested
      cx_1p5 > all5 > cx_2x → optimal convnext weight is between 0.25 and 0.333
      all5 > everything   → equal weights are optimal; convnext effect saturates at 0.20
    ========================================================================



```python

```


```python

```
