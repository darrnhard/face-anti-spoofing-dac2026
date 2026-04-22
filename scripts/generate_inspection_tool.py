"""
generate_inspection_tool.py

Run from project root:
    python scripts/generate_inspection_tool.py

Output: reports/test_inspection.html

Features:
  - Visual inspection of all test predictions across 3 submissions
  - MD5-based leaked label detection (test crops matched against train crops)
  - Full probability table per image
  - Manual mark: correct / wrong / flag
  - Export corrections.csv
"""

import sys
import base64
import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Hardcoded paths (no torch dependency) ────────────────────────────────────
CROP_TRAIN_DIR = PROJECT_ROOT / 'data/processed/crops/train'
CROP_TEST_DIR  = PROJECT_ROOT / 'data/processed/crops/test'
OOF_DIR        = PROJECT_ROOT / 'oof'
SUBMISSION_DIR = PROJECT_ROOT / 'submissions'
REPORTS_DIR    = PROJECT_ROOT / 'reports'
CLASSES = ['fake_mannequin', 'fake_mask', 'fake_printed',
           'fake_screen', 'fake_unknown', 'realperson']

EXP_ID   = 'exp03'
PROB_DIR = OOF_DIR / EXP_ID

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — MD5 leaked label detection
# ─────────────────────────────────────────────────────────────────────────────
def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

print('=' * 60)
print('STEP 1 — MD5 leaked label detection')
print('=' * 60)

print('Hashing train crops...')
train_hash_map = {}
train_crops = list(CROP_TRAIN_DIR.glob('*.jpg')) + list(CROP_TRAIN_DIR.glob('*.png'))
for i, path in enumerate(train_crops):
    md5   = md5_file(path)
    name  = path.stem
    label = next((c for c in CLASSES if name.startswith(c)), None)
    train_hash_map[md5] = {'train_file': path.name, 'leaked_label': label}
    if (i + 1) % 200 == 0:
        print(f'  {i+1}/{len(train_crops)} train crops hashed...')

print(f'  Total train crops hashed: {len(train_crops)}')

print('Hashing test crops and matching...')
test_crops = list(CROP_TEST_DIR.glob('*.jpg')) + list(CROP_TEST_DIR.glob('*.png'))
leaked_map = {}
for path in test_crops:
    img_id = path.stem
    md5    = md5_file(path)
    if md5 in train_hash_map:
        leaked_map[img_id] = train_hash_map[md5]

n_leaked = len(leaked_map)
print(f'  Total test crops : {len(test_crops)}')
print(f'  Leaked (matched) : {n_leaked} ({n_leaked/max(len(test_crops),1)*100:.1f}%)')
if leaked_map:
    print('  Sample matches:')
    for img_id, info in list(leaked_map.items())[:5]:
        print(f'    {img_id} -> {info["train_file"]}  label={info["leaked_label"]}')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load submission CSVs
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('STEP 2 — Loading submission CSVs')
print('=' * 60)

subs = {}
for name in ['top3_argmax', 'all4_argmax', 'dinov2_argmax',
             'all4_thresh', 'convnext_thresh']:
    path = SUBMISSION_DIR / f'{name}.csv'
    if not path.exists():
        continue
    subs[name] = pd.read_csv(path).set_index('id')
    print(f'  {name}: {len(subs[name])} rows')

if not subs:
    raise FileNotFoundError(f'No submission CSVs found in {SUBMISSION_DIR}')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Load probability arrays
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('STEP 3 — Loading probability arrays')
print('=' * 60)

prob_map = {
    'top3_argmax'    : 'test_probs_top3.npy',
    'all4_argmax'    : 'test_probs_all4.npy',
    'dinov2_argmax'  : 'test_probs_dinov2.npy',
    'all4_thresh'    : 'test_probs_all4.npy',
    'convnext_thresh': 'test_probs_convnext.npy',
}
probs = {}
for name in subs:
    fname = prob_map.get(name)
    if fname:
        path = PROB_DIR / fname
        if path.exists():
            probs[name] = np.load(path)
            print(f'  {name}: {probs[name].shape}')
        else:
            print(f'  WARNING: {fname} not found — confidence N/A for {name}')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Build per-image records
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('STEP 4 — Building image records')
print('=' * 60)

sub_names   = list(subs.keys())
primary_sub = next(iter(subs.values()))
image_ids   = list(primary_sub.index)

records = []
for i, img_id in enumerate(image_ids):
    img_path = CROP_TEST_DIR / f'{img_id}.jpg'
    if not img_path.exists():
        img_path = CROP_TEST_DIR / f'{img_id}.png'
    if not img_path.exists():
        print(f'  WARNING: image not found for {img_id}')
        continue

    with open(img_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    predictions = {}
    confidences = {}
    all_probs   = {}
    for name, sub_df in subs.items():
        pred = sub_df.loc[img_id, 'label'] if img_id in sub_df.index else 'N/A'
        predictions[name] = pred
        if name in probs:
            prob_row          = probs[name][i].tolist()
            confidences[name] = round(float(max(prob_row)), 4)
            all_probs[name]   = {cls: round(float(p), 4)
                                  for cls, p in zip(CLASSES, prob_row)}
        else:
            confidences[name] = None
            all_probs[name]   = {}

    leak_info    = leaked_map.get(img_id, {})
    leaked_label = leak_info.get('leaked_label', None)
    leaked_file  = leak_info.get('train_file', None)
    has_leak     = leaked_label is not None

    pred_set     = set(predictions.values())
    agrees       = len(pred_set) == 1

    primary_pred = predictions.get('top3_argmax') or predictions.get(sub_names[0])
    primary_conf = confidences.get('top3_argmax') or confidences.get(sub_names[0])
    leak_correct = (primary_pred == leaked_label) if has_leak else None

    records.append({
        'id'          : img_id,
        'img_b64'     : img_b64,
        'predictions' : predictions,
        'confidences' : confidences,
        'all_probs'   : all_probs,
        'agrees'      : agrees,
        'primary_pred': primary_pred,
        'primary_conf': primary_conf,
        'leaked_label': leaked_label,
        'leaked_file' : leaked_file,
        'has_leak'    : has_leak,
        'leak_correct': leak_correct,
        'manual_label': '',
        'manual_note' : '',
    })

    if (i + 1) % 50 == 0:
        print(f'  Processed {i+1}/{len(image_ids)}...')

leaked_records = [r for r in records if r['has_leak']]
n_leak_err = sum(1 for r in leaked_records if not r['leak_correct'])
if leaked_records:
    n_correct_leak = sum(1 for r in leaked_records if r['leak_correct'])
    print(f'\nLeaked label accuracy (primary submission):')
    print(f'  {n_correct_leak}/{len(leaked_records)} correct '
          f'({n_correct_leak/len(leaked_records)*100:.1f}%)')
    from collections import Counter
    errors = [(r['leaked_label'], r['primary_pred'])
              for r in leaked_records if not r['leak_correct']]
    if errors:
        print('  Errors (true -> predicted):')
        for (true, pred), cnt in Counter(errors).most_common():
            print(f'    {true:<22} -> {pred:<22}: {cnt}')

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Generate HTML
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('STEP 5 — Generating HTML')
print('=' * 60)

n_disagree = sum(1 for r in records if not r['agrees'])
data_json  = json.dumps(records)

cls_filter_btns = ''.join(
    f'<button class="btn" onclick="setFilter(\'cls_{c}\',this)">'
    f'{c.replace("fake_","")}</button>'
    for c in CLASSES
)
sub_headers = ''.join(f'<th>{s}</th>' for s in sub_names)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FAS Test Inspection — exp03</title>
<style>
  :root {{
    --bg:#1a1a2e; --card:#16213e; --card2:#0f3460;
    --accent:#e94560; --text:#eaeaea; --muted:#888; --border:#2a2a4a;
    --mannequin:#9c27b0; --mask:#ff9800; --printed:#f44336;
    --screen:#2196f3; --unknown:#607d8b; --real:#4caf50; --leak:#00bcd4;
  }}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;}}
  .header{{background:var(--card2);padding:14px 22px;border-bottom:1px solid var(--border);
           display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;}}
  .header h1{{font-size:1rem;font-weight:700;}}
  .stats{{display:flex;gap:14px;font-size:0.8rem;color:var(--muted);flex-wrap:wrap;}}
  .stats b{{color:var(--text);}}
  .controls{{padding:10px 22px;background:var(--card);border-bottom:1px solid var(--border);
             display:flex;gap:6px;align-items:center;flex-wrap:wrap;}}
  .btn{{padding:5px 12px;border-radius:6px;border:1px solid var(--border);
        background:var(--card2);color:var(--text);cursor:pointer;font-size:0.78rem;
        transition:all 0.15s;white-space:nowrap;}}
  .btn:hover{{border-color:#666;}}
  .btn.active{{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:700;}}
  .sep{{width:1px;height:22px;background:var(--border);margin:0 3px;}}
  .search{{padding:5px 10px;border-radius:6px;border:1px solid var(--border);
           background:var(--card2);color:var(--text);font-size:0.78rem;width:140px;}}
  .cnt{{font-size:0.7rem;background:rgba(255,255,255,0.08);border-radius:8px;
        padding:1px 6px;margin-left:3px;}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(195px,1fr));
         gap:10px;padding:14px 22px;}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:10px;
         overflow:hidden;cursor:pointer;transition:transform 0.15s,border-color 0.15s;}}
  .card:hover{{transform:translateY(-2px);border-color:#555;}}
  .card.disagree  {{border-color:#ff9800;}}
  .card.low-conf  {{border-color:#607d8b;}}
  .card.has-leak  {{border-color:var(--leak);}}
  .card.leak-wrong{{border-color:var(--accent);box-shadow:0 0 8px rgba(233,69,96,0.4);}}
  .card.flagged   {{border-color:#ff9800;}}
  .card.correct   {{border-color:#4caf50;}}
  .card.wrong     {{border-color:#f44336;}}
  .card img{{width:100%;aspect-ratio:1;object-fit:cover;display:block;}}
  .card-body{{padding:7px;}}
  .card-id{{font-size:0.68rem;color:var(--muted);margin-bottom:3px;
            display:flex;align-items:center;gap:4px;flex-wrap:wrap;}}
  .card-pred{{font-size:0.74rem;font-weight:700;margin-bottom:2px;}}
  .card-conf{{font-size:0.68rem;color:var(--muted);margin-bottom:5px;}}
  .leak-row{{font-size:0.68rem;margin-bottom:5px;padding:3px 5px;border-radius:4px;
             background:rgba(0,188,212,0.12);border:1px solid rgba(0,188,212,0.3);}}
  .leak-match{{color:#4caf50;font-weight:700;}}
  .leak-nomatch{{color:var(--accent);font-weight:700;}}
  .preds-row{{display:flex;gap:3px;flex-wrap:wrap;margin-bottom:5px;}}
  .pred-chip{{font-size:0.60rem;padding:2px 4px;border-radius:4px;
              background:var(--card2);border:1px solid var(--border);}}
  .manual-row{{display:flex;gap:3px;}}
  .manual-btn{{flex:1;padding:3px;font-size:0.65rem;border-radius:4px;
               border:1px solid var(--border);background:var(--card2);
               color:var(--text);cursor:pointer;text-align:center;}}
  .manual-btn:hover{{border-color:#aaa;}}
  .manual-btn.sel-correct{{background:#1b5e20;border-color:#4caf50;}}
  .manual-btn.sel-wrong{{background:#b71c1c;border-color:#f44336;}}
  .manual-btn.sel-flag{{background:#e65100;border-color:#ff9800;}}
  .badge{{font-size:0.6rem;border-radius:3px;padding:1px 4px;font-weight:700;}}
  .badge-disagree{{background:#e65100;color:#fff;}}
  .badge-leak{{background:var(--leak);color:#000;}}
  .badge-leakerr{{background:var(--accent);color:#fff;}}
  .overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.8);
            z-index:100;align-items:center;justify-content:center;}}
  .overlay.open{{display:flex;}}
  .modal{{background:var(--card);border:1px solid var(--border);border-radius:12px;
          width:min(740px,96vw);max-height:92vh;overflow-y:auto;padding:22px;}}
  .modal h2{{font-size:0.95rem;margin-bottom:14px;}}
  .modal-top{{display:flex;gap:14px;margin-bottom:14px;flex-wrap:wrap;}}
  .modal-top img{{flex:1;min-width:200px;max-height:320px;object-fit:contain;
                  border-radius:8px;background:#000;}}
  .leak-box{{flex:1;min-width:180px;background:rgba(0,188,212,0.08);
             border:1px solid rgba(0,188,212,0.3);border-radius:8px;
             padding:12px;font-size:0.8rem;}}
  .leak-box h3{{font-size:0.8rem;color:var(--leak);margin-bottom:8px;}}
  .prob-table{{width:100%;border-collapse:collapse;font-size:0.78rem;margin-bottom:14px;}}
  .prob-table th{{text-align:left;color:var(--muted);padding:4px 8px;
                  border-bottom:1px solid var(--border);font-weight:500;}}
  .prob-table td{{padding:5px 8px;border-bottom:1px solid var(--border);}}
  .bar{{height:5px;border-radius:3px;display:inline-block;
        margin-left:6px;vertical-align:middle;background:var(--accent);}}
  .modal-note{{width:100%;background:var(--card2);border:1px solid var(--border);
               border-radius:6px;padding:8px;color:var(--text);font-size:0.8rem;
               resize:vertical;min-height:50px;margin-bottom:10px;}}
  .modal-actions{{display:flex;gap:7px;flex-wrap:wrap;}}
  .modal-close{{margin-left:auto;background:none;border:1px solid var(--border);
                color:var(--muted);border-radius:6px;padding:5px 12px;cursor:pointer;}}
  .export-bar{{position:fixed;bottom:0;left:0;right:0;background:var(--card2);
               border-top:1px solid var(--border);padding:9px 22px;
               display:flex;align-items:center;gap:10px;font-size:0.8rem;z-index:50;}}
  .export-btn{{padding:6px 16px;background:var(--accent);border:none;border-radius:6px;
               color:#fff;font-weight:700;cursor:pointer;font-size:0.8rem;}}
  .no-results{{grid-column:1/-1;text-align:center;color:var(--muted);padding:48px;}}
</style>
</head>
<body>

<div class="header">
  <h1>FAS Test Inspection — exp03
    <span style="font-weight:400;color:var(--muted);font-size:0.82rem">
      &nbsp;{len(records)} images &middot; {len(sub_names)} submissions
    </span>
  </h1>
  <div class="stats">
    Disagree <b style="color:#ff9800">{n_disagree}</b> &nbsp;|&nbsp;
    Leaked <b style="color:var(--leak)">{n_leaked}</b> &nbsp;|&nbsp;
    Leak errors <b style="color:var(--accent)">{n_leak_err}</b> &nbsp;|&nbsp;
    Reviewed <b id="stat-reviewed">0</b>
  </div>
</div>

<div class="controls">
  <button class="btn active" onclick="setFilter('all',this)">All <span class="cnt">{len(records)}</span></button>
  <button class="btn" onclick="setFilter('disagree',this)">Disagree <span class="cnt">{n_disagree}</span></button>
  <button class="btn" onclick="setFilter('leaked',this)">&#x1F511; Leaked <span class="cnt">{n_leaked}</span></button>
  <button class="btn" onclick="setFilter('leak_wrong',this)">&#x1F511;&#x274C; Leak errors <span class="cnt">{n_leak_err}</span></button>
  <button class="btn" onclick="setFilter('low_conf',this)">Low conf</button>
  <button class="btn" onclick="setFilter('flagged',this)">&#x1F6A9; Flagged <span class="cnt" id="cnt-flagged">0</span></button>
  <div class="sep"></div>
  {cls_filter_btns}
  <div class="sep"></div>
  <input class="search" type="text" placeholder="Search ID..." oninput="setSearch(this.value)">
</div>

<div class="grid" id="grid"></div>
<div style="height:52px"></div>

<div class="export-bar">
  <div><span id="export-count">0</span> reviewed</div>
  <button class="export-btn" onclick="exportCSV(false)">&#x2B07; Export reviewed</button>
  <button class="btn" onclick="exportCSV(true)">&#x2B07; Export all</button>
  <div style="margin-left:auto;color:var(--muted);font-size:0.73rem">
    Cyan border = leaked label known &nbsp;|&nbsp;
    Red glow = model wrong on leaked &nbsp;|&nbsp;
    Orange = submissions disagree
  </div>
</div>

<div class="overlay" id="overlay" onclick="closeModal(event)">
<div class="modal" id="modal">
  <h2 id="modal-title"></h2>
  <div class="modal-top">
    <img id="modal-img" src="" alt="">
    <div id="modal-leak-box" class="leak-box" style="display:none">
      <h3>&#x1F511; Leaked Label Found</h3>
      <div id="modal-leak-content"></div>
    </div>
  </div>
  <table class="prob-table">
    <thead><tr><th>Class</th>{sub_headers}</tr></thead>
    <tbody id="prob-tbody"></tbody>
  </table>
  <div style="font-size:0.78rem;color:var(--muted);margin-bottom:6px">Notes:</div>
  <textarea class="modal-note" id="modal-note" placeholder="Optional note..."></textarea>
  <div class="modal-actions">
    <button class="btn" style="background:#1b5e20;border-color:#4caf50"
            onclick="modalMark('correct')">&#x2705; Correct</button>
    <button class="btn" style="background:#b71c1c;border-color:#f44336"
            onclick="modalMark('wrong')">&#x274C; Wrong</button>
    <button class="btn" style="background:#e65100;border-color:#ff9800"
            onclick="modalMark('flag')">&#x1F6A9; Flag</button>
    <button class="modal-close" onclick="closeModal()">Close</button>
  </div>
</div>
</div>

<script>
const CLASSES   = {json.dumps(CLASSES)};
const SUB_NAMES = {json.dumps(sub_names)};
const DATA      = {data_json};

let state        = DATA.map(r => ({{...r, manual_label:'', manual_note:''}}));
let activeFilter = 'all';
let activeSearch = '';
let modalIdx     = null;

const CLS_COLOR = {{
  fake_mannequin:'#9c27b0', fake_mask:'#ff9800', fake_printed:'#f44336',
  fake_screen:'#2196f3', fake_unknown:'#607d8b', realperson:'#4caf50',
}};
const cc = cls => CLS_COLOR[cls] || '#eaeaea';
const sc = cls => cls ? cls.replace('fake_','') : 'N/A';

function filtered() {{
  return state.filter(r => {{
    if (activeSearch && !r.id.includes(activeSearch)) return false;
    switch(activeFilter) {{
      case 'all'       : return true;
      case 'disagree'  : return !r.agrees;
      case 'leaked'    : return r.has_leak;
      case 'leak_wrong': return r.has_leak && !r.leak_correct;
      case 'low_conf'  : return r.primary_conf !== null && r.primary_conf < 0.65;
      case 'flagged'   : return r.manual_label === 'flag';
      default:
        if (activeFilter.startsWith('cls_'))
          return r.primary_pred === activeFilter.slice(4);
        return true;
    }}
  }});
}}

function setFilter(f, btn) {{
  activeFilter = f;
  document.querySelectorAll('.controls .btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  renderGrid();
}}
function setSearch(v) {{ activeSearch = v.trim(); renderGrid(); }}

function renderGrid() {{
  const data = filtered();
  const grid = document.getElementById('grid');
  if (!data.length) {{
    grid.innerHTML = '<div class="no-results">No images match this filter.</div>';
    updateStats(); return;
  }}
  grid.innerHTML = data.map(r => {{
    const gi   = state.indexOf(r);
    const conf = r.primary_conf !== null ? (r.primary_conf*100).toFixed(1)+'%' : 'N/A';
    const classes = ['card'];
    if (r.has_leak && !r.leak_correct) classes.push('leak-wrong');
    else if (r.has_leak)  classes.push('has-leak');
    else if (!r.agrees)   classes.push('disagree');
    if (r.primary_conf !== null && r.primary_conf < 0.65) classes.push('low-conf');
    if (r.manual_label === 'flag')    classes.push('flagged');
    if (r.manual_label === 'correct') classes.push('correct');
    if (r.manual_label === 'wrong')   classes.push('wrong');

    let badges = '';
    if (r.has_leak) {{
      badges += r.leak_correct
        ? `<span class="badge badge-leak">&#x1F511;&#x2705; ${{sc(r.leaked_label)}}</span>`
        : `<span class="badge badge-leakerr">&#x1F511;&#x274C; ${{sc(r.leaked_label)}}</span>`;
    }} else if (!r.agrees) {{
      badges += '<span class="badge badge-disagree">disagree</span>';
    }}

    let leakRow = '';
    if (r.has_leak) {{
      const mc = r.leak_correct ? 'leak-match' : 'leak-nomatch';
      const mt = r.leak_correct ? 'correct' : `should be ${{sc(r.leaked_label)}}`;
      leakRow = `<div class="leak-row">
        &#x1F511; true: <b style="color:${{cc(r.leaked_label)}}">${{sc(r.leaked_label)}}</b>
        &nbsp;<span class="${{mc}}">${{mt}}</span></div>`;
    }}

    const chips = SUB_NAMES.map(n => {{
      const p = r.predictions[n] || 'N/A';
      const c = r.confidences[n] ? (r.confidences[n]*100).toFixed(0)+'%' : '';
      return `<span class="pred-chip" style="border-color:${{cc(p)}}">${{sc(p)}} ${{c}}</span>`;
    }}).join('');

    const mb = v => r.manual_label===v ? `manual-btn sel-${{v}}` : 'manual-btn';
    return `
    <div class="${{classes.join(' ')}}" onclick="openModal(${{gi}})">
      <img src="data:image/jpeg;base64,${{r.img_b64}}" loading="lazy" alt="${{r.id}}">
      <div class="card-body">
        <div class="card-id">${{r.id}} ${{badges}}</div>
        <div class="card-pred" style="color:${{cc(r.primary_pred)}}">${{sc(r.primary_pred)}}</div>
        <div class="card-conf">${{conf}}</div>
        ${{leakRow}}
        <div class="preds-row">${{chips}}</div>
        <div class="manual-row" onclick="event.stopPropagation()">
          <div class="${{mb('correct')}}" onclick="mark(${{gi}},'correct')">&#x2705;</div>
          <div class="${{mb('wrong')}}"   onclick="mark(${{gi}},'wrong')">&#x274C;</div>
          <div class="${{mb('flag')}}"    onclick="mark(${{gi}},'flag')">&#x1F6A9;</div>
        </div>
      </div>
    </div>`;
  }}).join('');
  updateStats();
}}

function updateStats() {{
  const rev = state.filter(r => r.manual_label !== '').length;
  document.getElementById('stat-reviewed').textContent = rev;
  document.getElementById('export-count').textContent  = rev;
  document.getElementById('cnt-flagged').textContent   =
    state.filter(r => r.manual_label === 'flag').length;
}}

function mark(idx, label) {{
  state[idx].manual_label = state[idx].manual_label === label ? '' : label;
  renderGrid();
}}

function openModal(idx) {{
  modalIdx = idx;
  const r  = state[idx];
  document.getElementById('modal-img').src = 'data:image/jpeg;base64,' + r.img_b64;

  let title = `${{r.id}} \u2014 <span style="color:${{cc(r.primary_pred)}}">${{sc(r.primary_pred)}}</span>`;
  if (!r.agrees) title += ' <span class="badge badge-disagree">disagree</span>';
  document.getElementById('modal-title').innerHTML = title;

  const leakBox = document.getElementById('modal-leak-box');
  if (r.has_leak) {{
    leakBox.style.display = 'block';
    const mc = r.leak_correct ? '#4caf50' : '#e94560';
    const mt = r.leak_correct
      ? '&#x2705; Primary prediction is CORRECT'
      : `&#x274C; WRONG \u2014 true label is <b style="color:${{cc(r.leaked_label)}}">${{sc(r.leaked_label)}}</b>`;
    document.getElementById('modal-leak-content').innerHTML = `
      <div style="margin-bottom:6px">
        True label: <b style="color:${{cc(r.leaked_label)}};font-size:1rem">${{sc(r.leaked_label)}}</b>
      </div>
      <div style="margin-bottom:6px;font-size:0.72rem">
        Matched: <code>${{r.leaked_file}}</code></div>
      <div style="color:${{mc}}">${{mt}}</div>
      <div style="margin-top:10px;font-size:0.72rem;color:#888">
        Per submission:<br>
        ${{SUB_NAMES.map(n =>
          `${{n}}: <b style="color:${{cc(r.predictions[n])}}">${{sc(r.predictions[n])}}</b>`
        ).join('<br>')}}
      </div>`;
  }} else {{
    leakBox.style.display = 'none';
  }}

  document.getElementById('modal-note').value = r.manual_note || '';

  const tbody = document.getElementById('prob-tbody');
  tbody.innerHTML = CLASSES.map(cls => {{
    const cells = SUB_NAMES.map(n => {{
      const p = r.all_probs[n] ? r.all_probs[n][cls] : null;
      if (p === null) return '<td style="color:#555">N/A</td>';
      const bar  = `<span class="bar" style="width:${{Math.round(p*70)}}px"></span>`;
      const bold = cls === r.predictions[n] ? 'font-weight:700' : '';
      const und  = cls === r.leaked_label   ? 'text-decoration:underline' : '';
      return `<td style="${{bold}};${{und}}">${{(p*100).toFixed(1)}}%${{bar}}</td>`;
    }}).join('');
    const hi  = cls===r.primary_pred || cls===r.leaked_label
      ? 'background:rgba(255,255,255,0.04)' : '';
    const lk  = cls===r.leaked_label ? ' &#x1F511;' : '';
    return `<tr style="${{hi}}">
      <td style="color:${{cc(cls)}}">${{sc(cls)}}${{lk}}</td>${{cells}}</tr>`;
  }}).join('');

  document.getElementById('overlay').classList.add('open');
}}

function closeModal(e) {{
  if (e && e.target !== document.getElementById('overlay')) return;
  _saveNote();
  document.getElementById('overlay').classList.remove('open');
  modalIdx = null;
  renderGrid();
}}
function modalMark(label) {{
  if (modalIdx===null) return;
  _saveNote(); mark(modalIdx, label);
  document.getElementById('overlay').classList.remove('open');
  modalIdx = null;
}}
function _saveNote() {{
  if (modalIdx !== null)
    state[modalIdx].manual_note = document.getElementById('modal-note').value;
}}

function exportCSV(all=false) {{
  const rows = all ? state : state.filter(r => r.manual_label !== '');
  if (!rows.length) {{ alert('Nothing to export.'); return; }}
  const hdr = ['id','primary_pred','primary_conf','agrees','leaked_label',
               'leak_correct',...SUB_NAMES.map(n=>'pred_'+n),
               'manual_label','manual_note'];
  const csv = [hdr.join(','), ...rows.map(r => [
    r.id, r.primary_pred, r.primary_conf??'', r.agrees,
    r.leaked_label??'', r.leak_correct??'',
    ...SUB_NAMES.map(n => r.predictions[n]||''),
    r.manual_label, '"'+(r.manual_note||'').replace(/"/g,'""')+'"',
  ].join(','))].join('\\n');
  const a = document.createElement('a');
  a.href = 'data:text/csv;charset=utf-8,'+encodeURIComponent(csv);
  a.download = 'test_corrections.csv';
  a.click();
}}

renderGrid();
</script>
</body>
</html>"""

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
out_path = REPORTS_DIR / 'test_inspection.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

size_mb = out_path.stat().st_size / 1_048_576
print(f'\nDone!')
print(f'  Output     : {out_path}')
print(f'  File size  : {size_mb:.1f} MB')
print(f'  Images     : {len(records)}')
print(f'  Leaked     : {n_leaked} ({n_leaked/max(len(records),1)*100:.1f}%)')
print(f'  Leak errors: {n_leak_err}')
print(f'\nOpen with:  firefox {out_path}')
