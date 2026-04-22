~/ML/Competition/FindIT-DAC/
├── __pycache__/
├── archive/
├── configs/
│   └── experiments/
│   └── base.yaml
├── data/
│   ├── interim/
│   │   ├── train/
│   │   │   ├── fake_mannequin/
│   │   │   ├── fake_mask/
│   │   │   ├── fake_printed/
│   │   │   ├── fake_screen/
│   │   │   ├── fake_unknown/
│   │   │   └── realperson/           # ← images inside (mannequin_*.jpg etc.)
│   │   └── test/
│   ├── processed/
│   ├── raw/
│   └── zip/
├── models/
│   ├── exp01/
│   └── exp02/
│       └── *.pth (many folds)
│           ├── best_convnext_*.pth
│           ├── best_dinov2_*.pth
│           ├── best_effnet_b4_*.pth
│           ├── best_eva02_*.pth
│           └── dinov2_probe_*.pth
├── nb-exports/
│   ├── old/
│   ├── v1/
│   └── v2/
│       ├── 02-data-preparation-v2_files.md
│       └── 03-full-training-v2.md
├── notebooks/
│   ├── analysis/
│   ├── eda/
│   ├── experiments/
│   ├── inference/
│   ├── preprocessing/
│   └── scratch/
├── oof/
│   ├── exp01/
│   └── exp02/
│       └── oof_*.csv (for convnext, dinov2, effnet, eva02…)
├── references/
│   ├── log/
│   └── research-planning/
│       ├── FAS_StratifiedGroupKFold_DeepResearch.md
│       ├── fas_training_deep_dive.md
│       └── ... (audit & planning docs)
├── scripts/
│   ├── sync_from_remote.sh
│   └── sync_to_remote.sh
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
│       ├── config.py
│       └── seed.py
├── submissions/
│   ├── submission.csv
│   ├── submission_ensemble_*.csv
│   ├── v2_*.csv (top2, all4, dinov2, thresh versions…)
│   └── sub_*.csv (raw, no_leaked, etc.)
├── README.md
└── requirements.txt
