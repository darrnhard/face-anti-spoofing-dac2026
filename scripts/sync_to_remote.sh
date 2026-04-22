#!/bin/bash
# =============================================================================
# sync_to_remote.sh — Push code + processed data to GPU server (Vast.ai)
# =============================================================================
# Usage:   ./scripts/sync_to_remote.sh <remote-ip> <remote-port>
# Example: ./scripts/sync_to_remote.sh 123.456.789.0 12345
#
# What gets pushed:
#   ✓ notebooks/        ← training, inference, and analysis notebooks
#   ✓ src/              ← reusable Python modules (dataset, losses, metrics…)
#   ✓ configs/          ← base.yaml + experiment configs
#   ✓ data/processed/   ← crops/train, crops/test, train_clean.csv, test.csv
#
# What is excluded:
#   ✗ data/interim/     ← source images (raw cleaned); crops already in processed
#   ✗ data/raw/         ← original competition images; never needed on remote
#   ✗ data/zip/         ← archive zips; not needed on remote
#   ✗ models/           ← pulled back from remote, never pushed
#   ✗ oof/              ← pulled back from remote, never pushed
#   ✗ submissions/      ← pulled back from remote, never pushed
#   ✗ archive/          ← local archiving only
#   ✗ nb-exports/       ← markdown exports; not needed on remote
#   ✗ references/       ← research docs; not needed on remote
#   ✗ .git/             ← version control metadata
#   ✗ __pycache__/      ← Python bytecode cache
#   ✗ .ipynb_checkpoints/ ← Jupyter autosave artifacts
# =============================================================================

set -e

if [ $# -lt 2 ]; then
  echo "Usage: $0 <remote-ip> <remote-port>"
  echo "Example: $0 123.456.789.0 12345"
  exit 1
fi

REMOTE_IP=$1
REMOTE_PORT=$2
REMOTE_USER="root"
REMOTE_DIR="/workspace/fas-competition"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SSH_KEY="$HOME/.ssh/id_ed25519_vast"
SSH_CMD="ssh -i $SSH_KEY -p $REMOTE_PORT"

echo "============================================"
echo "SYNCING TO REMOTE"
echo "============================================"
echo "Local:  $PROJECT_DIR"
echo "Remote: $REMOTE_USER@$REMOTE_IP:$REMOTE_DIR"
echo "Port:   $REMOTE_PORT"
echo ""

# Create remote directory structure
# models/, oof/, submissions/ must exist before training writes to them
echo "--- Creating remote directory structure ---"
$SSH_CMD "$REMOTE_USER@$REMOTE_IP" "
  mkdir -p $REMOTE_DIR/data/processed/crops/train
  mkdir -p $REMOTE_DIR/data/processed/crops/test
  mkdir -p $REMOTE_DIR/models
  mkdir -p $REMOTE_DIR/oof
  mkdir -p $REMOTE_DIR/submissions
  mkdir -p $REMOTE_DIR/src
  mkdir -p $REMOTE_DIR/configs/experiments
  mkdir -p $REMOTE_DIR/notebooks
"

# Sync notebooks, src, configs, and processed data
echo ""
echo "--- Syncing notebooks/, src/, configs/, data/processed/ ---"
rsync -avz --progress \
  -e "ssh -i $SSH_KEY -p $REMOTE_PORT" \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='*.pyc' \
  --exclude='.env' \
  --exclude='data/raw/' \
  --exclude='data/interim/' \
  --exclude='data/zip/' \
  --exclude='/models/' \
  --exclude='/oof/' \
  --exclude='/submissions/' \
  --exclude='archive/' \
  --exclude='nb-exports/' \
  --exclude='references/' \
  --exclude='reports/' \
  "$PROJECT_DIR/" "$REMOTE_USER@$REMOTE_IP:$REMOTE_DIR/"

echo ""
echo "============================================"
echo "SYNC COMPLETE"
echo "============================================"
echo ""
echo "Pushed:"
echo "  notebooks/      ✓"
echo "  src/            ✓"
echo "  configs/        ✓"
echo "  data/processed/ ✓"
echo ""
echo "Connect with VS Code Remote-SSH:"
echo "  Host: $REMOTE_USER@$REMOTE_IP   Port: $REMOTE_PORT"
echo "  Open folder: $REMOTE_DIR"
echo ""
echo "Or SSH directly:"
echo "  ssh -i $SSH_KEY -p $REMOTE_PORT $REMOTE_USER@$REMOTE_IP"
