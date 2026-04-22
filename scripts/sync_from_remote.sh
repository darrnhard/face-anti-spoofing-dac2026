#!/bin/bash
# =============================================================================
# sync_from_remote.sh — Pull trained models + results back from GPU server
# =============================================================================
# Usage:   ./scripts/sync_from_remote.sh <remote-ip> <remote-port>
# Example: ./scripts/sync_from_remote.sh 123.456.789.0 12345
#
# What gets pulled (all flat — no version subdirs on remote):
#   ✓ models/      ← trained .pth checkpoints
#   ✓ oof/         ← out-of-fold prediction CSVs
#   ✓ src/         ← in case you hotfixed code on the server
#   ✓ configs/     ← in case you tweaked configs on the server
#   ✓ notebooks/   ← in case you edited notebooks on the server
#
# Overwrite protection:
#   Before pulling models/ and oof/, existing local files are automatically
#   backed up to archive/remote_pull_YYYYMMDD_HHMMSS/ via rsync --backup.
#   The backup folder is only created if files actually differ (rsync is smart).
#   src/, configs/, notebooks/ are small text files — no backup needed.
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

# Timestamped backup dir — created under archive/ so it's easy to browse later
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$PROJECT_DIR/archive/remote_pull_$TIMESTAMP"

echo "============================================"
echo "PULLING FROM REMOTE"
echo "============================================"
echo "Remote: $REMOTE_USER@$REMOTE_IP:$REMOTE_DIR"
echo "Local:  $PROJECT_DIR"
echo "Backup: $BACKUP_DIR (only populated if files differ)"
echo ""

# Pull models/ — auto-backup any local files that would be overwritten
echo "--- Pulling models/ (with auto-backup) ---"
mkdir -p "$BACKUP_DIR/models"
rsync -avz --progress \
  --backup --backup-dir="$BACKUP_DIR/models" \
  -e "ssh -i $SSH_KEY -p $REMOTE_PORT" \
  "$REMOTE_USER@$REMOTE_IP:$REMOTE_DIR/models/" \
  "$PROJECT_DIR/models/"

# Pull oof/ — auto-backup any local files that would be overwritten
echo ""
echo "--- Pulling oof/ (with auto-backup) ---"
mkdir -p "$BACKUP_DIR/oof"
rsync -avz --progress \
  --backup --backup-dir="$BACKUP_DIR/oof" \
  -e "ssh -i $SSH_KEY -p $REMOTE_PORT" \
  "$REMOTE_USER@$REMOTE_IP:$REMOTE_DIR/oof/" \
  "$PROJECT_DIR/oof/"

# Pull src/ — no backup needed (text files, git-tracked)
echo ""
echo "--- Pulling src/ ---"
rsync -avz --progress \
  -e "ssh -i $SSH_KEY -p $REMOTE_PORT" \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  "$REMOTE_USER@$REMOTE_IP:$REMOTE_DIR/src/" \
  "$PROJECT_DIR/src/"

# Pull configs/ — no backup needed (text files, git-tracked)
echo ""
echo "--- Pulling configs/ ---"
rsync -avz --progress \
  -e "ssh -i $SSH_KEY -p $REMOTE_PORT" \
  "$REMOTE_USER@$REMOTE_IP:$REMOTE_DIR/configs/" \
  "$PROJECT_DIR/configs/"

# Pull notebooks/ — no backup needed (git-tracked)
echo ""
echo "--- Pulling notebooks/ ---"
rsync -avz --progress \
  -e "ssh -i $SSH_KEY -p $REMOTE_PORT" \
  --exclude='.ipynb_checkpoints/' \
  "$REMOTE_USER@$REMOTE_IP:$REMOTE_DIR/notebooks/" \
  "$PROJECT_DIR/notebooks/"

# Clean up empty backup dirs (rsync creates them even if nothing was backed up)
find "$BACKUP_DIR" -type d -empty -delete 2>/dev/null
# Remove the top-level backup dir too if nothing was actually backed up
rmdir "$BACKUP_DIR" 2>/dev/null && echo "" && echo "No local files were overwritten — backup dir removed." || true

echo ""
echo "============================================"
echo "PULL COMPLETE"
echo "============================================"
echo ""
echo "Pulled:"
echo "  models/:   $(find "$PROJECT_DIR/models" -name "*.pth" 2>/dev/null | wc -l) .pth files"
echo "  oof/:      $(find "$PROJECT_DIR/oof" -name "*.csv" 2>/dev/null | wc -l) .csv files"
echo "  src/:      ✓"
echo "  configs/:  ✓"
echo "  notebooks/:✓"
echo ""
# Only print backup info if the dir still exists (meaning something was backed up)
if [ -d "$BACKUP_DIR" ]; then
  echo "Backed up overwritten files to:"
  echo "  $BACKUP_DIR"
  echo ""
fi
echo "Safe to destroy the remote instance now."
