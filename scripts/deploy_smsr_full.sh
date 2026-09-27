#!/usr/bin/env bash
# deploy_smsr_full.sh — Deploy full DeBERTa training + SMSR validation to a
# remote inference machine
#
# Usage:
#   export SMSR_REMOTE_HOST=your-ssh-alias-or-hostname
#   export SMSR_REMOTE_USER=your-remote-username
#   ./scripts/deploy_smsr_full.sh

set -euo pipefail

: "${SMSR_REMOTE_HOST:?Set SMSR_REMOTE_HOST to the SSH alias/hostname of the remote inference machine}"
: "${SMSR_REMOTE_USER:?Set SMSR_REMOTE_USER to the remote username}"

REMOTE="${SMSR_REMOTE_HOST}"
REMOTE_USER="${SMSR_REMOTE_USER}"
REMOTE_BASE="${SMSR_REMOTE_BASE:-/Users/${REMOTE_USER}/canaries_smsr}"
LOCAL_SCRIPT="scripts/train_and_validate_smsr_mac.py"

echo "=== SMSR Full Pipeline — Remote Deployment ==="
echo "Remote: ${REMOTE} (${REMOTE_USER})"
echo "Remote base: ${REMOTE_BASE}"

# 1. Create remote dirs
echo "[1/4] Creating remote directories..."
ssh "${REMOTE}" "mkdir -p ${REMOTE_BASE}/scripts ${REMOTE_BASE}/results ${REMOTE_BASE}/checkpoints"

# 2. Transfer script
echo "[2/4] Transferring pipeline script..."
scp "${LOCAL_SCRIPT}" "${REMOTE}:${REMOTE_BASE}/scripts/"

# 3. Install deps
echo "[3/4] Installing dependencies..."
ssh "${REMOTE}" "REMOTE_BASE='${REMOTE_BASE}' bash -s" <<'REMOTE_DEPS'
cd "${REMOTE_BASE}"
if [ ! -d .venv ]; then
    echo "Creating venv..."
    python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install torch transformers datasets accelerate scipy matplotlib numpy sentencepiece protobuf -q
echo "Deps installed."
REMOTE_DEPS

# 4. Launch
echo "[4/4] Launching pipeline (nohup)..."
ssh "${REMOTE}" "REMOTE_BASE='${REMOTE_BASE}' bash -s" <<'REMOTE_RUN'
cd "${REMOTE_BASE}"
nohup .venv/bin/python scripts/train_and_validate_smsr_mac.py > results/pipeline.log 2>&1 &
PID=$!
echo "Launched PID=${PID}"
REMOTE_RUN

echo ""
echo "=== Deployment complete ==="
echo "Monitor:  ssh ${REMOTE} 'tail -f ${REMOTE_BASE}/results/pipeline.log'"
echo "Results:  scp ${REMOTE}:${REMOTE_BASE}/results/smsr_validation_full.json results/"
echo "Figure:   scp ${REMOTE}:${REMOTE_BASE}/results/smsr_validation_full.png results/"
