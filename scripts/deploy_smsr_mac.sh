#!/usr/bin/env bash
# deploy_smsr_mac.sh — Deploy SMSR validation to a remote inference machine
#
# Prerequisites:
#   - SMSR_REMOTE_HOST and SMSR_REMOTE_USER set (see below) with a working
#     SSH connection to the remote machine (e.g. an SSH config alias)
#   - Checkpoint exists locally at checkpoints/deberta-wildguardmix/
#   - Remote machine has (or this script will create) a Python venv with
#     torch, transformers, scipy, matplotlib
#
# Usage:
#   export SMSR_REMOTE_HOST=your-ssh-alias-or-hostname
#   export SMSR_REMOTE_USER=your-remote-username
#   ./scripts/deploy_smsr_mac.sh
#   ./scripts/deploy_smsr_mac.sh --skip-checkpoint  # if already transferred

set -euo pipefail

: "${SMSR_REMOTE_HOST:?Set SMSR_REMOTE_HOST to the SSH alias/hostname of the remote inference machine}"
: "${SMSR_REMOTE_USER:?Set SMSR_REMOTE_USER to the remote username}"

REMOTE="${SMSR_REMOTE_HOST}"
REMOTE_USER="${SMSR_REMOTE_USER}"
REMOTE_BASE="${SMSR_REMOTE_BASE:-/Users/${REMOTE_USER}/canaries_smsr}"
REMOTE_CKPT="${REMOTE_BASE}/checkpoints/deberta-wildguardmix"
REMOTE_SCRIPTS="${REMOTE_BASE}/scripts"
REMOTE_RESULTS="${REMOTE_BASE}/results"

LOCAL_CKPT="checkpoints/deberta-wildguardmix"
LOCAL_SCRIPT="scripts/exp_smsr_validation_mac.py"

SKIP_CHECKPOINT=false

for arg in "$@"; do
    case $arg in
        --skip-checkpoint) SKIP_CHECKPOINT=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "=== SMSR Remote Deployment ==="
echo "Remote: ${REMOTE} (${REMOTE_USER})"
echo "Remote base: ${REMOTE_BASE}"
echo ""

# --- 1. Create remote directories ---
echo "[1/4] Creating remote directories..."
ssh "${REMOTE}" "mkdir -p ${REMOTE_CKPT} ${REMOTE_SCRIPTS} ${REMOTE_RESULTS}"

# --- 2. Transfer checkpoint ---
if [ "$SKIP_CHECKPOINT" = false ]; then
    echo "[2/4] Transferring checkpoint (~700MB, may take a few minutes)..."
    echo "  Source: ${LOCAL_CKPT}/"
    echo "  Dest:   ${REMOTE}:${REMOTE_CKPT}/"

    # Use rsync for resumability; fall back to scp if rsync unavailable
    if command -v rsync &>/dev/null; then
        rsync -avz --progress "${LOCAL_CKPT}/" "${REMOTE}:${REMOTE_CKPT}/"
    else
        scp -r "${LOCAL_CKPT}/"* "${REMOTE}:${REMOTE_CKPT}/"
    fi
    echo "  Checkpoint transfer complete."
else
    echo "[2/4] Skipping checkpoint transfer (--skip-checkpoint)"
fi

# --- 3. Transfer validation script ---
echo "[3/4] Transferring validation script..."
scp "${LOCAL_SCRIPT}" "${REMOTE}:${REMOTE_SCRIPTS}/exp_smsr_validation_mac.py"

# --- 4. Run on the remote machine via nohup ---
echo "[4/4] Launching validation on ${REMOTE} (nohup)..."

# REMOTE_BASE is passed as an env var to the remote command (not expanded
# inside the heredoc, which stays single-quoted so that $! and ${PWD}
# below are evaluated on the REMOTE side, not the local shell).
ssh "${REMOTE}" "REMOTE_BASE='${REMOTE_BASE}' bash -s" <<'REMOTE_SCRIPT'
cd "${REMOTE_BASE}"

# Set up venv if it doesn't exist
if [ ! -d .venv ]; then
    echo "Creating venv..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install torch transformers scipy matplotlib numpy
fi

# Run with nohup
nohup .venv/bin/python scripts/exp_smsr_validation_mac.py \
    --checkpoint-dir checkpoints/deberta-wildguardmix \
    --output-dir results \
    > results/smsr_validation_mac.log 2>&1 &

PID=$!
echo "Launched with PID=${PID}"
echo "Log: ${PWD}/results/smsr_validation_mac.log"
REMOTE_SCRIPT

echo ""
echo "=== Deployment complete ==="
echo ""
echo "Monitor progress:"
echo "  ssh ${REMOTE} 'tail -f ${REMOTE_BASE}/results/smsr_validation_mac.log'"
echo ""
echo "Fetch results when done:"
echo "  scp ${REMOTE}:${REMOTE_BASE}/results/smsr_validation_mac.json results/"
echo "  scp ${REMOTE}:${REMOTE_BASE}/results/smsr_validation_mac.png results/"
