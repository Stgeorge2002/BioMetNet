#!/bin/bash
# =============================================================================
# Sync results back from RunPod to local machine
# Run this FROM YOUR LOCAL WSL terminal.
#
# Usage: bash cloud/sync_results.sh <pod-ip> [ssh-port]
# =============================================================================
set -e

POD_IP="${1:?Usage: bash cloud/sync_results.sh <pod-ip> [ssh-port]}"
SSH_PORT="${2:-22}"

echo "Syncing results from ${POD_IP}:${SSH_PORT}..."

# Sync checkpoints
rsync -avz --progress \
    -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no" \
    root@${POD_IP}:/workspace/BioMetNet/checkpoints/ \
    ~/BioMetNet/checkpoints/

# Sync results
rsync -avz --progress \
    -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no" \
    root@${POD_IP}:/workspace/BioMetNet/results/ \
    ~/BioMetNet/results/

# Sync processed data config (not the heavy .pt files)
rsync -avz --progress \
    --include='*.json' \
    --exclude='*.pt' \
    -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no" \
    root@${POD_IP}:/workspace/BioMetNet/data/processed/ecoli_strains/ \
    ~/BioMetNet/data/processed/ecoli_strains/

echo ""
echo "Sync complete! Results are in:"
echo "  ~/BioMetNet/results/eval_ecoli_strains.json"
echo "  ~/BioMetNet/checkpoints/best.pt"
