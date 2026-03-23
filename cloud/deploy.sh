#!/bin/bash
# =============================================================================
# Deploy BioMetNet to RunPod via rsync
# Run this FROM YOUR LOCAL WSL terminal.
#
# Usage: bash cloud/deploy.sh <pod-ip> [ssh-port]
# Example: bash cloud/deploy.sh 194.68.245.10 22186
#
# The pod IP and SSH port are shown in your RunPod dashboard under
# "Connect" -> "SSH over exposed TCP" for your pod.
# =============================================================================
set -e

POD_IP="${1:?Usage: bash cloud/deploy.sh <pod-ip> [ssh-port]}"
SSH_PORT="${2:-22}"

echo "Deploying BioMetNet to ${POD_IP}:${SSH_PORT}..."

rsync -avz --progress \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache/' \
    --exclude 'data/' \
    --exclude 'checkpoints/' \
    --exclude 'results/' \
    --exclude '*.pt' \
    --exclude '*.pth' \
    --exclude '.git/' \
    -e "ssh -p ${SSH_PORT} -o StrictHostKeyChecking=no" \
    ~/BioMetNet/ \
    root@${POD_IP}:/workspace/BioMetNet/

echo ""
echo "Deploy complete! Now SSH in and run setup:"
echo "  ssh -p ${SSH_PORT} root@${POD_IP}"
echo "  cd /workspace/BioMetNet && bash cloud/setup_pod.sh"
