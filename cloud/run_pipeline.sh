#!/bin/bash
# =============================================================================
# Full BioMetNet E. coli Strain Pipeline
# Run this after setup_pod.sh to execute the entire pipeline.
# Usage: bash cloud/run_pipeline.sh [--max-models N]
# =============================================================================
set -e

# Auto-stop the pod when the script exits (success or error) so it doesn't
# keep running and costing money.
_shutdown() {
    if [ -n "$RUNPOD_POD_ID" ]; then
        echo "Stopping pod $RUNPOD_POD_ID..."
        runpodctl stop pod "$RUNPOD_POD_ID"
    fi
}
trap _shutdown EXIT

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=/root/.venv
export UV_LINK_MODE=copy
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /workspace/BioMetNet

# Ensure package is up to date in the venv
uv sync

MAX_MODELS="${1:---max-models}"
MAX_MODELS_VAL="${2:-}"

echo "========================================"
echo "  BioMetNet E. coli Strain Pipeline"
echo "========================================"
echo ""

# Step 1: Download BiGG models and prepare dataset
echo "=== Step 1/3: Downloading E. coli strain models & preparing dataset ==="
echo "This downloads E. coli COBRA models from BiGG and generates training data."
echo ""
# Clear old processed data so it regenerates with new config
rm -rf data/processed/ecoli_strains
if [ -n "$MAX_MODELS_VAL" ]; then
    uv run python scripts/prepare_data.py --max-models "$MAX_MODELS_VAL"
else
    uv run python scripts/prepare_data.py
fi

echo ""
echo "=== Step 2/3: Training E. coli strain classifier ==="
echo ""
# Clear old checkpoints for clean run
rm -rf checkpoints
uv run python scripts/train.py --dataset ecoli_strains

echo ""
echo "=== Step 3/3: Evaluating on held-out strains ==="
echo ""
uv run python scripts/evaluate.py --dataset ecoli_strains --sweep

echo ""
echo "========================================"
echo "  Pipeline Complete!"
echo "========================================"
echo ""
echo "Results saved to results/eval_ecoli_strains.json"
echo "Checkpoint saved to checkpoints/best.pt"
echo ""
echo "To copy results back to your local machine, run this FROM YOUR WSL TERMINAL:"
echo "  bash cloud/sync_results.sh"
