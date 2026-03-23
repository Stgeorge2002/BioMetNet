#!/bin/bash
# =============================================================================
# Full BioMetNet Multi-Organism Pipeline
# Run this after setup_pod.sh to execute the entire pipeline.
# Usage: bash cloud/run_pipeline.sh [--max-models N]
# =============================================================================
set -e

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=/root/.venv
export UV_LINK_MODE=copy
cd /workspace/BioMetNet

# Ensure package is up to date in the venv
uv sync

MAX_MODELS="${1:---max-models}"
MAX_MODELS_VAL="${2:-}"

echo "========================================"
echo "  BioMetNet Multi-Organism Pipeline"
echo "========================================"
echo ""

# Step 1: Download BiGG models and prepare dataset
echo "=== Step 1/3: Downloading BiGG models & preparing dataset ==="
echo "This downloads ~108 COBRA models from BiGG and generates training data."
echo ""
# Clear old processed data so it regenerates with new config
rm -rf data/processed/multi_organism
if [ -n "$MAX_MODELS_VAL" ]; then
    uv run python scripts/prepare_multi_organism.py --max-models "$MAX_MODELS_VAL"
else
    uv run python scripts/prepare_multi_organism.py
fi

echo ""
echo "=== Step 2/3: Training multi-organism classifier ==="
echo ""
# Clear old checkpoints for clean run
rm -rf checkpoints
uv run python scripts/train.py --dataset multi_organism

echo ""
echo "=== Step 3/3: Evaluating on held-out organisms ==="
echo ""
uv run python scripts/evaluate.py --dataset multi_organism --sweep

echo ""
echo "========================================"
echo "  Pipeline Complete!"
echo "========================================"
echo ""
echo "Results saved to results/eval_multi_organism.json"
echo "Checkpoint saved to checkpoints/best.pt"
echo ""
echo "To copy results back to your local machine, run this FROM YOUR WSL TERMINAL:"
echo "  bash cloud/sync_results.sh"
