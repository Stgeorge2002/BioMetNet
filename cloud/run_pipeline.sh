#!/bin/bash
# =============================================================================
# Full BioMetNet E. coli Strain Pipeline
# Run this after setup_pod.sh to execute the entire pipeline.
# Usage:
#   bash cloud/run_pipeline.sh                          # BiGG-only (default)
#   bash cloud/run_pipeline.sh --carveme 200            # BiGG + 200 CarveMe models
#   bash cloud/run_pipeline.sh --carveme-only 200       # CarveMe build only (no training)
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

export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=/root/.venv
export UV_LINK_MODE=copy
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /workspace/BioMetNet

# Ensure package is up to date in the venv
uv sync

# Parse arguments
CARVEME_GENOMES=""
CARVEME_ONLY=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --carveme)
            CARVEME_GENOMES="$2"
            shift 2
            ;;
        --carveme-only)
            CARVEME_GENOMES="$2"
            CARVEME_ONLY=1
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "========================================"
echo "  BioMetNet E. coli Strain Pipeline"
echo "========================================"
echo ""

# Optional: Build CarveMe models from NCBI genomes
CARVEME_DIR="data/raw/carveme_models"
if [ -n "$CARVEME_GENOMES" ]; then
    echo "=== Step 0: Building CarveMe models from NCBI genomes ==="
    echo "  Target: $CARVEME_GENOMES genomes"
    echo ""
    uv sync --extra carveme

    # Ensure diamond is available (required by CarveMe)
    if ! command -v diamond &> /dev/null; then
        echo "Installing diamond aligner..."
        wget -q https://github.com/bbuchfink/diamond/releases/download/v2.1.11/diamond-linux64.tar.gz \
            -O /tmp/diamond.tar.gz \
            && tar -xzf /tmp/diamond.tar.gz -C /usr/local/bin diamond \
            && rm /tmp/diamond.tar.gz
        echo "diamond installed"
    fi
    uv run python scripts/build_carveme_models.py --max-genomes "$CARVEME_GENOMES"
    echo ""

    if [ "$CARVEME_ONLY" = "1" ]; then
        echo "CarveMe build complete. Skipping training (--carveme-only mode)."
        exit 0
    fi
fi

# Step 1: Download BiGG models and prepare dataset
echo "=== Step 1/3: Downloading E. coli strain models & preparing dataset ==="
echo "This downloads E. coli COBRA models from BiGG and generates training data."
echo ""
# Clear old processed data so it regenerates with new config
rm -rf data/processed/ecoli_strains
PREPARE_ARGS=""
if [ -d "$CARVEME_DIR" ] && [ "$(ls -A $CARVEME_DIR/*.xml 2>/dev/null)" ]; then
    PREPARE_ARGS="--carveme-dir $CARVEME_DIR"
    echo "  Including CarveMe models from $CARVEME_DIR"
fi
uv run python scripts/prepare_data.py $PREPARE_ARGS

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
