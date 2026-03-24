#!/bin/bash
# =============================================================================
# RunPod GPU Pod Setup Script
# Run this ONCE after SSH-ing into your new pod.
# Usage: bash setup_pod.sh
# =============================================================================
set -e

echo "=== BioMetNet RunPod Setup ==="

# 1. Install uv (fast Python package manager)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
else
    echo "uv already installed"
fi

# 2. Navigate to project
cd /workspace/BioMetNet

# 3. Create venv and install dependencies
echo "Installing Python dependencies..."
uv sync

# 3b. Install CarveMe extras (optional, for NCBI+CarveMe pipeline)
if [ "${INSTALL_CARVEME:-0}" = "1" ]; then
    echo "Installing CarveMe + LP solver..."
    uv sync --extra carveme
    # Verify CarveMe is available
    if uv run carve --help > /dev/null 2>&1; then
        echo "CarveMe installed successfully"
    else
        echo "WARNING: CarveMe installation may have issues"
    fi

    # Install diamond (required by CarveMe for protein alignment)
    echo "Installing diamond aligner..."
    if ! command -v diamond &> /dev/null; then
        conda install -y -c bioconda diamond 2>/dev/null || \
        wget -q https://github.com/bbuchfink/diamond/releases/download/v2.1.11/diamond-linux64.tar.gz \
            -O /tmp/diamond.tar.gz \
            && tar -xzf /tmp/diamond.tar.gz -C /usr/local/bin diamond \
            && rm /tmp/diamond.tar.gz
    fi
    if command -v diamond &> /dev/null; then
        echo "diamond $(diamond version 2>&1 | head -1) installed"
    else
        echo "WARNING: diamond not installed — CarveMe will fail"
    fi
fi

# 4. Verify GPU is available
echo ""
echo "=== GPU Check ==="
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# 5. Verify imports
echo ""
echo "=== Import Check ==="
uv run python -c "
from biometnet.data.strain_data import download_all_bigg_models, prepare_strain_dataset
from biometnet.model.strain_classifier import EcoliStrainClassifier
from biometnet.data.dataset import StrainDataset, strain_collate_fn
print('All imports OK')
"

# 6. Run tests (if pytest available)
echo ""
echo "=== Running Tests ==="
if uv run python -m pytest tests/ -v 2>/dev/null; then
    echo "Tests passed"
else
    echo "Skipped (pytest not installed — use 'uv sync --extra dev' to enable)"
fi

echo ""
echo "=== Setup Complete ==="
echo "Now run the pipeline with:"
echo "  bash cloud/run_pipeline.sh"
