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

# 4. Verify GPU is available
echo ""
echo "=== GPU Check ==="
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
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

# 6. Run tests
echo ""
echo "=== Running Tests ==="
uv run python -m pytest tests/ -v

echo ""
echo "=== Setup Complete ==="
echo "Now run the pipeline with:"
echo "  bash cloud/run_pipeline.sh"
