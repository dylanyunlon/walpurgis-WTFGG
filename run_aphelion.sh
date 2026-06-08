#!/bin/bash
# ===========================================
# run_aphelion.sh — Aphelion(远日点) D2STGNN变体
# Walpurgis-WTFGG Project
# 算法: Hypernetwork gate, VMD decomp, GATv2+edge,
#        SimCLR adjacency, entropy mask, wavelet norm,
#        Retention+cross-scale, FPN output, TERM loss,
#        Sophia optimizer + ExponentialLR
# ===========================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
DATASET="${1:-SYNTH}"
DEVICE="${2:-cpu}"
EPOCHS="${3:-100}"
CONDA_ENV="${CONDA_ENV:-walpurgis}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

print_step() {
    echo ""
    echo "=== $1 ==="
    echo ""
}

# ===========================================
# Environment Setup (conda + gpu)
# ===========================================
setup_environment() {
    print_step "Setting up Conda Environment: $CONDA_ENV"

    if ! command -v conda &>/dev/null; then
        echo "Warning: Conda not found, using system python"
        return 0
    fi

    eval "$(conda shell.bash hook)" 2>/dev/null || true

    if conda env list | grep -q "^${CONDA_ENV} "; then
        echo "Activating existing environment: $CONDA_ENV"
        conda activate "$CONDA_ENV"
    else
        echo "Creating new conda environment: $CONDA_ENV (Python 3.10)"
        conda create -n "$CONDA_ENV" python=3.10 -y
        conda activate "$CONDA_ENV"

        echo "Installing PyTorch with CUDA support..."
        pip install --upgrade pip
        pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
        pip install numpy scipy pyyaml
    fi
}

# ===========================================
# Data Generation
# ===========================================
generate_data() {
    print_step "Checking / Generating SYNTH Data"
    if [ -f "datasets/SYNTH/train.npz" ]; then
        echo "SYNTH data already exists, skipping generation"
    else
        echo "Generating synthetic traffic data..."
        python -c "
import sys; sys.path.insert(0, 'src')
from walpurgis_aphelion.generate_synth_data import generate_synth_traffic
generate_synth_traffic()
"
    fi
}

# ===========================================
# Training
# ===========================================
train_model() {
    print_step "Training Aphelion (远日点) D2STGNN — $DATASET on $DEVICE"
    if [ "$DEVICE" != "cpu" ]; then
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_aphelion.py \
            --dataset "$DATASET" \
            --device "$DEVICE" \
            --epochs "$EPOCHS"
    else
        python train_aphelion.py \
            --dataset "$DATASET" \
            --device "$DEVICE" \
            --epochs "$EPOCHS"
    fi
}

# ===========================================
# Main
# ===========================================
main() {
    echo "=========================================="
    echo "  Walpurgis-WTFGG: Aphelion (远日点)"
    echo "  Dataset: $DATASET"
    echo "  Device:  $DEVICE"
    echo "  Epochs:  $EPOCHS"
    echo "=========================================="

    setup_environment
    generate_data
    train_model

    print_step "Aphelion Training Complete"
    echo "Output: output/D2STGNN_APHELION_${DATASET}.pt"
}

main "$@"
