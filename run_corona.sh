#!/bin/bash
# ===========================================
# run_corona.sh — Corona(日冕) D2STGNN变体
# Walpurgis-WTFGG Project
# 算法: EMA decomp, LSTM+RoPE, quantile loss,
#        attention-weighted graph conv, top-k mask,
#        RAdam+CosineAnnealingWR, gated residual agg
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

        echo "Installing dependencies..."
        pip install numpy scipy pyyaml matplotlib seaborn
        pip install setproctitle rich

        echo "Verifying PyTorch..."
        python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
    fi
    echo "✓ Environment ready"
}

# ===========================================
# GPU Detection
# ===========================================
detect_gpu() {
    print_step "Detecting GPU"
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        echo "GPU detected:"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
        if [ "$DEVICE" = "cpu" ]; then
            DEVICE="cuda:${CUDA_DEVICE}"
            echo "Auto-switching to: $DEVICE"
        fi
    else
        echo "No GPU detected, using CPU"
        DEVICE="cpu"
    fi
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
}

# ===========================================
# Generate SYNTH Data
# ===========================================
generate_synth() {
    if [ "$DATASET" = "SYNTH" ] && [ ! -f "datasets/SYNTH/train.npz" ]; then
        print_step "Generating SYNTH dataset"
        python -c "
import sys; sys.path.insert(0, 'src')
from walpurgis_corona.generate_synth_data import generate_synth_traffic
generate_synth_traffic()
"
        echo "✓ SYNTH data generated"
    fi
}

# ===========================================
# Training
# ===========================================
run_training() {
    print_step "Training Corona variant"
    echo "  Dataset: $DATASET"
    echo "  Device:  $DEVICE"
    echo "  Epochs:  $EPOCHS"
    echo ""

    python train_corona.py \
        --dataset "$DATASET" \
        --device "$DEVICE" \
        --epochs "$EPOCHS" \
        --debug
}

# ===========================================
# Main
# ===========================================
main() {
    echo "=========================================="
    echo " Walpurgis Corona (日冕) Training Pipeline"
    echo " Dataset: $DATASET | Device: $DEVICE"
    echo "=========================================="

    setup_environment
    detect_gpu
    generate_synth
    run_training

    echo ""
    echo "✓ Corona pipeline complete"
}

# Help
case "${1:-}" in
    -h|--help|help)
        echo "Usage: ./run_corona.sh [DATASET] [DEVICE] [EPOCHS]"
        echo "  DATASET: SYNTH (default), METR-LA, PEMS-BAY, PEMS04, PEMS08"
        echo "  DEVICE:  cpu (default, auto-detects GPU)"
        echo "  EPOCHS:  100 (default)"
        echo ""
        echo "Environment variables:"
        echo "  CONDA_ENV=walpurgis   Conda environment name"
        echo "  CUDA_DEVICE=0         GPU device ID"
        exit 0
        ;;
esac

main "$@"
