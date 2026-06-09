#!/bin/bash
set -eo pipefail
# prepare_metrla.sh — 薄包装层, 逻辑全在 prepare_metrla.py
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# 确保 h5py 可用 (METR-LA 发行格式是 HDF5, 无法绕过)
python3 -c "import h5py" 2>/dev/null || {
    echo "[prepare] Installing h5py (required for HDF5 data format)..."
    pip install --quiet h5py
}

exec python3 "$REPO_DIR/prepare_metrla.py" "$@"
