#!/bin/bash
set -eo pipefail
# prepare_metrla.sh — 薄包装层, 逻辑全在 prepare_metrla.py
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# metr-la.h5 是 PyTables 编码的 HDF5, 只有 tables 包能读
python3 -c "import tables" 2>/dev/null || {
    echo "[prepare] Installing tables (required to read PyTables HDF5)..."
    pip install --quiet tables
}

exec python3 "$REPO_DIR/prepare_metrla.py" "$@"
