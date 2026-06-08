#!/usr/bin/env bash
# ============================================================
#  run_meridian.sh — Run walpurgis_meridian D2STGNN variant
#  Usage: bash run_meridian.sh [SYNTH|METR-LA|PEMS-BAY|PEMS04|PEMS08]
# ============================================================
set -euo pipefail

DATASET="${1:-SYNTH}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VARIANT_DIR="${SCRIPT_DIR}/src/walpurgis_meridian"

echo "═══════════════════════════════════════════"
echo "  walpurgis_meridian — D2STGNN Meridian"
echo "  Dataset: ${DATASET}"
echo "  CWD: ${VARIANT_DIR}"
echo "═══════════════════════════════════════════"

cd "${VARIANT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

# Generate SYNTH data if needed
if [ "${DATASET}" = "SYNTH" ] && [ ! -f "datasets/SYNTH/train.npz" ]; then
    echo "[meridian] Generating SYNTH dataset..."
    python3 generate_synth_data.py
fi

# Enable debug if requested
export MERIDIAN_DEBUG="${MERIDIAN_DEBUG:-0}"

# Run
python3 -u main.py --dataset "${DATASET}" 2>&1 | tee "/tmp/meridian_${DATASET}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "═══ meridian run complete ═══"
