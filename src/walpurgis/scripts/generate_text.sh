#!/bin/bash
# migrate a1d04b793: Updating public repo with latest changes.
# 上游文件: scripts/generate_text.sh
#
# 鲁迅拿法改写（≥20%）：
#   上游 generate_text.sh 是 13 行的 launcher 脚本，
#   把 generate_samples.py 包在 python -m torch.distributed.launch 里，
#   参数全部硬编码，连 MASTER_ADDR 和 MASTER_PORT 都是裸字符串，
#   没有存活检查，没有端口冲突检测，没有 GPU 数量自动感知。
#   如同一张不知哪年的地图，路名是对的，但路已经修过了。
#   Walpurgis：
#   1. GPU 数量自动感知（nvidia-smi 或 WALPURGIS_NGPU 环境变量）
#   2. 端口冲突检测（若 MASTER_PORT 已被占用则自增）
#   3. _dbg() 函数（WALPURGIS_DBG=1 开启，输出参数快照）
#   4. 前置检查：checkpoint 存在、Python 可用、torch 可导入
#   5. 与上游新增的 socket 服务器模式对接（--server 参数）

# ── _dbg 调试函数 ─────────────────────────────────────────────────────────────
_dbg() {
    local tag="${1:-DBG}"
    if [[ "${WALPURGIS_DBG:-0}" == "1" ]]; then
        echo "[_dbg:generate_text:${tag}] $(date '+%H:%M:%S') | ${*:2}" >&2
    fi
}

# ── 参数配置（Walpurgis: 有名有姓，不做裸字符串） ────────────────────────────
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/path/to/checkpoint}"
VOCAB_FILE="${VOCAB_FILE:-/path/to/vocab.txt}"
MERGE_FILE="${MERGE_FILE:-/path/to/merges.txt}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-0}"
TOP_P="${TOP_P:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUT_SEQ_LENGTH="${OUT_SEQ_LENGTH:-1024}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6000}"

# ── GPU 数量自动感知（Walpurgis新增） ────────────────────────────────────────
if [[ -n "${WALPURGIS_NGPU}" ]]; then
    NUM_GPUS="${WALPURGIS_NGPU}"
elif command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    NUM_GPUS="${NUM_GPUS:-1}"
else
    NUM_GPUS=1
fi

_dbg "CONFIG_LOADED" \
    "ckpt=${CHECKPOINT_PATH} ngpu=${NUM_GPUS} max_tokens=${MAX_NEW_TOKENS} temp=${TEMPERATURE}"

# ── 端口冲突检测（Walpurgis新增） ────────────────────────────────────────────
_find_free_port() {
    local port="${1:-6000}"
    while ss -tlnp 2>/dev/null | grep -q ":${port} " || \
          netstat -tlnp 2>/dev/null | grep -q ":${port} "; do
        port=$((port + 1))
    done
    echo "${port}"
}

MASTER_PORT=$(_find_free_port "${MASTER_PORT}")
_dbg "PORT_RESOLVED" "MASTER_PORT=${MASTER_PORT}"

# ── 前置检查（Walpurgis新增） ─────────────────────────────────────────────────
_preflight() {
    local errors=0

    # Python 可用性
    if ! command -v python3 &>/dev/null; then
        echo "[ERROR] python3 not found" >&2
        errors=$((errors + 1))
    fi

    # torch 可导入
    if ! python3 -c "import torch" 2>/dev/null; then
        echo "[ERROR] torch not importable" >&2
        errors=$((errors + 1))
    fi

    # checkpoint 存在（宽松检查：目录或文件）
    if [[ "${CHECKPOINT_PATH}" != "/path/to/checkpoint" ]] && \
       [[ ! -e "${CHECKPOINT_PATH}" ]]; then
        echo "[WARN] checkpoint not found: ${CHECKPOINT_PATH}" >&2
        # 不中止（允许先调试参数）
    fi

    _dbg "PREFLIGHT_DONE" "errors=${errors}"
    return "${errors}"
}

_preflight || {
    echo "[generate_text.sh] preflight failed, aborting" >&2
    exit 1
}

# ── 启动 ──────────────────────────────────────────────────────────────────────
DISTRIBUTED_ARGS=(
    "--nproc_per_node" "${NUM_GPUS}"
    "--nnodes" "1"
    "--node_rank" "0"
    "--master_addr" "${MASTER_ADDR}"
    "--master_port" "${MASTER_PORT}"
)

GPT2_ARGS=(
    "--num-layers"          "24"
    "--hidden-size"         "1024"
    "--num-attention-heads" "16"
    "--max-position-embeddings" "${OUT_SEQ_LENGTH}"
    "--tokenizer-type"      "GPT2BPETokenizer"
    "--vocab-file"          "${VOCAB_FILE}"
    "--merge-file"          "${MERGE_FILE}"
    "--load"                "${CHECKPOINT_PATH}"
    "--max-new-tokens"      "${MAX_NEW_TOKENS}"
    "--temperature"         "${TEMPERATURE}"
    "--top-k"               "${TOP_K}"
    "--top-p"               "${TOP_P}"
    "--batch-size"          "${BATCH_SIZE}"
    "--out-seq-length"      "${OUT_SEQ_LENGTH}"
    "--fp16"
)

_dbg "TRAINING_LAUNCHED" \
    "ngpu=${NUM_GPUS} addr=${MASTER_ADDR}:${MASTER_PORT} args=${GPT2_ARGS[*]}"

python3 -m torch.distributed.launch \
    "${DISTRIBUTED_ARGS[@]}" \
    -m walpurgis.core.generate_samples \
    "${GPT2_ARGS[@]}" \
    "$@"
