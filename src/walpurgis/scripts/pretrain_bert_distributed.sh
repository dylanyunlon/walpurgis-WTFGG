#!/bin/bash
# migrate: megatron 3573423f4 — added presplit-sentences to scripts
# 鲁迅拿法改写：分布式训练脚本，原版像个工头的口令，
# MASTER_ADDR 写死，端口写死，workers 写死——
# 如同叫人"照着做"，却不告诉人"为什么这样做"。
# Walpurgis 将分布式参数显式命名，_dbg() 在关键节点留痕。

# ── _dbg: 调试工具函数 ─────────────────────────────────────────────────────────
_dbg() {
    local tag="${1:-DBG}"
    if [[ "${WALPURGIS_DBG:-1}" == "1" ]]; then
        echo "[_dbg:${tag}] $(date '+%H:%M:%S') | RANK=${NODE_RANK} | WORLD=${WORLD_SIZE} | PRESPLIT=${PRESPLIT_SENTENCES}" >&2
    fi
}

# ── 分布式环境参数（鲁迅拿法：显式胜于隐式）────────────────────────────────────
# MASTER_ADDR / MASTER_PORT: 主节点通信地址，多机训练须从外部注入
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6000}"
# NNODES: 物理节点数；GPUS_PER_NODE: 每节点 GPU 数
NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

DISTRIBUTED_ARGS=(
    --nproc_per_node ${GPUS_PER_NODE}
    --nnodes         ${NNODES}
    --node_rank      ${NODE_RANK}
    --master_addr    ${MASTER_ADDR}
    --master_port    ${MASTER_PORT}
)

_dbg "DIST_ENV_READY"

# ── 模型与训练参数 ──────────────────────────────────────────────────────────────
NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_ATTN_HEADS=16
BATCH_SIZE=8           # 分布式下 per-GPU batch size
SEQ_LENGTH=512
MAX_PREDS_PER_SEQ=80
TRAIN_ITERS=1000000
SAVE_ITERS=10000

TOKENIZER_TYPE="BertWordPieceLowerCase"
TOKENIZER_MODEL="bert-large-uncased"
VOCAB_SIZE=30522

TRAIN_DATA="wikipedia"
# presplit-sentences: megatron 3573423f4 新增。
# 分布式场景下此参数尤为重要：多 worker 并行读取时，
# 若依赖运行时切句则各 worker 切分结果可能因随机种子不同而发散；
# 预切后每条样本边界固定，多卡采样确定性得到保证。
PRESPLIT_SENTENCES="--presplit-sentences"

DATA_PATH="${WALPURGIS_DATA_PATH:-/data/bert}"
CHECKPOINT_PATH="${WALPURGIS_CKPT_PATH:-checkpoints/bert_large_dist}"

_dbg "CONFIG_LOADED"

# ── 主训练入口（torch.distributed.launch）──────────────────────────────────────
python -m torch.distributed.launch "${DISTRIBUTED_ARGS[@]}" \
    pretrain_bert.py                                          \
    --num-layers          ${NUM_LAYERS}                       \
    --hidden-size         ${HIDDEN_SIZE}                      \
    --num-attention-heads ${NUM_ATTN_HEADS}                   \
    --batch-size          ${BATCH_SIZE}                       \
    --seq-length          ${SEQ_LENGTH}                       \
    --max-preds-per-seq   ${MAX_PREDS_PER_SEQ}                \
    --train-iters         ${TRAIN_ITERS}                      \
    --save-iters          ${SAVE_ITERS}                       \
    --save                ${CHECKPOINT_PATH}                  \
    --tokenizer-type      ${TOKENIZER_TYPE}                   \
    --tokenizer-model-type ${TOKENIZER_MODEL}                 \
    --vocab-size          ${VOCAB_SIZE}                       \
    --train-data          ${TRAIN_DATA}                       \
    ${PRESPLIT_SENTENCES}                                     \
    --loose-json                                              \
    --text-key            text                                \
    --split               1000,1,1                            \
    --lr                  0.0001                              \
    --lr-decay-style      linear                              \
    --lr-decay-iters      990000                              \
    --warmup              0.01                                \
    --weight-decay        1e-2                                \
    --clip-grad           1.0                                 \
    --fp16

_dbg "TRAINING_LAUNCHED"
