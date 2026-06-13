#!/bin/bash
# migrate: megatron 3573423f4 — added presplit-sentences to scripts
# 鲁迅拿法改写：原脚本是一纸告示，参数罗列，无人解释为何如此，
# 后人照抄，照抄再抄，presplit-sentences 一行加上去，亦无注释，
# 如同在铁屋子里开了一扇窗，却不告诉人窗外是何物。
# Walpurgis 在此补全：每个参数都该有名有姓，每个断点都该有迹可查。

# ── _dbg: 调试工具函数 ─────────────────────────────────────────────────────────
_dbg() {
    # 断点：输出当前参数快照供调试，生产环境设 WALPURGIS_DBG=0 关闭
    local tag="${1:-DBG}"
    if [[ "${WALPURGIS_DBG:-1}" == "1" ]]; then
        echo "[_dbg:${tag}] $(date '+%H:%M:%S') | PRESPLIT=${PRESPLIT_SENTENCES} | DATA_PATH=${DATA_PATH} | VOCAB=${VOCAB_SIZE}" >&2
    fi
}

# ── 参数配置区（鲁迅拿法：有名有姓，不做无名隐士）───────────────────────────
NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_ATTN_HEADS=16
BATCH_SIZE=32
SEQ_LENGTH=512
MAX_PREDS_PER_SEQ=80
TRAIN_ITERS=1000000
SAVE_ITERS=10000

TOKENIZER_TYPE="BertWordPieceLowerCase"
TOKENIZER_MODEL="bert-large-uncased"
VOCAB_SIZE=30522

TRAIN_DATA="wikipedia"
# presplit-sentences: 数据预切句，upstream 3573423f4 新增。
# 含义：输入 JSON 中每条 text 已按句切分，不再由训练脚本二次切割。
# 无此 flag 则训练时动态切句，速度慢且结果不可复现。
PRESPLIT_SENTENCES="--presplit-sentences"

DATA_PATH="${WALPURGIS_DATA_PATH:-/data/bert}"
VOCAB_FILE="${DATA_PATH}/vocab.txt"
CHECKPOINT_PATH="${WALPURGIS_CKPT_PATH:-checkpoints/bert_large}"

_dbg "CONFIG_LOADED"

# ── 运行前自检（上游脚本无此项，鲁迅所谓"先生只顾讲，不问听没听懂"）─────────
if [[ ! -f "${VOCAB_FILE}" ]]; then
    echo "[warn] 词表文件不存在: ${VOCAB_FILE}，将依赖 tokenizer 自动下载" >&2
fi

_dbg "PREFLIGHT_DONE"

# ── 主训练入口 ──────────────────────────────────────────────────────────────────
python pretrain_bert.py \
    --num-layers        ${NUM_LAYERS}                     \
    --hidden-size       ${HIDDEN_SIZE}                    \
    --num-attention-heads ${NUM_ATTN_HEADS}               \
    --batch-size        ${BATCH_SIZE}                     \
    --seq-length        ${SEQ_LENGTH}                     \
    --max-preds-per-seq ${MAX_PREDS_PER_SEQ}              \
    --train-iters       ${TRAIN_ITERS}                    \
    --save-iters        ${SAVE_ITERS}                     \
    --save              ${CHECKPOINT_PATH}                \
    --tokenizer-type    ${TOKENIZER_TYPE}                 \
    --tokenizer-model-type ${TOKENIZER_MODEL}             \
    --vocab-size        ${VOCAB_SIZE}                     \
    --train-data        ${TRAIN_DATA}                     \
    ${PRESPLIT_SENTENCES}                                 \
    --loose-json                                          \
    --text-key          text                              \
    --split             1000,1,1                          \
    --lr                0.0001                            \
    --lr-decay-style    linear                            \
    --lr-decay-iters    990000                            \
    --warmup            0.01                              \
    --weight-decay      1e-2                              \
    --clip-grad         1.0                               \
    --fp16

_dbg "TRAINING_LAUNCHED"
