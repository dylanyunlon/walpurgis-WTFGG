#!/bin/bash
# migrate: megatron 3573423f4 — added presplit-sentences to scripts
# 鲁迅拿法改写：SentencePiece 变体脚本，与 WordPiece 版本差异仅在 tokenizer 路径，
# 原脚本并排列出，如同两篇文章只换了题目，正文照抄——
# 鲁迅曾言，抄书是有益的，但抄而不知其别，则与不抄无异。
# Walpurgis 在此明确 SentencePiece 与 WordPiece 的语义差异，并以 _dbg() 留痕。

# ── _dbg: 调试工具函数 ─────────────────────────────────────────────────────────
_dbg() {
    local tag="${1:-DBG}"
    if [[ "${WALPURGIS_DBG:-1}" == "1" ]]; then
        echo "[_dbg:${tag}] $(date '+%H:%M:%S') | TOKENIZER=sentencepiece | MODEL=${TOKENIZER_MODEL_PATH} | PRESPLIT=${PRESPLIT_SENTENCES}" >&2
    fi
}

# ── Tokenizer 配置（SentencePiece 与 WordPiece 的根本差异在此）─────────────────
# WordPiece (bert-large-uncased): 依赖预定义词表文件 vocab.txt
# SentencePiece: 依赖训练好的 .model 文件，支持任意语言，无需固定词表
# 此脚本对应 SentencePiece 路径，tokenizer.model 须事先训练或下载
TOKENIZER_TYPE="BertWordPieceTokenizer"   # Megatron 内部路由到 SP 实现
TOKENIZER_MODEL_PATH="${WALPURGIS_SP_MODEL:-tokenizer.model}"
VOCAB_SIZE=30522   # 与 WordPiece 版本保持一致，便于跨 tokenizer 对比实验

_dbg "TOKENIZER_CONF_LOADED"

# ── 检查 SentencePiece 模型文件存在性（上游脚本无此防护）──────────────────────
if [[ ! -f "${TOKENIZER_MODEL_PATH}" ]]; then
    echo "[error:_dbg] SentencePiece 模型文件不存在: ${TOKENIZER_MODEL_PATH}" >&2
    echo "[hint]  可通过 sentencepiece 训练：" >&2
    echo "        spm_train --input=corpus.txt --model_prefix=tokenizer --vocab_size=${VOCAB_SIZE}" >&2
    exit 1
fi

_dbg "PREFLIGHT_DONE"

# ── 模型与训练参数 ──────────────────────────────────────────────────────────────
NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_ATTN_HEADS=16
BATCH_SIZE=32
SEQ_LENGTH=512
MAX_PREDS_PER_SEQ=80
TRAIN_ITERS=1000000
SAVE_ITERS=10000

TRAIN_DATA="wikipedia"
# presplit-sentences: megatron 3573423f4 新增。
# SentencePiece 场景下此参数与 WordPiece 版本含义相同：
# 要求输入 JSON 中的 text 字段已完成句子边界切分，
# 避免训练时的动态 sentence splitting 引入不确定性。
PRESPLIT_SENTENCES="--presplit-sentences"

CHECKPOINT_PATH="${WALPURGIS_CKPT_PATH:-checkpoints/bert_sentencepiece}"

_dbg "CONFIG_LOADED"

# ── 主训练入口 ──────────────────────────────────────────────────────────────────
python pretrain_bert.py \
    --num-layers          ${NUM_LAYERS}             \
    --hidden-size         ${HIDDEN_SIZE}            \
    --num-attention-heads ${NUM_ATTN_HEADS}         \
    --batch-size          ${BATCH_SIZE}             \
    --seq-length          ${SEQ_LENGTH}             \
    --max-preds-per-seq   ${MAX_PREDS_PER_SEQ}      \
    --train-iters         ${TRAIN_ITERS}            \
    --save-iters          ${SAVE_ITERS}             \
    --save                ${CHECKPOINT_PATH}        \
    --tokenizer-type      ${TOKENIZER_TYPE}         \
    --tokenizer-path      ${TOKENIZER_MODEL_PATH}   \
    --vocab-size          ${VOCAB_SIZE}             \
    --train-data          ${TRAIN_DATA}             \
    ${PRESPLIT_SENTENCES}                           \
    --loose-json                                    \
    --text-key            text                      \
    --split               1000,1,1                  \
    --lr                  0.0001                    \
    --lr-decay-style      linear                    \
    --lr-decay-iters      990000                    \
    --warmup              0.01                      \
    --weight-decay        1e-2                      \
    --clip-grad           1.0                       \
    --fp16

_dbg "TRAINING_LAUNCHED"
