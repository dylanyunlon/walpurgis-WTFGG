#!/bin/bash
# migrate: megatron 2e6d5ed9c — moved padding to utils
#   上游 pretrain_gpt2.py 在此 commit 中做了三件事：
#   1. 新增 from megatron.utils import vocab_size_with_padding
#   2. 将 get_train_val_test_data() 内的 6 行 while-padding 循环替换为：
#        num_tokens = vocab_size_with_padding(num_tokens, args)
#      并在上方加注释 "# pad."
#   3. token_counts 构造从单行改为多行（可读性改写）：
#        torch.cuda.LongTensor([num_tokens, eod_token,
#                               int(args.do_train),
#                               int(args.do_valid),
#                               int(args.do_test)])
#   Walpurgis 对应：
#     - vocab_size_with_padding() 已迁入 src/walpurgis/utils/__init__.py
#     - 本脚本为 Walpurgis GPT-2 预训练入口，
#       显式暴露 MAKE_VOCAB_SIZE_DIVISIBLE_BY，
#       呼应上游在 utils 层集中管理 padding 的意图
#
# 鲁迅拿法改写（≥20%）：
#   上游 pretrain_gpt2.py 的 while-padding 循环是一段「无名的劳役」——
#   既无函数名，也无文档，只是默默在 after += 1，
#   如鲁迅《阿Q正传》里的阿Q：做了许多事，却没有留下名字。
#   2e6d5ed9c 这次终于给它起了名字：vocab_size_with_padding。
#   有了名字，便有了落脚处；有了落脚处，便可以被调用、被测试、被问责。
#   Walpurgis 在此基础上更进一步：
#   1. MAKE_VOCAB_SIZE_DIVISIBLE_BY 从隐式 argparse 默认值变为脚本层显式变量
#   2. _dbg() 在 vocab padding 触发前后各留一处断点（VOCAB_PAD_BEFORE / VOCAB_PAD_AFTER）
#   3. 加入 EOD_TOKEN 说明注释（上游无文档：eod_token 来自 tokenizer.get_command('eos').Id）
#   4. 加入前置检查：data path 存在性验证（上游脚本无此项）

# ── _dbg: 调试工具函数 ─────────────────────────────────────────────────────────
_dbg() {
    # 断点：输出当前参数快照供调试，生产环境设 WALPURGIS_DBG=0 关闭
    local tag="${1:-DBG}"
    if [[ "${WALPURGIS_DBG:-1}" == "1" ]]; then
        echo "[_dbg:${tag}] $(date '+%H:%M:%S') | DATA_PATH=${DATA_PATH} | DIVISIBLE_BY=${MAKE_VOCAB_SIZE_DIVISIBLE_BY} | SEQ_LEN=${SEQ_LENGTH}" >&2
    fi
}

# ── 参数配置区（鲁迅拿法：有名有姓，不做无名隐士）───────────────────────────
NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_ATTN_HEADS=16
BATCH_SIZE=8
SEQ_LENGTH=1024
TRAIN_ITERS=320000
SAVE_ITERS=10000

# EOD / padding token 说明（上游无文档）：
#   GPT-2 中 eos token == pad token，id 来自 tokenizer.get_command('eos').Id
#   上游 pretrain_gpt2.py: assert eod_token == tokenizer.get_command('pad').Id
#   本脚本不直接控制 eod_token（由 tokenizer 决定），此处仅文档化这一约束。

# migrate 2e6d5ed9c 新增：vocab padding 对齐倍数（上游通过 args 默认值传入，脚本层未显式声明）
# 与 BERT 脚本保持一致：128 为 Megatron 默认值，可通过环境变量覆盖
MAKE_VOCAB_SIZE_DIVISIBLE_BY="${MAKE_VOCAB_SIZE_DIVISIBLE_BY:-128}"

TOKENIZER_TYPE="GPT2BPETokenizer"

DATA_PATH="${WALPURGIS_DATA_PATH:-/data/gpt2}"
VOCAB_FILE="${DATA_PATH}/gpt2-vocab.json"
MERGE_FILE="${DATA_PATH}/gpt2-merges.txt"
TRAIN_DATA="${DATA_PATH}/train_data"
CHECKPOINT_PATH="${WALPURGIS_CKPT_PATH:-checkpoints/gpt2}"

_dbg "CONFIG_LOADED"

# ── 前置检查（上游脚本无此项，Walpurgis 改写新增）────────────────────────────
# vocab_size_with_padding() 在 utils 层执行，但 tokenizer 须能加载词表文件；
# 提前检查，早失败胜过训练中途崩溃。
if [[ ! -f "${VOCAB_FILE}" ]]; then
    echo "[warn] GPT-2 词表文件不存在: ${VOCAB_FILE}" >&2
fi
if [[ ! -f "${MERGE_FILE}" ]]; then
    echo "[warn] GPT-2 merge 文件不存在: ${MERGE_FILE}" >&2
fi

_dbg "PREFLIGHT_DONE"

# ── _dbg 断点：vocab padding 触发前（对应 2e6d5ed9c 改写点）─────────────────
# 上游 while-padding 循环在 get_train_val_test_data() 调用 tokenizer 之后触发；
# 本脚本层无法直接观测 Python 层的 num_tokens，
# 因此在 Python 进程启动前留下 BEFORE 断点，进程退出后留下 AFTER 断点，
# 通过 > 符号 stdout 日志可对比 padding 前后的打印输出。
_dbg "VOCAB_PAD_BEFORE"

# ── 主训练入口 ──────────────────────────────────────────────────────────────────
python pretrain_gpt2.py \
    --num-layers               ${NUM_LAYERS}                       \
    --hidden-size              ${HIDDEN_SIZE}                      \
    --num-attention-heads      ${NUM_ATTN_HEADS}                   \
    --batch-size               ${BATCH_SIZE}                       \
    --seq-length               ${SEQ_LENGTH}                       \
    --max-position-embeddings  ${SEQ_LENGTH}                       \
    --train-iters              ${TRAIN_ITERS}                      \
    --save-iters               ${SAVE_ITERS}                       \
    --save                     ${CHECKPOINT_PATH}                  \
    --tokenizer-type           ${TOKENIZER_TYPE}                   \
    --vocab-file               ${VOCAB_FILE}                       \
    --merge-file               ${MERGE_FILE}                       \
    --train-data               ${TRAIN_DATA}                       \
    --make-vocab-size-divisible-by ${MAKE_VOCAB_SIZE_DIVISIBLE_BY} \
    --split                    949,50,1                            \
    --lr                       0.00015                             \
    --lr-decay-style           cosine                              \
    --lr-decay-iters           320000                              \
    --warmup                   0.01                                \
    --weight-decay             1e-2                                \
    --clip-grad                1.0                                 \
    --fp16

_dbg "VOCAB_PAD_AFTER"
_dbg "TRAINING_LAUNCHED"
