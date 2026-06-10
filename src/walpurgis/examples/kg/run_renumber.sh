#!/bin/bash
# run_renumber.sh — 05fe6f4 迁移: KG 重编号启动脚本
#
# migrate 05fe6f4: [FEA] Knowledge Graph/Graph Database Renumbering
#
# 上游 (05fe6f4 run_renumber.sh):
#   torchrun --nnodes 1 --nproc-per-node 2 renumber_kg.py \
#     --node_types "paper,author" ...
#   硬编码路径 /home/nfs/abarghi/test_renumber_kg/...
#
# Walpurgis 改写20%(鲁迅拿法):
#   - 改为环境变量驱动 (DATA_ROOT / NNODES / NPROC_PER_NODE)，
#     替代上游硬编码路径，适配 Walpurgis 部署环境
#   - 加 WALPURGIS_DEBUG=1 透传，触发 renumber_kg.py 断点 print
#   - 加 set -euo pipefail 防静默失败
#   - 保留上游 torchrun 参数结构，与 05fe6f4 完全对应
#
# 使用示例 (单机双卡, 对应上游默认配置):
#   DATA_ROOT=/data/kg bash run_renumber.sh
#
# 使用示例 (调试模式):
#   WALPURGIS_DEBUG=1 DATA_ROOT=/data/kg bash run_renumber.sh
#
# 作者: dylanyunlon<dogechat@163.com>

set -euo pipefail

# ── 环境变量 (改写: 替代上游硬编码路径) ──────────────────────────
DATA_ROOT="${DATA_ROOT:-/home/nfs/abarghi/test_renumber_kg}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# 调试开关: WALPURGIS_DEBUG=1 开启 renumber_kg.py 全链路断点 print
# 对应上游: 无调试开关
export WALPURGIS_DEBUG="${WALPURGIS_DEBUG:-0}"

if [ "${WALPURGIS_DEBUG}" = "1" ]; then
    echo "[DEBUG 05fe6f4 run_renumber.sh] DATA_ROOT=${DATA_ROOT}" >&2
    echo "[DEBUG 05fe6f4 run_renumber.sh] NNODES=${NNODES} NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
fi

# ── torchrun 启动 (对应上游结构) ──────────────────────────────────
# 上游: torchrun --nnodes 1 --nproc-per-node 2 renumber_kg.py ...
# 改写: 路径通过 DATA_ROOT 变量传入，SCRIPT_DIR 动态定位脚本位置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

torchrun \
    --nnodes "${NNODES}" \
    --nproc-per-node "${NPROC_PER_NODE}" \
    "${SCRIPT_DIR}/renumber_kg.py" \
        --node_types "paper,author" \
        --node_input_folders "${DATA_ROOT}/paper,${DATA_ROOT}/author" \
        --node_output_folders "${DATA_ROOT}/paper_renumbered,${DATA_ROOT}/author_renumbered" \
        --node_colname "ID" \
        --edge_types "paper,cites,paper;author,writes,paper" \
        --edge_input_folders "${DATA_ROOT}/paper_cites_paper,${DATA_ROOT}/author_writes_paper" \
        --edge_output_folders "${DATA_ROOT}/paper_cites_paper_renumbered,${DATA_ROOT}/author_writes_paper_renumbered" \
        --source_colname "SRC" \
        --destination_colname "DST" \
        --input_format "csv" \
        --output_format "csv" \
        --use_managed_memory
