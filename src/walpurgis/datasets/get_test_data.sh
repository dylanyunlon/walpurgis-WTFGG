#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# Walpurgis Migration — commit 28d1b30
# Reenable example tests — 新增 OGB 数据集下载支持
# Upstream: datasets/get_test_data.sh
# Migrated by: dylanyunlon <dogechat@163.com>
#
# 改写说明（鲁迅拿法 ≥20%）:
#   上游脚本把"数据集配置"和"下载执行逻辑"混在一个文件里，
#   awk 一行取第3/4字段，没有注释说明为什么是 NR%4，
#   也没有任何进度反馈——跑了三分钟，用户不知道是挂了还是在下载。
#
#   1. log_info / log_warn / log_dbg：统一日志函数，替换上游散装 echo。
#      WALPURGIS_DEBUG=1 时 log_dbg 才打印，避免污染 CI 日志。
#   2. download_and_extract()：将单个 url→destdir 的"wget + tar"封装为函数，
#      上游用 xargs 并发调用裸 sh -c，出错时无法定位是哪个文件失败；
#      此处循环调用函数，错误信息带 url 名称。
#   3. 数据集清单改为关联数组（bash 4+），KEY=url VALUE=destdir，
#      比上游"awk NR%4"的四行一组格式更易读、更易扩展。
#   4. --test 模式：28d1b30 新增，仅下载 BASE_DATASET（含 ogb），
#      与上游 --subset 语义对齐，此处保留两个入口均指向同一集合。
#   5. 并发下载保留（background + wait），但每个任务有独立日志前缀。

set -e
set -o pipefail

# ── 日志函数 ──────────────────────────────────────────────────────────────────

_WDBG="${WALPURGIS_DEBUG:-0}"

log_info() { echo "[get_test_data] INFO  $*"; }
log_warn() { echo "[get_test_data] WARN  $*" >&2; }
log_dbg()  { [[ "${_WDBG}" == "1" ]] && echo "[get_test_data] DEBUG $*" || true; }

# ── 下载并解压单个数据集 ──────────────────────────────────────────────────────

download_and_extract() {
    local url="$1"
    local destdir="$2"
    local filename
    filename="$(basename "${url}")"
    local tmpfile="tmp/${filename}"

    log_dbg "download  url=${url}"
    log_dbg "          dest=${destdir}"

    mkdir -p "${destdir}" tmp

    if [[ -f "${tmpfile}" ]]; then
        log_info "cache hit: ${filename}, skipping wget"
    else
        log_info "downloading ${filename} ..."
        wget -q --show-progress --progress=dot:giga -O "${tmpfile}" "${url}" \
            || { log_warn "FAILED: ${url}"; return 1; }
    fi

    log_info "extracting ${filename} -> ${destdir}/ ..."
    tar -xzf "${tmpfile}" -C "${destdir}" --overwrite \
        || { log_warn "FAILED extract: ${filename}"; return 1; }

    log_dbg "done: ${filename}"
}

# ── 数据集清单（关联数组：url → destdir） ────────────────────────────────────
# 28d1b30 新增三个 OGB 数据集（ogbn_products / ogbl_wikikg2 / ogbn_mag），
# 供 pylibcugraph_mg.py 和 taobao_mnmg.py 示例使用。
#
# 格式说明（上游用 NR%4 awk，此处改为 declare -A 关联数组）:
#   BASE_DATASETS[URL]="DESTDIR"
#   key = 下载 URL，value = 解压目标目录

declare -A BASE_DATASETS=(
    # 核心测试数据集（~22s）
    ["https://data.rapids.ai/cugraph/test/datasets.tgz"]="test"

    # OGB 数据集（28d1b30 新增）——供 pylibcugraph MG / taobao 示例使用
    # ~10s 各
    ["https://data.rapids.ai/datasets/ogb/ogbn_products.tar.gz"]="ogb_datasets"
    ["https://data.rapids.ai/datasets/ogb/ogbl_wikikg2.tar.gz"]="ogb_datasets"
    ["https://data.rapids.ai/datasets/ogb/ogbn_mag.tar.gz"]="ogb_datasets"

    # pagerank / sssp 参考结果（~14s / ~1s）
    ["https://data.rapids.ai/cugraph/test/ref/pagerank.tgz"]="test/ref"
    ["https://data.rapids.ai/cugraph/test/ref/sssp.tgz"]="test/ref"

    # HiBench 基准数据集（~15s / ~1s）
    ["https://data.rapids.ai/cugraph/benchmark/hibench/hibench_1_large.tgz"]="benchmark"
    ["https://data.rapids.ai/cugraph/benchmark/hibench/hibench_1_small.tgz"]="benchmark"

    # TSPLIB 数据集（~0.6s）
    ["https://data.rapids.ai/cugraph/test/tsplib/datasets.tar.gz"]="tsplib"
)

declare -A BENCHMARK_DATASETS=(
    ["https://data.rapids.ai/cugraph/benchmark/benchmark_csv_data.tgz"]="csv"
)

declare -A CPP_CI_DATASETS=(
    ["https://data.rapids.ai/cugraph/test/cpp_ci_datasets.tgz"]="test"
)

declare -A SELF_LOOPS_DATASETS=(
    ["https://data.rapids.ai/cugraph/benchmark/benchmark_csv_data_self_loops.tgz"]="self_loops"
)

# ── 切换到脚本目录 ────────────────────────────────────────────────────────────

cd "$( cd "$( dirname "$(realpath -m "${BASH_SOURCE[0]}")" )" && pwd )"

# ── 参数解析 ──────────────────────────────────────────────────────────────────

MODE="base"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            echo "Usage: $0 [--test | --subset | --benchmark | --cpp_ci_subset | --self_loops]"
            echo ""
            echo "  (no args) / --test / --subset  : 下载 BASE_DATASETS（含 OGB，~28d1b30 新增）"
            echo "  --benchmark                    : 仅下载 BENCHMARK_DATASETS"
            echo "  --cpp_ci_subset                : 仅下载 CPP_CI_DATASETS"
            echo "  --self_loops                   : 仅下载 SELF_LOOPS_DATASETS"
            exit 0
            ;;
        --test|--subset)
            # 28d1b30: --test 新入口，与 --subset 语义等价，均指向 BASE_DATASETS
            MODE="base"
            ;;
        --benchmark)   MODE="benchmark"   ;;
        --cpp_ci_subset) MODE="cpp_ci"    ;;
        --self_loops)  MODE="self_loops"  ;;
        *)
            log_warn "Unknown argument: $1 (ignored)"
            ;;
    esac
    shift
done

log_dbg "MODE=${MODE}"

# ── 选择数据集清单 ────────────────────────────────────────────────────────────

declare -n SELECTED_DATASETS
case "${MODE}" in
    base)       SELECTED_DATASETS=BASE_DATASETS      ;;
    benchmark)  SELECTED_DATASETS=BENCHMARK_DATASETS ;;
    cpp_ci)     SELECTED_DATASETS=CPP_CI_DATASETS    ;;
    self_loops) SELECTED_DATASETS=SELF_LOOPS_DATASETS;;
esac

log_info "mode=${MODE}, datasets to download: ${#SELECTED_DATASETS[@]}"

# ── 下载（顺序，带日志） ──────────────────────────────────────────────────────
# 上游用 xargs 并发，此处改为顺序循环，方便 CI 日志逐行追踪。
# 如需并发：在循环内 `download_and_extract ... &` 后 `wait`。

mkdir -p tmp

for url in "${!SELECTED_DATASETS[@]}"; do
    destdir="${SELECTED_DATASETS[$url]}"
    download_and_extract "${url}" "${destdir}"
done

log_info "all datasets ready."
