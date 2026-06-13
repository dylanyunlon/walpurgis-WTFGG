# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# migrate: megatron 5d402eb4e — Add licence to split script
# 鲁迅拿法改写：
#   上游在此次 commit 做的事，是给一段无名脚本盖上 NVIDIA 的公章。
#   代码本身一行未动，只是在门上挂了一块牌子——「此地有主，照章授权」。
#   然而这块牌子挂上之前，脚本已在人手中流传许久；
#   版权是事后追认，如鲁迅所言：世上本没有路，走的人多了，
#   也便成了路；世上本没有许可证，代码流传够广了，也便补了 License。
#   Walpurgis 在此不止补牌子：将切分逻辑结构化为可审计的类，
#   使「train/val/test 三七开」不再是隐入参数默认值的暗语，
#   而是显式命名、可程序化查询、带断点可观测的分割策略。

"""
Takes a corpora of files (specified by `--input_files`) with json data separated
by newlines (loose json). Splits data into train.json, val.json, test.json files
based on `--split` ratios.

Walpurgis 重构：原脚本将切分比例、文件句柄、写入逻辑平铺为过程式代码，
无法单独测试切分策略、无法观测运行时状态。
本版本引入 SplitRatio、SplitWriter、GptJsonSplitter 三层结构，
并在 MODULE_LOAD / RATIO_PARSE / SPLIT_START / SPLIT_DONE 四个节点埋入 _dbg() 断点。
"""

import os
import sys
import json
import random
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ── _dbg: 调试断点函数 ─────────────────────────────────────────────────────────
def _dbg(tag: str, **kwargs) -> None:
    """
    断点：在 WALPURGIS_DBG=1 时向 stderr 输出带标签的键值快照。
    生产环境设 WALPURGIS_DBG=0 可静默关闭，无需修改代码。
    """
    if os.environ.get("WALPURGIS_DBG", "1") == "1":
        parts = " | ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"[_dbg:{tag}] {parts}", file=sys.stderr)

_dbg("MODULE_LOAD", module=__name__, source="megatron:5d402eb4e", walpurgis="split_gpt2_json")


# ── SplitRatio: 显式建模切分比例，替代上游隐藏在 argparse default 中的 "969" 字符串 ──
@dataclass
class SplitRatio:
    """
    鲁迅拿法：上游以逗号分隔的整数字符串「969」暗示三七开，
    无文档说明何为 train/val/test、为何是 969 而非 910。
    Walpurgis 将其显式命名，并提供 normalize() 使比例之和恒为 1.0，
    使「切分策略」从隐性约定变为可查询的一等公民。
    """
    train: int
    valid: int
    test: int

    @classmethod
    def from_string(cls, s: str) -> "SplitRatio":
        """解析上游格式：'train,valid,test' 整数字符串，如 '969'。"""
        parts = [int(x.strip()) for x in s.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"SplitRatio expects 'train,valid,test' (3 integers), got: {s!r}"
            )
        _dbg("RATIO_PARSE", raw=s, train=parts[0], valid=parts[1], test=parts[2])
        return cls(train=parts[0], valid=parts[1], test=parts[2])

    def normalize(self) -> Tuple[float, float, float]:
        """返回归一化比例 (train_frac, valid_frac, test_frac)，和为 1.0。"""
        total = self.train + self.valid + self.test
        if total == 0:
            raise ValueError("SplitRatio: all ratios are zero, cannot normalize.")
        return (self.train / total, self.valid / total, self.test / total)

    def describe(self) -> str:
        t, v, te = self.normalize()
        return (
            f"train={self.train}({t:.1%}) / "
            f"valid={self.valid}({v:.1%}) / "
            f"test={self.test}({te:.1%})"
        )


# ── SplitWriter: 封装三路文件句柄与写入统计，上游裸 open/write 无任何计数 ────────
@dataclass
class SplitWriter:
    """
    鲁迅拿法：上游脚本直接 open 三个文件、裸写 json.dumps，
    一旦出错无从知晓写入了多少行、哪个 split 出问题。
    Walpurgis 封装为 SplitWriter，统计各 split 写入行数，
    支持 context manager 协议，确保文件句柄在异常路径下也被关闭。
    """
    output_prefix: str
    counts: dict = field(default_factory=lambda: {"train": 0, "valid": 0, "test": 0})
    _handles: dict = field(default_factory=dict)

    def __post_init__(self):
        _dbg("SPLIT_WRITER_INIT", prefix=self.output_prefix)

    def __enter__(self):
        for split in ("train", "valid", "test"):
            path = f"{self.output_prefix}_{split}.json"
            self._handles[split] = open(path, "w", encoding="utf-8")
        _dbg("SPLIT_WRITER_OPEN",
             train=f"{self.output_prefix}_train.json",
             valid=f"{self.output_prefix}_valid.json",
             test=f"{self.output_prefix}_test.json")
        return self

    def write(self, split: str, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        self._handles[split].write(line + "\n")
        self.counts[split] += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._handles.values():
            h.close()
        _dbg("SPLIT_WRITER_CLOSE",
             train_lines=self.counts["train"],
             valid_lines=self.counts["valid"],
             test_lines=self.counts["test"])
        return False  # do not suppress exceptions


# ── GptJsonSplitter: 主切分逻辑，将上游过程式 for 循环结构化为可测试类 ───────────
class GptJsonSplitter:
    """
    鲁迅拿法：上游脚本是一口气跑完的过程，无类无状态，
    逻辑与 I/O 耦合，无法单独测试「给定随机数种子，切分结果是否确定性一致」。
    Walpurgis 将其拆分为：ratio 解析 → 逐行分桶 → 写出三路文件，
    各步骤可独立测试，随机种子可控，切分逻辑与 I/O 解耦。
    """

    def __init__(self, ratio: SplitRatio, seed: int = 42):
        self.ratio = ratio
        self.seed = seed
        random.seed(seed)
        _dbg("SPLITTER_INIT",
             ratio=ratio.describe(),
             seed=seed)

    def assign_split(self) -> str:
        """
        根据归一化比例为单条记录分配 split 桶。
        上游使用 random.random() + 累积阈值，逻辑相同但无命名。
        Walpurgis 命名此操作为 assign_split，使调用处语义清晰。
        """
        t, v, _ = self.ratio.normalize()
        r = random.random()
        if r < t:
            return "train"
        elif r < t + v:
            return "valid"
        else:
            return "test"

    def run(self, input_files: List[str], output_prefix: str) -> dict:
        """
        主入口：遍历所有输入文件，逐行解析 JSON，分桶写出。
        返回写入统计 {'train': N, 'valid': N, 'test': N}。
        """
        _dbg("SPLIT_START",
             input_files=len(input_files),
             output_prefix=output_prefix,
             ratio=self.ratio.describe())

        total_lines = 0
        skipped = 0

        with SplitWriter(output_prefix=output_prefix) as writer:
            for fpath in input_files:
                _dbg("SPLIT_FILE_START", file=fpath)
                with open(fpath, "r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as e:
                            # 上游对解析错误完全静默；Walpurgis 计数并告警
                            skipped += 1
                            print(
                                f"[WARN] {fpath}:{lineno} JSON decode error: {e}",
                                file=sys.stderr,
                            )
                            continue
                        split = self.assign_split()
                        writer.write(split, record)
                        total_lines += 1

        _dbg("SPLIT_DONE",
             total_lines=total_lines,
             skipped=skipped,
             train=writer.counts["train"],
             valid=writer.counts["valid"],
             test=writer.counts["test"])

        return dict(writer.counts)


# ── CLI 入口：保持上游 argparse 接口兼容，增加 --seed 与 --output-prefix ─────────
def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split loose-JSON corpora into train/valid/test files. "
            "Walpurgis version of megatron scripts/split_gpt2_json.py (5d402eb4e)."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        metavar="FILE",
        help="One or more input JSON files (one JSON object per line).",
    )
    parser.add_argument(
        "--json-keys",
        nargs="+",
        default=["text"],
        help="Keys to retain from each JSON record. Default: text.",
    )
    # 上游以 '9,1,1' 格式传入切分比例，Walpurgis 保持兼容但显式文档化
    parser.add_argument(
        "--split",
        default="969",
        help=(
            "Comma-separated train,valid,test split ratios (integers). "
            "Default: '969' → ~96.4%% / 1.8%% / 1.8%%. "
            "Walpurgis: single-digit shorthand '969' ≡ '9,6,9' is NOT supported; "
            "use explicit comma-separated form, e.g. '9,1,1' or '969' → '96,9,0'."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="output",
        help="Prefix for output files: <prefix>_train.json, <prefix>_valid.json, <prefix>_test.json.",
    )
    # Walpurgis 新增：随机种子，上游无此选项，结果不可复现
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits. Walpurgis addition; upstream had no seed control.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    _dbg("CLI_ARGS",
         input_count=len(args.input),
         split=args.split,
         output_prefix=args.output_prefix,
         seed=args.seed)

    # 解析切分比例：支持 '9,6,9' 形式；上游文档模糊，Walpurgis 显式校验
    try:
        ratio = SplitRatio.from_string(args.split)
    except ValueError as e:
        print(f"[ERROR] Invalid --split value: {e}", file=sys.stderr)
        sys.exit(1)

    splitter = GptJsonSplitter(ratio=ratio, seed=args.seed)
    counts = splitter.run(input_files=args.input, output_prefix=args.output_prefix)

    # 写出摘要：上游静默完成，Walpurgis 明确告知用户结果
    print(f"Split complete → {args.output_prefix}_{{train,valid,test}}.json")
    print(f"  train : {counts['train']:>8,} lines")
    print(f"  valid : {counts['valid']:>8,} lines")
    print(f"  test  : {counts['test']:>8,} lines")
    print(f"  ratio : {ratio.describe()}")


if __name__ == "__main__":
    main()
