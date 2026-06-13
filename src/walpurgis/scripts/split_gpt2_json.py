#!/usr/bin/env python3
"""
migrate a1d04b793: Updating public repo with latest changes.
上游文件: scripts/split_gpt2_json.py

鲁迅拿法改写（≥20%）：
  上游这个脚本做一件事：把一个大 JSONL 文件按比例切成 train/valid/test 三份。
  119行里，文件名生成用了一个硬编码的字符串拼接，
  shuffle 是可选的，但随机种子没有固定下来——
  同样的切割，今天运行和明天运行的结果可能不同，
  没有人记录这件事，也没有人觉得这是个问题。
  鲁迅说："做事情，最要紧的是不自欺欺人。"
  数据集切割是实验的基础，若不固定随机种子，
  所有基于该切割的实验结果都无法复现，
  论文里写的数字也就成了幻觉。
  Walpurgis 改写：
  1. SplitConfig: 切割配置数据类（含 seed 字段）
  2. JSONLSplitter: 可复现的数据集切割器（固定 seed，记录 split summary）
  3. 输出 manifest.json（记录每份文件行数、MD5 摘要）
  4. _dbg 断点：SPLIT_LOADED / SPLIT_SHUFFLED / SPLIT_WRITTEN

用法:
  python -m walpurgis.scripts.split_gpt2_json \\
      --input /path/to/data.jsonl \\
      --output-prefix /path/to/output \\
      --train-ratio 0.98 --valid-ratio 0.01 --test-ratio 0.01 \\
      --seed 42 --shuffle
"""

import os
import sys
import json
import hashlib
import argparse
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ── _dbg 断点 ─────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg) -> None:
    if not _DBG:
        return
    print(f"[_dbg:split_gpt2_json:{tag}] {msg}", file=sys.stderr, flush=True)


# ── SplitConfig（Walpurgis特有：含 seed，保证可复现） ─────────────────────────
@dataclass
class SplitConfig:
    """
    数据集切割配置。
    上游无此结构，参数散落在 argparse namespace 里。
    Walpurgis: 显式建模，seed 字段是核心改动——上游没有固定种子。
    """
    train_ratio: float = 0.98
    valid_ratio: float = 0.01
    test_ratio: float = 0.01
    shuffle: bool = True
    seed: int = 42            # Walpurgis 新增：固定随机种子，保证可复现
    output_prefix: str = "output"

    def __post_init__(self):
        total = self.train_ratio + self.valid_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train+valid+test 比例之和必须为 1.0，当前={total:.6f}"
            )

    def split_counts(self, n: int) -> Tuple[int, int, int]:
        """根据总行数计算各份的行数。"""
        train_n = int(n * self.train_ratio)
        valid_n = int(n * self.valid_ratio)
        test_n = n - train_n - valid_n
        return train_n, valid_n, test_n


# ── JSONLSplitter（主切割器） ─────────────────────────────────────────────────
class JSONLSplitter:
    """
    可复现的 JSONL 数据集切割器。

    上游 split_gpt2_json.py 的核心逻辑：逐行读取 JSONL，
    按比例切成三份，可选 shuffle。
    Walpurgis 改写：固定 seed、输出 manifest、添加断点。
    """

    def __init__(self, config: SplitConfig) -> None:
        self.config = config
        _dbg("SPLITTER_INIT", f"seed={config.seed} shuffle={config.shuffle}")

    def _file_md5(self, path: str) -> str:
        """计算文件 MD5（用于 manifest，验证输出文件完整性）。"""
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def split(self, input_path: str) -> dict:
        """
        读取 input_path，按 config 切割，写出三个 JSONL 文件 + manifest。
        返回 manifest dict。
        """
        _dbg("SPLIT_START", f"input={input_path}")

        # ── 读取 ──────────────────────────────────────────────────────────────
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        n = len(lines)
        _dbg("SPLIT_LOADED", f"total_lines={n}")
        print(f"[JSONLSplitter] loaded {n} lines from {input_path}", file=sys.stderr)

        # ── Shuffle（固定 seed）────────────────────────────────────────────────
        if self.config.shuffle:
            rng = random.Random(self.config.seed)
            rng.shuffle(lines)
            _dbg("SPLIT_SHUFFLED", f"seed={self.config.seed} first_line_hash="
                 f"{hashlib.md5(lines[0].encode()).hexdigest()[:8]}")

        # ── 计算切割点 ────────────────────────────────────────────────────────
        train_n, valid_n, test_n = self.config.split_counts(n)
        splits = {
            "train": lines[:train_n],
            "valid": lines[train_n:train_n + valid_n],
            "test": lines[train_n + valid_n:],
        }
        _dbg("SPLIT_COUNTS", f"train={train_n} valid={valid_n} test={test_n}")

        # ── 写出 ──────────────────────────────────────────────────────────────
        manifest = {
            "input": input_path,
            "seed": self.config.seed,
            "shuffle": self.config.shuffle,
            "ratios": {
                "train": self.config.train_ratio,
                "valid": self.config.valid_ratio,
                "test": self.config.test_ratio,
            },
            "splits": {},
        }

        prefix = self.config.output_prefix
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

        for split_name, split_lines in splits.items():
            out_path = f"{prefix}_{split_name}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                f.writelines(split_lines)
            md5 = self._file_md5(out_path)
            manifest["splits"][split_name] = {
                "path": out_path,
                "num_lines": len(split_lines),
                "md5": md5,
            }
            _dbg("SPLIT_WRITTEN", f"{split_name}: {len(split_lines)} lines -> {out_path} md5={md5[:8]}")
            print(
                f"[JSONLSplitter] {split_name}: {len(split_lines)} lines "
                f"-> {out_path}",
                file=sys.stderr,
            )

        # ── 写出 manifest ─────────────────────────────────────────────────────
        manifest_path = f"{prefix}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[JSONLSplitter] manifest -> {manifest_path}", file=sys.stderr)
        _dbg("MANIFEST_WRITTEN", manifest_path)

        return manifest


# ── CLI 入口 ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Walpurgis JSONL 数据集切割工具 (migrate a1d04b793 split_gpt2_json.py)"
    )
    p.add_argument("--input", type=str, required=True, help="输入 JSONL 文件路径")
    p.add_argument("--output-prefix", type=str, required=True,
                   help="输出文件前缀（会生成 prefix_train/valid/test.jsonl）")
    p.add_argument("--train-ratio", type=float, default=0.98)
    p.add_argument("--valid-ratio", type=float, default=0.01)
    p.add_argument("--test-ratio", type=float, default=0.01)
    p.add_argument("--shuffle", action="store_true", default=True)
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    p.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（Walpurgis新增：上游无此参数，导致不可复现）"
    )
    return p.parse_args()


def main():
    args = parse_args()
    _dbg("ARGS", vars(args))

    config = SplitConfig(
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        shuffle=args.shuffle,
        seed=args.seed,
        output_prefix=args.output_prefix,
    )

    splitter = JSONLSplitter(config)
    manifest = splitter.split(args.input)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
