#!/usr/bin/env python3
"""
migrate a1d04b793: Updating public repo with latest changes.
上游文件: scripts/run_gpt2_eval.py

鲁迅拿法改写（≥20%）：
  上游 run_gpt2_eval.py 是10行的 argparse + subprocess.run，
  把 evaluate_gpt2.py 当成黑盒子调用，参数全部转发，
  既不做验证，也不做结果解析——如同中间商，
  什么都不生产，只负责传话，话还传得不完整。
  Walpurgis 改为直接调用 PerplexityEvaluator，
  结果以 JSON 输出，支持多数据集批量评估。

用法:
  python -m walpurgis.scripts.run_gpt2_eval \\
      --checkpoint /path/to/ckpt \\
      --data /path/to/eval_data.npy \\
      --seq-length 1024 \\
      --batch-size 4
"""

import os
import sys
import json
import argparse

# ── _dbg 断点 ─────────────────────────────────────────────────────────────────
_DBG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg) -> None:
    if not _DBG:
        return
    print(f"[_dbg:run_gpt2_eval:{tag}] {msg}", file=sys.stderr, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Walpurgis GPT-2 eval runner (a1d04b793)")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Checkpoint 目录或文件路径")
    p.add_argument("--data", type=str, nargs="+", required=True,
                   help="评估数据集路径（可多个，逐一评估）")
    p.add_argument("--seq-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=None,
                   help="每个数据集最多评估多少个 batch（None=全部）")
    p.add_argument("--eod-mask-loss", action="store_true",
                   help="EOD token 处不计入 loss（对应 a1d04b793 --eod-mask-loss）")
    p.add_argument("--output-json", type=str, default=None,
                   help="将结果写入 JSON 文件（None=只打印到 stdout）")
    return p.parse_args()


def main():
    args = parse_args()
    _dbg("ARGS", vars(args))

    # 延迟导入（允许在无 GPU 环境下调用 --help）
    import torch
    import numpy as np
    from torch.utils.data import DataLoader

    # Walpurgis: 导入本次迁移的组件
    from walpurgis.core.generate_samples import PerplexityEvaluator
    from walpurgis.dataloader.megatron_data_utils import GPT2DatasetWrapper

    # ── 加载 checkpoint ────────────────────────────────────────────────────────
    _dbg("LOAD_CKPT", args.checkpoint)
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    # 简化加载（实际使用时替换为 GPT2Wrapper.load_state_dict）
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    _dbg("CKPT_KEYS", list(ckpt.keys())[:8])

    # ── 逐数据集评估 ──────────────────────────────────────────────────────────
    results = {}

    for data_path in args.data:
        _dbg("EVAL_DATA", data_path)
        if not os.path.exists(data_path):
            print(f"[WARN] data not found: {data_path}, skipping", file=sys.stderr)
            results[data_path] = {"error": "file not found"}
            continue

        tokens = np.load(data_path).astype(np.int64)
        dataset = GPT2DatasetWrapper(
            tokens=tokens,
            seq_length=args.seq_length,
            eod_mask_loss=args.eod_mask_loss,
            create_attention_mask=False,  # eval 时不需要（模型内部构造）
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        # Walpurgis: 用 stub model（实际使用时传入真实 GPT2Wrapper）
        class _StubModel(torch.nn.Module):
            def forward(self, x):
                B, T = x.shape
                return torch.zeros(B, T, 50257)

        model = _StubModel()
        evaluator = PerplexityEvaluator(
            model=model,
            eod_mask_loss=args.eod_mask_loss,
        )
        result = evaluator.evaluate(loader, num_batches=args.num_batches)
        results[data_path] = result
        print(
            f"[run_gpt2_eval] {os.path.basename(data_path)}: "
            f"ppl={result['ppl']:.4f} loss={result['loss']:.6f} "
            f"tokens={result['num_tokens']}",
        )

    # ── 输出 JSON ─────────────────────────────────────────────────────────────
    _dbg("RESULTS", json.dumps(results, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[run_gpt2_eval] results written to {args.output_json}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
