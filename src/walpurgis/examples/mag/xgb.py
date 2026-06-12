# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Walpurgis Migration — commit 75cd001
# [FEA] Add New Unsupervised Learning Example (#371) — xgb.py
# Upstream author: Alex Barghi <alexbarghi-nv>
# Migrated by: dylanyunlon <dogechat@163.com>
#
# 改写说明（鲁迅拿法 20%）:
#   1. XGBConfig dataclass 替代散落的 argparse 命名空间——上游直接把 args.xxx 散装
#      传给 dxgb.train，此处封装后可序列化/复现/打印配置摘要
#   2. WalpurgisXGBSplitter 封装 train/test 分割逻辑——上游 inline 的 random 列
#      追加 + partition 过滤 + 列删除三步骤，此处提取为独立方法，便于替换分割策略
#   3. _dbg() 统一调试出口，WALPURGIS_DEBUG=1 时才打印，不污染生产日志
#   4. EmbeddingLoader 封装 dask_cudf.read_parquet 路径拼接，上游两处
#      os.path.join 硬编码，此处提取并加入路径合法性检查
#   5. confusion_summary() 替代上游连续 4 行 print——以 dict 返回
#      {"accuracy", "correct", "total"} 供下游（如 MLflow/W&B）使用
#
# 鲁迅：世上本没有路，走的人多了，也便成了路。——XGBoost 路径亦然。
#
# WALPURGIS_DEBUG=1 可启用全链路断点 print

from __future__ import annotations

import os
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import cupy
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask_cudf

from xgboost import dask as dxgb

print("[M75cd001]")

# ─── 调试工具 ──────────────────────────────────────────────────────────────────

_WALPURGIS_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str = "") -> None:
    """WALPURGIS_DEBUG=1 时打印调试断点。"""
    if _WALPURGIS_DEBUG:
        print(f"[WALPURGIS_DEBUG][xgb][{tag}] {msg}", flush=True)


# ─── 配置 dataclass ─────────────────────────────────────────────────────────────

@dataclass
class XGBConfig:
    """封装 XGBoost 训练超参与路径配置。
    上游直接用 argparse.Namespace 散装传参，此处对象化以便序列化/复现。
    """
    data_dir: str
    num_boost_round: int = 4
    max_depth: int = 10
    eta: float = 0.1
    subsample: float = 0.7
    train_frac: float = 0.8  # 上游硬编码 0.8，此处参数化

    # XGBoost 固定超参（上游 inline dict，此处收拢）
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "verbosity": 2,
        "tree_method": "hist",
        "sampling_method": "gradient_based",
        "objective": "multi:softmax",
        "device": "cuda",
        "eval_metric": "mlogloss",
    })

    def summary(self) -> str:
        return (
            f"XGBConfig(data_dir={self.data_dir!r}, "
            f"rounds={self.num_boost_round}, max_depth={self.max_depth}, "
            f"eta={self.eta}, subsample={self.subsample}, "
            f"train_frac={self.train_frac})"
        )


# ─── 数据加载器 ─────────────────────────────────────────────────────────────────

class EmbeddingLoader:
    """封装 embedding parquet 的路径拼接与合法性检查。
    上游直接调用 dask_cudf.read_parquet(os.path.join(...))，无任何路径校验。
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        _dbg("EmbeddingLoader.__init__", f"data_dir={data_dir!r}")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"[Walpurgis] data_dir 不存在: {data_dir!r}。"
                f"请先运行 mag_lp_mnmg.py 生成 embedding。"
            )

    def load_x(self) -> dask_cudf.DataFrame:
        path = os.path.join(self.data_dir, "x")
        _dbg("EmbeddingLoader.load_x", f"path={path!r}")
        return dask_cudf.read_parquet(path)

    def load_y(self) -> dask_cudf.DataFrame:
        path = os.path.join(self.data_dir, "y")
        _dbg("EmbeddingLoader.load_y", f"path={path!r}")
        return dask_cudf.read_parquet(path)


# ─── 分割器 ────────────────────────────────────────────────────────────────────

class WalpurgisXGBSplitter:
    """封装 dask_cudf DataFrame 的随机 train/test 分割。
    上游 inline 三步（追加 random 列→过滤→删列），此处提取为独立方法便于替换策略。
    """

    def __init__(self, train_frac: float = 0.8, seed: Optional[int] = None) -> None:
        self.train_frac = train_frac
        self.seed = seed
        _dbg("WalpurgisXGBSplitter.__init__", f"train_frac={train_frac}, seed={seed}")

    def split(self, df: dask_cudf.DataFrame):
        """返回 (df_train, df_test)，均已 drop random 列。"""
        _dbg("WalpurgisXGBSplitter.split", f"partitions={df.npartitions}")

        if self.seed is not None:
            cupy.random.seed(self.seed)

        df = df.map_partitions(
            lambda part: part.assign(random=cupy.random.rand(len(part)))
        ).persist()

        df_train = df[df["random"] <= self.train_frac].drop(columns=["random"])
        df_test = df[df["random"] > self.train_frac].drop(columns=["random"])

        _dbg("WalpurgisXGBSplitter.split", "split done")
        return df_train, df_test


# ─── 评估结果 ──────────────────────────────────────────────────────────────────

def confusion_summary(
    predictions_computed: cupy.ndarray,
    labels_computed,
) -> Dict[str, Any]:
    """返回 {"accuracy", "correct", "total"} dict。
    上游直接 4 行 print，此处返回 dict 供下游（MLflow/W&B）使用。
    """
    correct = int((predictions_computed == labels_computed).sum())
    total = int(len(labels_computed))
    accuracy = correct / total if total > 0 else 0.0
    _dbg("confusion_summary", f"acc={accuracy:.4f}, {correct}/{total}")
    return {"accuracy": accuracy, "correct": correct, "total": total}


# ─── 主流程 ────────────────────────────────────────────────────────────────────

def main(cfg: XGBConfig) -> None:
    _dbg("main", cfg.summary())
    print(f"[Walpurgis xgb] config: {cfg.summary()}", flush=True)

    cluster = LocalCUDACluster()
    client = Client(cluster)
    _dbg("main", "Dask cluster started")

    # 加载 embedding
    loader = EmbeddingLoader(cfg.data_dir)
    dfx = loader.load_x()
    dfy = loader.load_y()
    _dbg("main", "parquet loaded")

    df = dfx.join(dfy, how="inner").persist()
    df = df.repartition(npartitions=32)
    df = df.reset_index(drop=True)

    # 分割
    splitter = WalpurgisXGBSplitter(train_frac=cfg.train_frac)
    df_train, df_test = splitter.split(df)

    dfx_train = df_train.drop(columns=["y"]).persist()
    dfy_train = df_train["y"].persist()
    dfx_test = df_test.drop(columns=["y"]).persist()
    dfy_test = df_test["y"].persist()

    _dbg("main", "DMatrix building")
    dtrain = dxgb.DaskDMatrix(client, dfx_train, label=dfy_train)
    dtest = dxgb.DaskDMatrix(client, dfx_test, label=dfy_test)

    # 合并超参（动态字段）
    num_class = int(df.y.nunique().compute())
    _dbg("main", f"num_class={num_class}")
    params = dict(cfg.xgb_params)
    params.update({
        "max_depth": cfg.max_depth,
        "subsample": cfg.subsample,
        "eta": cfg.eta,
        "num_class": num_class,
    })

    print("Training XGBoost model...", flush=True)
    _dbg("main", f"xgb params={params}")
    out = dxgb.train(
        client,
        params,
        dtrain,
        num_boost_round=cfg.num_boost_round,
        evals=[(dtrain, "train"), (dtest, "test")],
    )

    booster = out["booster"]
    print("\nTraining complete!", flush=True)

    # 评估
    print("\nEvaluating model on test set...", flush=True)
    predictions = dxgb.predict(client, booster, dtest)
    predictions_computed = cupy.array(predictions.compute())
    dfy_test_computed = dfy_test.compute()

    result = confusion_summary(predictions_computed, dfy_test_computed)

    print(f"\n{'=' * 50}")
    print("Test Set Evaluation Results (Walpurgis):")
    print(f"{'=' * 50}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Total test samples: {result['total']}")
    print(f"Correct predictions: {result['correct']}")
    print(f"{'=' * 50}\n")

    _dbg("main", "closing cluster")
    client.close()
    cluster.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Walpurgis XGBoost downstream example (from 75cd001)"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="包含 x/ 和 y/ parquet 的目录（由 mag_lp_mnmg.py 生成）")
    parser.add_argument("--num_boost_round", type=int, default=4)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.7)
    parser.add_argument("--train_frac", type=float, default=0.8,
                        help="训练集比例（上游硬编码 0.8，此处参数化）")
    args = parser.parse_args()

    cfg = XGBConfig(
        data_dir=args.data_dir,
        num_boost_round=args.num_boost_round,
        max_depth=args.max_depth,
        eta=args.eta,
        subsample=args.subsample,
        train_frac=args.train_frac,
    )
    main(cfg)
