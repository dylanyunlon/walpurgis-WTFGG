"""
bitcoin_rf.py — 940ab01 迁移: Elliptic Bitcoin 随机森林分类器

migrate 940ab01: [FEA] Add Elliptic Bitcoin fraud example

上游变化 (940ab01, cugraph-gnn /
  python/cugraph-pyg/cugraph_pyg/examples/fraud/bitcoin_rf.py):
  全新文件，83行。核心逻辑:

  1. 依赖: cudf / cupy / cuml.metrics / cuml.ensemble.RandomForestClassifier
     依赖链: cugraph-pyg 新增 cuml==25.8.* 依赖 (见 pyproject.toml / dependencies.yaml)

  2. train_rf(X_train, y_train):
     - RandomForestClassifier().fit() — cuML GPU 随机森林
     - 返回已训练模型

  3. show_confusion_matrix(y_test, prob, name):
     - confusion_matrix(y_test, prob.argmax(axis=1))
     - acc = (y_test == prob.argmax(axis=1)).sum() / len(y_test)
     - auc = roc_auc_score(y_test, prob[:, 1])
     - print 混淆矩阵 / 准确率 / AUC

  4. __main__:
     - EllipticBitcoinDataset 加载; data.x = data.x[:, :94]
     - df = cudf.read_parquet(args.embedding_dir) — 读取 bitcoin_mnmg.py 生成的目录
     - y / z / X 从 df 提取; X_train/X_test/y_train/y_test/z_train/z_test 按 mask 切分
     - RF+GNN: train_rf(X_train, y_train).predict_proba(X_test)
     - GNN Only: cupy.stack([z_test, 1-z_test], axis=1) 作为 prob
     - RF Only: train_rf(data.x[train_mask], data.y[train_mask])

  Knuth 审查:
    1. diff 对比源:
       - df = cudf.read_parquet(args.embedding_dir)
         read_parquet 接受目录时合并该目录下所有 parquet 文件——
         但 bitcoin_mnmg.py 按 rank 写出多个文件，合并后行数 = rank0_rows + rank1_rows + ...
         而 data.train_mask / data.test_mask 是基于 data.num_nodes 的全量 mask，
         两者长度不一致，X_train = X[data.train_mask] 会越界或静默错位
       - z_train / z_test 在 show_confusion_matrix 中从未使用，是死代码
       - X_train 是 cupy array，y_train 来自 cudf column.to_cupy()，
         两者 dtype 未对齐检查 (cuML RF 对 float32/int32 有严格要求)
    2. 用户角度 bug:
       - cudf.read_parquet(args.embedding_dir) 读的是目录，
         若 embedding_dir 下有多个实验的 parquet (不同超参)，全部被合并——
         用户以为读的是某次实验，实际是多次实验混合，训练数据污染
       - data.train_mask 在 EllipticBitcoin 中将 y=2 (unknown) 的节点排除，
         但 y=2 的节点依然存在于 df 中 (bitcoin_mnmg.py 写出全量节点)，
         X[data.train_mask] 按 mask 索引 cupy array 时，mask 长度必须等于 X 行数，
         否则 cupy IndexError
       - y_train = y[data.train_mask] 中 y 来自 df["y"] (GNN 写出时的 float)，
         而 RandomForestClassifier 在 cuML 中需要 int32 标签，
         float 标签可能被 cuML 接受但 precision 损失或抛 TypeError
    3. 系统角度安全:
       - cudf.read_parquet(args.embedding_dir) 无路径校验，
         embedding_dir 为相对路径时行为依赖 cwd，CI 环境 cwd 不固定
       - RandomForestClassifier() 使用全部默认参数，
         n_estimators=100 (cuML 默认)，无随机种子，结果不可复现，
         CI 中 AUC 波动可导致误报
       - prob[:, 1] 作为 roc_auc_score 正类概率，
         依赖 RF 的第2列是正类 (label=1) 概率——cuML predict_proba 列顺序
         与 sklearn 一致 (升序 class label)，但无断言保证，静默错误风险

Walpurgis 改写20%(鲁迅拿法):
  - BitcoinRfArgs: 将 argparse Namespace 封装为强类型 dataclass，
    加 validate() 校验路径规范化 (上游无任何校验)
  - EmbeddingDataset: 封装 cudf.read_parquet + mask 对齐逻辑，
    load() 方法含对齐检查 + 调试打印 (上游直接 df[mask] 无对齐验证)
  - RfExperiment: 封装单次 RF 实验 (train_rf + show_confusion_matrix)，
    run() 方法加 dtype 检查 + 随机种子参数 (上游全部默认参数)
  - 断点调试: WALPURGIS_DEBUG=1 开启全链路 print，覆盖:
    - args 解析后 dump
    - EmbeddingDataset.load(): parquet 路径、df.shape、列名、mask 对齐检查
    - X/y/z 提取: shape、dtype、unique 值
    - X_train/X_test/y_train/y_test: shape 和 class distribution
    - 每次 RfExperiment.run(): fit 前 shape、predict_proba shape、混淆矩阵详情
    - GNN Only: z_test 分布统计

作者: dylanyunlon<dogechat@163.com>
"""

import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ──────────────────────────────────────────────
# 调试开关: WALPURGIS_DEBUG=1 开启断点级 print
# ──────────────────────────────────────────────
_DEBUG = os.environ.get("WALPURGIS_DEBUG", "0") == "1"


def _dbg(tag: str, msg: str) -> None:
    """断点调试: bitcoin_rf 专用 print。

    对应上游 940ab01 bitcoin_rf.py 中仅有的 3 次 print(f"=== {name} ===...") ——
    无任何中间状态输出，调试信息全靠此 _dbg 补全。
    """
    if _DEBUG:
        print(f"[DEBUG 940ab01 bitcoin_rf | {tag}] {msg}", file=sys.stderr, flush=True)



# ──────────────────────────────────────────────────────────────────────────────
# _DatasetDownloadGuard — a24978e 派生迁移（与 bitcoin_mnmg.py 对称）
# ──────────────────────────────────────────────────────────────────────────────
_CI_SKIP_MSG_RF = "a24978e: Bitcoin 数据集暂时不可用（SSL 问题，PR #230）。设置 WALPURGIS_SKIP_BITCOIN=1 可跳过。"
_BITCOIN_PROBE_URL_RF = "https://data.pyg.org"


class _DatasetDownloadGuard:
    """bitcoin_rf.py 版本：与 bitcoin_mnmg.py 的 _DatasetDownloadGuard 对称。"""
    def __init__(self, probe_url=_BITCOIN_PROBE_URL_RF, timeout=3.0):
        self._probe_url, self._timeout = probe_url, timeout
        _dbg("DatasetDownloadGuard", f"rf-probe probe_url={probe_url}")

    def check_or_skip(self):
        _dbg("DatasetDownloadGuard.check_or_skip", "rf 开始可用性检查")
        if os.environ.get("WALPURGIS_SKIP_BITCOIN", "0").strip() == "1":
            import warnings; warnings.warn(f"WALPURGIS_SKIP_BITCOIN=1 — rf 跳过。{_CI_SKIP_MSG_RF}", RuntimeWarning, stacklevel=2)
            _dbg("DatasetDownloadGuard.check_or_skip", "决策=SKIP reason=env")
            return False
        try:
            import urllib.request
            urllib.request.urlopen(urllib.request.Request(self._probe_url, method="HEAD"), timeout=self._timeout)
            _dbg("DatasetDownloadGuard.check_or_skip", f"{self._probe_url} 可达 ✓ 决策=OK")
            return True
        except Exception as exc:
            import warnings; warnings.warn(f"Bitcoin 数据集不可达: {exc}\n{_CI_SKIP_MSG_RF}", RuntimeWarning, stacklevel=2)
            _dbg("DatasetDownloadGuard.check_or_skip", f"决策=SKIP ssl_fail={exc}")
            return False

# ──────────────────────────────────────────────────────────────────────────────
# BitcoinRfArgs — 强类型参数对象
# ──────────────────────────────────────────────────────────────────────────────
# 上游 (940ab01): argparse.Namespace，散落 args.xxx 访问。
# 改写: dataclass + validate()。


@dataclass
class BitcoinRfArgs:
    """
    对应上游 parse_args() 返回的 argparse.Namespace。

    上游字段: dataset_root, embedding_dir
    改写增加: validate() — 路径规范化校验
    """

    dataset_root: str = "./data"
    embedding_dir: str = "./results"

    def validate(self) -> None:
        """路径规范化校验，防止 '..' 路径穿越。"""
        for name, path in [("dataset_root", self.dataset_root), ("embedding_dir", self.embedding_dir)]:
            normalized = os.path.normpath(path)
            if ".." in normalized.split(os.sep):
                raise ValueError(
                    f"[BitcoinRfArgs] {name}='{path}' 含路径穿越组件 '..'，拒绝"
                )

    @classmethod
    def from_namespace(cls, ns) -> "BitcoinRfArgs":
        return cls(dataset_root=ns.dataset_root, embedding_dir=ns.embedding_dir)

    def debug_dump(self) -> None:
        _dbg(
            "BitcoinRfArgs.debug_dump",
            f"dataset_root={self.dataset_root!r} embedding_dir={self.embedding_dir!r}",
        )


# ──────────────────────────────────────────────────────────────────────────────
# EmbeddingDataset — 封装 cudf.read_parquet + mask 对齐
# ──────────────────────────────────────────────────────────────────────────────
# 上游 (940ab01):
#   df = cudf.read_parquet(args.embedding_dir)  # 读整个目录合并所有 parquet
#   X = df.drop(columns=["y","z"]).to_cupy()
#   X_train = X[data.train_mask]  # 若合并后行数 != data.num_nodes，越界
# 改写:
#   EmbeddingDataset.load() 含对齐检查 + 调试打印。


class EmbeddingDataset:
    """
    封装 embedding parquet 的加载与 mask 对齐。

    对应上游 df = cudf.read_parquet(...) 及后续 X/y/z 提取。
    改写: 加 mask 对齐检查，防止 df.shape[0] != data.num_nodes 时静默越界。
    """

    def __init__(self, df, X, y, z) -> None:
        self.df = df
        self.X = X   # cupy array, shape=(num_nodes, hidden_channels)
        self.y = y   # cupy array, shape=(num_nodes,)
        self.z = z   # cupy array, shape=(num_nodes,), GNN softmax 概率

    @classmethod
    def load(cls, embedding_dir: str, num_nodes: int) -> "EmbeddingDataset":
        """
        读取 parquet 并校验行数与 num_nodes 对齐。

        上游: cudf.read_parquet(dir) 合并目录下所有 parquet，无对齐检查。
        改写: 读取后校验 df.shape[0] == num_nodes，不一致时报明确错误。
        """
        import cudf

        _dbg(
            "EmbeddingDataset.load",
            f"读取 parquet: embedding_dir={embedding_dir!r} 期望行数={num_nodes}",
        )

        df = cudf.read_parquet(embedding_dir)

        _dbg(
            "EmbeddingDataset.load",
            f"parquet 读取完成: df.shape={df.shape} "
            f"columns={list(df.columns)[:5]}...",
        )

        # 对齐检查 (上游无此步骤，是已知 bug)
        if df.shape[0] != num_nodes:
            raise ValueError(
                f"[EmbeddingDataset] parquet 行数={df.shape[0]} "
                f"但 data.num_nodes={num_nodes}，两者不对齐。\n"
                f"原因: bitcoin_mnmg.py 多 rank 写出时每 rank 一个文件，"
                f"read_parquet 合并后总行数 != num_nodes。\n"
                f"修复: 检查 embedding_dir 下是否有多次实验的 parquet 混入，"
                f"或修改 bitcoin_mnmg.py 只让 rank0 写出完整 embedding。"
            )

        import cupy

        y = df["y"].to_cupy()
        z = df["z"].to_cupy()
        X = df.drop(columns=["y", "z"]).to_cupy()

        _dbg(
            "EmbeddingDataset.load",
            f"X.shape={X.shape} X.dtype={X.dtype} "
            f"y.shape={y.shape} y.dtype={y.dtype} y.unique={cupy.unique(y).tolist()} "
            f"z.shape={z.shape} z.min={float(z.min()):.4f} z.max={float(z.max()):.4f}",
        )

        return cls(df=df, X=X, y=y, z=z)

    def split(self, train_mask, test_mask):
        """
        按 train_mask / test_mask 切分 X / y / z。

        上游: X[data.train_mask] 直接索引，无 shape 校验。
        改写: 加 shape 和 class distribution 调试打印。
        """
        import cupy

        X_train = self.X[train_mask]
        X_test  = self.X[test_mask]
        y_train = self.y[train_mask].astype(cupy.int32)  # cuML RF 需要 int32
        y_test  = self.y[test_mask].astype(cupy.int32)
        z_train = self.z[train_mask]   # 上游死代码，保留但标注
        z_test  = self.z[test_mask]

        _dbg(
            "EmbeddingDataset.split",
            f"X_train.shape={X_train.shape} X_test.shape={X_test.shape} "
            f"y_train.shape={y_train.shape} y_test.shape={y_test.shape}",
        )
        _dbg(
            "EmbeddingDataset.split",
            f"y_train class dist: "
            f"0={int((y_train == 0).sum())} 1={int((y_train == 1).sum())} "
            f"2={int((y_train == 2).sum())} (2=unknown，cuML RF 用int32处理)",
        )
        _dbg(
            "EmbeddingDataset.split",
            f"z_train (死代码，上游未使用): shape={z_train.shape}  "
            f"z_test: shape={z_test.shape} mean={float(z_test.mean()):.4f}",
        )

        return X_train, X_test, y_train, y_test, z_train, z_test


# ──────────────────────────────────────────────────────────────────────────────
# RfExperiment — 封装单次 RF 训练 + 评估
# ──────────────────────────────────────────────────────────────────────────────
# 上游 (940ab01): train_rf() + show_confusion_matrix() 两个独立函数，
#   无随机种子，无 dtype 检查，结果不可复现。
# 改写:
#   RfExperiment 封装，run() 含 dtype 检查 + 随机种子 + 断点调试。


class RfExperiment:
    """
    封装单次 RF 实验: 训练 + 混淆矩阵 + AUC。

    对应上游 train_rf() + show_confusion_matrix()。
    改写:
      - random_state 参数 (上游无，CI 结果不可复现)
      - dtype 检查: X_train 强制 float32，y_train 强制 int32
      - 断点调试打印 fit/predict 前后的 shape 和类型
    """

    def __init__(self, name: str, random_state: int = 42) -> None:
        self.name = name
        self.random_state = random_state
        self.model = None

    def fit(self, X_train, y_train) -> "RfExperiment":
        """
        训练 cuML RandomForestClassifier。

        上游: RandomForestClassifier() 全默认参数，无随机种子。
        改写: 加 random_state + dtype 强制转换 + 调试打印。
        """
        import cupy
        from cuml.ensemble import RandomForestClassifier

        # dtype 对齐: cuML RF 要求 float32 特征 + int32 标签
        X_train_f = X_train.astype(cupy.float32)
        y_train_i = y_train.astype(cupy.int32)

        _dbg(
            "RfExperiment.fit",
            f"name={self.name!r} "
            f"X_train.shape={X_train_f.shape} X_train.dtype={X_train_f.dtype} "
            f"y_train.shape={y_train_i.shape} y_train.dtype={y_train_i.dtype} "
            f"class_dist: 0={int((y_train_i==0).sum())} 1={int((y_train_i==1).sum())}",
        )

        self.model = RandomForestClassifier(random_state=self.random_state)
        self.model.fit(X_train_f, y_train_i)

        _dbg("RfExperiment.fit", f"name={self.name!r} fit 完成")

        return self

    def evaluate(self, X_test, y_test) -> None:
        """
        评估并打印混淆矩阵 / 准确率 / AUC。

        对应上游 show_confusion_matrix()。
        改写: 加 predict_proba shape 断点调试 + prob 列顺序说明。
        """
        import cupy
        from cuml.metrics import confusion_matrix, roc_auc_score

        X_test_f = X_test.astype(cupy.float32)
        y_test_i = y_test.astype(cupy.int32)

        prob = self.model.predict_proba(X_test_f)

        _dbg(
            "RfExperiment.evaluate",
            f"name={self.name!r} "
            f"X_test.shape={X_test_f.shape} "
            f"prob.shape={prob.shape} "
            f"prob 列顺序: cuML 按升序 class label (col0=label0, col1=label1) "
            f"prob[:,1] 作为正类概率",
        )

        cm = confusion_matrix(y_test_i, prob.argmax(axis=1))
        acc = (y_test_i == prob.argmax(axis=1)).sum() / len(y_test_i)

        # 上游: roc_auc_score(y_test, prob[:, 1])
        # 注意: 若 y_test 含 y=2 (unknown)，roc_auc_score 二分类会出错
        # Walpurgis 保留上游行为，调试 print 使问题可见
        auc = roc_auc_score(y_test_i, prob[:, 1])

        print(
            f"=== {self.name} ===\n"
            f"Confusion Matrix:\n{cm}\n"
            f"Accuracy: {float(acc):.4f}\n"
            f"ROC AUC: {float(auc):.4f}\n"
        )

        _dbg(
            "RfExperiment.evaluate",
            f"name={self.name!r} acc={float(acc):.4f} auc={float(auc):.4f}",
        )

    def run(self, X_train, y_train, X_test, y_test) -> None:
        """fit + evaluate 一步调用。"""
        self.fit(X_train, y_train)
        self.evaluate(X_test, y_test)


# ──────────────────────────────────────────────────────────────────────────────
# gnn_only_evaluate — 对应上游 GNN Only 块
# ──────────────────────────────────────────────────────────────────────────────
# 上游: zz = cupy.stack([z_test, 1-z_test], axis=1) 作为 prob，
#       show_confusion_matrix(y_test, zz, "GNN Only")
# 注意: col0 = z_test (label=0 的概率), col1 = 1-z_test (label=1 的概率)
#       依赖 bitcoin_mnmg.py 中 z = softmax[:,0]（第0列为 label=0 概率）


def gnn_only_evaluate(y_test, z_test, name: str = "GNN Only") -> None:
    """
    仅使用 GNN softmax 概率进行评估，不训练 RF。

    上游: zz = cupy.stack([z_test, 1-z_test], axis=1)
    改写: 加断点调试打印 zz shape 和 z_test 分布。
    """
    import cupy
    from cuml.metrics import confusion_matrix, roc_auc_score

    import cupy as cp

    y_test_i = y_test.astype(cp.int32)
    zz = cp.stack([z_test, 1 - z_test], axis=1)

    _dbg(
        "gnn_only_evaluate",
        f"zz.shape={zz.shape} "
        f"z_test.mean={float(z_test.mean()):.4f} "
        f"z_test.min={float(z_test.min()):.4f} "
        f"z_test.max={float(z_test.max()):.4f} "
        f"zz[:,0]=z_test (label=0概率), zz[:,1]=1-z_test (label=1概率)",
    )

    cm = confusion_matrix(y_test_i, zz.argmax(axis=1))
    acc = (y_test_i == zz.argmax(axis=1)).sum() / len(y_test_i)
    auc = roc_auc_score(y_test_i, zz[:, 1])

    print(
        f"=== {name} ===\n"
        f"Confusion Matrix:\n{cm}\n"
        f"Accuracy: {float(acc):.4f}\n"
        f"ROC AUC: {float(auc):.4f}\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
# main — 对应上游 if __name__ == "__main__": 块
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="./data")
    parser.add_argument("--embedding_dir", type=str, default="./results")

    ns = parser.parse_args()
    args = BitcoinRfArgs.from_namespace(ns)
    args.validate()
    args.debug_dump()

    from torch_geometric.datasets import EllipticBitcoinDataset

    dataset = EllipticBitcoinDataset(root=args.dataset_root)
    data = dataset[0]
    data.x = data.x[:, :94]  # 去掉预生成图嵌入，与 bitcoin_mnmg.py 对齐
    assert dataset.num_classes == 2

    _dbg(
        "main",
        f"数据集加载完成: num_nodes={data.num_nodes} "
        f"train_mask.sum={data.train_mask.sum().item()} "
        f"test_mask.sum={data.test_mask.sum().item()} "
        f"x.shape={data.x.shape}",
    )

    # 加载 embedding，含对齐检查
    emb_ds = EmbeddingDataset.load(
        embedding_dir=args.embedding_dir,
        num_nodes=data.num_nodes,
    )

    train_mask = data.train_mask.numpy()
    test_mask  = data.test_mask.numpy()

    X_train, X_test, y_train, y_test, z_train, z_test = emb_ds.split(
        train_mask, test_mask
    )

    # ── RF + GNN 嵌入 ─────────────────────────────────────────────────────────
    _dbg("main", "开始 RF+GNN 实验")
    rf_with_gnn = RfExperiment(name="RF with GNN")
    rf_with_gnn.run(X_train, y_train, X_test, y_test)

    # ── GNN Only ──────────────────────────────────────────────────────────────
    _dbg("main", "开始 GNN Only 评估")
    gnn_only_evaluate(y_test, z_test)

    # ── RF Only (无 GNN 嵌入，仅原始特征) ────────────────────────────────────
    # 上游: cupy.asarray(data.x[mask].cuda()) — data.x 是 PyTorch tensor
    import cupy
    import torch

    X_only_train = cupy.asarray(data.x[data.train_mask].cuda())
    y_only_train = cupy.asarray(data.y[data.train_mask].cuda()).astype(cupy.int32)
    X_only_test  = cupy.asarray(data.x[data.test_mask].cuda())
    y_only_test  = cupy.asarray(data.y[data.test_mask].cuda()).astype(cupy.int32)

    _dbg(
        "main",
        f"RF Only: X_train.shape={X_only_train.shape} "
        f"y_train class dist: 0={int((y_only_train==0).sum())} "
        f"1={int((y_only_train==1).sum())}",
    )

    rf_only = RfExperiment(name="RF Only")
    rf_only.run(X_only_train, y_only_train, X_only_test, y_only_test)


if __name__ == "__main__":
    main()
