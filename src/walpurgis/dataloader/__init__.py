from .dataloader import DataLoader

# ── 以下为cugraph-gnn迁移模块, 需要cupy/dgl环境 ──
# 训练pipeline (train_walpurgis.py) 只需要上面的DataLoader
# GPU环境下这些模块自动可用

# ── migrate 1b2fce2: fix bad import — DataloaderAliasGuard ──────────────────
# 上游 1b2fce2 在 test_dask_dataloader_mg.py 中将:
#   cugraph_dgl.dataloading.DaskDataLoader(...)
# 改为:
#   cugraph_dgl.dataloading.DataLoader(...)
#
# 根因: DaskDataLoader 已从包级别 __init__ 中移除（456d5a2/1e91ed7 废弃），
# 测试代码应引用当前有效的 DataLoader 名称。
#
# Walpurgis 20% 改写（鲁迅拿法）：
# 不只是静默更改，而是在 __init__ 留下 DataloaderAliasGuard 记录
# 哪些别名是历史遗留、哪些是当前有效入口——防止未来的开发者再次犯同样的错误。
import os as _os
import sys as _sys


class DataloaderAliasGuard:
    """DataLoader 别名守卫（migrate 1b2fce2 Walpurgis 改写）。

    记录 walpurgis.dataloader 包中有效的 DataLoader 入口与历史废弃别名，
    防止误用废弃名称（对应上游 1b2fce2 修正的问题：测试代码误用 DaskDataLoader）。
    """

    # 当前有效的 DataLoader 入口名称
    ACTIVE_ALIASES: tuple = ("DataLoader",)

    # 历史废弃别名（不再从包级别导出）
    DEPRECATED_ALIASES: tuple = (
        "DaskDataLoader",    # 1b2fce2: 上游测试中误用，应改为 DataLoader
        "BulkDataLoader",    # 历史命名，从未迁移
        "CuDFDataLoader",    # Dask/cuDF 路径，随 Dask API 删除
    )

    @classmethod
    def is_active(cls, name: str) -> bool:
        """检查 name 是否为当前有效的 DataLoader 入口。"""
        return name in cls.ACTIVE_ALIASES

    @classmethod
    def is_deprecated(cls, name: str) -> bool:
        """检查 name 是否为已废弃的别名（对应 1b2fce2 的问题入口）。"""
        return name in cls.DEPRECATED_ALIASES

    @classmethod
    def warn_if_deprecated(cls, name: str) -> None:
        """如果 name 是废弃别名，输出警告（WALPURGIS_DEBUG=1 时到 stderr）。"""
        if cls.is_deprecated(name):
            _debug = _os.environ.get("WALPURGIS_DEBUG", "0").strip() == "1"
            if _debug:
                print(
                    f"[WALPURGIS-DATALOADER-INIT] 检测到废弃 DataLoader 别名: {name!r}。"
                    f"请使用 DataLoader（migrate 1b2fce2）。",
                    file=_sys.stderr,
                    flush=True,
                )


try:
    from .dgl_dataloader import DataLoader as DGLDataLoader
except ImportError:
    pass

try:
    from .edge_input_id import (
        EdgeInputIdError, validate_edges_shape,
        make_edge_input_id, assert_input_id_consistent,
        resolve_edge_input_id,
    )
except ImportError:
    pass

try:
    from .hetero_link_pred_fixes import (
        InputTypeResolver, NegativeSeedFilter,
        EdgeLabelBuilder, FanoutConverter,
        NegSampleLocalizer,
    )
except ImportError:
    pass

try:
    from .node_classification import (
        NodeClassificationDataset, NodeClassificationData,
    )
except ImportError:
    pass

try:
    from .loader_deprecation import (
        DaskNeighborLoader, BulkSampleLoader,
        CuGraphNeighborLoader,
    )
except ImportError:
    pass

try:
    from .link_loader import LinkLoader, NegSamplingSpec, EdgeSamplerSpec
    from .link_neighbor_loader import LinkNeighborLoader
except ImportError:
    pass
