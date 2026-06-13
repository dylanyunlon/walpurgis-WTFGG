from .dataloader import DataLoader
# f4ca484 迁移新增：DGL DataLoader (非 dask 路径)
# 61a370e REMOVED: DaskDataLoader — cuGraph-DGL Dask API 已删除
#   上游 PR #199 Remove Dask API from cuGraph-DGL 删除了:
#     - cugraph_dgl/dataloading/dask_dataloader.py (321行)
#     - dataloading/__init__.py 中的 DaskDataLoader 导出
#   walpurgis 对应: dask_dataloader.py 降级为墓碑模块，不再从 __init__ 导出
from .dgl_dataloader import DataLoader as DGLDataLoader
# DaskDataLoader 已在 61a370e 中随 Dask API 一并移除
# 若需迁移路径请参见 src/walpurgis/dataloader/dask_dataloader.py 中的迁移指引
from .edge_input_id import (
    EdgeInputIdError,
    validate_edges_shape,
    make_edge_input_id,
    assert_input_id_consistent,
    resolve_edge_input_id,
)
from .hetero_link_pred_fixes import (
    InputTypeResolver,
    NegativeSeedFilter,
    EdgeLabelBuilder,
    FanoutConverter,
    NegSampleLocalizer,
    resolve_input_type,
    filter_negative_seeds,
    build_edge_label,
    convert_fanout,
    local_neg_count,
)
from .node_classification import (
    NodeClassificationDataset,
    NodeClassificationData,
    PickleLoader,
    DatasetSplitValidator,
    create_node_classification_datasets,
    create_node_claffication_datasets,  # compat alias (typo)
    load_node_classification_datasets_from_pickle,
)
from .loader_deprecation import (
    DaskNeighborLoader,
    BulkSampleLoader,
    CuGraphNeighborLoader,
    loader_deprecation_registry,
)
# f57ed88 迁移新增: LinkLoader (边批次采样基类) 和 LinkNeighborLoader (GraphSAGE 风格边采样)
from .link_loader import (
    LinkLoader,
    NegSamplingSpec,
    EdgeSamplerSpec,
)
from .link_neighbor_loader import (
    LinkNeighborLoader,
    CompressionMode,
    SubgraphGuard,
    SamplerBuildSpec,
)
