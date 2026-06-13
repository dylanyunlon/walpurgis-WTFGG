from .dataloader import DataLoader

# ── 以下为cugraph-gnn迁移模块, 需要cupy/dgl环境 ──
# 训练pipeline (train_walpurgis.py) 只需要上面的DataLoader
# GPU环境下这些模块自动可用

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
