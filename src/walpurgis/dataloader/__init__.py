from .dataloader import DataLoader
# f4ca484 迁移新增：DGL DataLoader (非 dask 路径) 和 DaskDataLoader
from .dgl_dataloader import DataLoader as DGLDataLoader
from .dask_dataloader import (
    DaskDataLoader,
    create_batch_df,
    get_batch_id_series,
)
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
