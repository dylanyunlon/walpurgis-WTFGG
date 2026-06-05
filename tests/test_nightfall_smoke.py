"""
test_nightfall_smoke.py — walpurgis_nightfall 冒烟测试
第二十位Claude (Opus 4.6) 编写

运行: cd walpurgis-WTFGG && PYTHONPATH=src python -m pytest tests/test_nightfall_smoke.py -v
"""
import os
import sys
import pytest
import torch

# 确保src在path上
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestNightfallImports:
    """测试所有模块可import"""

    def test_init(self):
        import walpurgis_nightfall
        from walpurgis_nightfall import _dbg, _is_debug, snapshot_model
        assert callable(_dbg)
        assert callable(_is_debug)
        assert callable(snapshot_model)

    def test_utils(self):
        from walpurgis_nightfall.utils.cal_adj import remove_nan_inf
        from walpurgis_nightfall.utils.train import EarlyStopping, data_reshaper
        from walpurgis_nightfall.utils.log import TrainLogger
        from walpurgis_nightfall.utils.load_data import load_dataset, load_adj

    def test_dataloader(self):
        from walpurgis_nightfall.dataloader import DataLoader

    def test_losses(self):
        from walpurgis_nightfall.models.losses import masked_mae, masked_rmse, masked_mape, metric

    def test_decouple(self):
        from walpurgis_nightfall.models.decouple.estimation_gate import EstimationGate
        from walpurgis_nightfall.models.decouple.residual_decomp import ResidualDecomp

    def test_diffusion_block(self):
        from walpurgis_nightfall.models.diffusion_block.dif_model import STLocalizedConv
        from walpurgis_nightfall.models.diffusion_block.dif_block import DifBlock
        from walpurgis_nightfall.models.diffusion_block.forecast import Forecast

    def test_dynamic_graph_conv(self):
        from walpurgis_nightfall.models.dynamic_graph_conv.utils.distance import DistanceFunction
        from walpurgis_nightfall.models.dynamic_graph_conv.utils.mask import Mask
        from walpurgis_nightfall.models.dynamic_graph_conv.utils.normalizer import Normalizer, MultiOrder
        from walpurgis_nightfall.models.dynamic_graph_conv.dy_graph_conv import DynamicGraphConstructor

    def test_inherent_block(self):
        from walpurgis_nightfall.models.inherent_block.inh_model import RNNLayer, TransformerLayer
        from walpurgis_nightfall.models.inherent_block.forecast import Forecast
        from walpurgis_nightfall.models.inherent_block.inh_block import InhBlock

    def test_model(self):
        from walpurgis_nightfall.models.model import D2STGNN

    def test_trainer(self):
        from walpurgis_nightfall.models.trainer import trainer


class TestNightfallForwardBackward:
    """测试前向+反向传播"""

    @pytest.fixture
    def model_and_input(self):
        os.environ['NIGHTFALL_DEBUG'] = '0'
        from walpurgis_nightfall.models.model import D2STGNN
        num_nodes, num_feat, num_hidden, T, batch, gap = 10, 3, 16, 12, 2, 3
        adj = torch.randn(num_nodes, num_nodes).abs()
        adj = (adj + adj.T) / 2
        model_args = {
            'num_feat': num_feat, 'num_hidden': num_hidden, 'node_hidden': 8,
            'time_emb_dim': 8, 'seq_length': T, 'num_nodes': num_nodes,
            'k_s': 2, 'k_t': 3, 'gap': gap, 'num_layers': 1,
            'cl_decay_steps': 2000, 'dropout': 0.1,
            'supports': [adj, adj], 'adjs': [adj, adj], 'adjs_ori': adj,
            'dataset': 'SMOKE', 'device': torch.device('cpu'), 'batch_size': batch,
        }
        model = D2STGNN(**model_args)
        x_raw = torch.randn(batch, T, num_nodes, num_feat)
        time_in_day = torch.rand(batch, T, num_nodes, 1)
        day_in_week = torch.randint(0, 7, (batch, T, num_nodes, 1)).float()
        x = torch.cat([x_raw, time_in_day, day_in_week], dim=-1)
        return model, x, batch, num_nodes

    def test_forward(self, model_and_input):
        model, x, batch, num_nodes = model_and_input
        output = model(x)
        assert output.shape[0] == batch
        assert num_nodes in output.shape
        assert not torch.isnan(output).any(), "NaN in output"
        assert not torch.isinf(output).any(), "Inf in output"

    def test_backward(self, model_and_input):
        model, x, batch, num_nodes = model_and_input
        output = model(x)
        loss = output.sum()
        loss.backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0, "No gradients computed"

    def test_param_count(self, model_and_input):
        model, _, _, _ = model_and_input
        total = sum(p.numel() for p in model.parameters())
        assert total > 10000, f"Suspiciously few params: {total}"


class TestNightfallDebug:
    """测试NIGHTFALL_DEBUG调试输出"""

    def test_debug_flag(self):
        os.environ['NIGHTFALL_DEBUG'] = '1'
        # 需要重新import以读取环境变量
        import importlib
        import walpurgis_nightfall
        importlib.reload(walpurgis_nightfall)
        assert walpurgis_nightfall._is_debug()
        os.environ['NIGHTFALL_DEBUG'] = '0'
        importlib.reload(walpurgis_nightfall)
        assert not walpurgis_nightfall._is_debug()

    def test_snapshot_model(self):
        os.environ['NIGHTFALL_DEBUG'] = '1'
        import importlib, walpurgis_nightfall
        importlib.reload(walpurgis_nightfall)
        model = torch.nn.Linear(10, 5)
        # DEBUG=1时返回state_dict snapshot
        snap = walpurgis_nightfall.snapshot_model(model)
        # snapshot_model打印到stderr,不一定返回dict
        # 主要验证不crash
        os.environ['NIGHTFALL_DEBUG'] = '0'
        importlib.reload(walpurgis_nightfall)
