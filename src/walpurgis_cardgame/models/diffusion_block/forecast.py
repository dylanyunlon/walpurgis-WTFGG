"""
forecast.py (diffusion) — CardGame Diffusion Forecast
算法改写 (vs upstream):
  - 新增spectral dropout: 在频域对隐状态做dropout
  - forecast_fc后加LayerNorm稳定输出
"""
import os
import sys
import torch
import torch.nn as nn

_CG_DEBUG = os.environ.get('CARDGAME_DEBUG', '0') == '1'

def _dbg(tag, tensor, module="DifForecast"):
    if not _CG_DEBUG: return
    if hasattr(tensor, 'shape'):
        msg = (f"[CG-DBG:{tag}@{module}] shape={list(tensor.shape)} dtype={tensor.dtype} "
               f"min={tensor.min().item():.6f} max={tensor.max().item():.6f} "
               f"mean={tensor.mean().item():.6f} std={tensor.std().item():.6f}")
        nan_count = tensor.isnan().sum().item()
        inf_count = tensor.isinf().sum().item()
        if nan_count > 0: msg += f" *** NaN={nan_count} ***"
        if inf_count > 0: msg += f" *** Inf={inf_count} ***"
    else:
        msg = f"[CG-DBG:{tag}@{module}] value={tensor}"
    print(msg, file=sys.stderr)


class SpectralDropout(nn.Module):
    """频域Dropout: 在FFT域随机丢弃频率分量
    对时间序列的频域正则化, 防止模型过度依赖特定频率
    """

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # 沿最后一维做FFT
        freq = torch.fft.rfft(x, dim=-1)
        # 频域dropout mask
        mask = torch.ones_like(freq.real)
        mask = torch.bernoulli(mask * (1 - self.p))
        mask = mask / (1 - self.p)  # scale
        freq = freq * mask
        # 逆FFT
        out = torch.fft.irfft(freq, n=x.shape[-1], dim=-1)
        return out


class Forecast(nn.Module):
    """CardGame Diffusion Forecast with spectral dropout"""

    def __init__(self, hidden_dim, forecast_hidden_dim=None, **model_args):
        super().__init__()
        self.k_t = model_args['k_t']
        self.output_seq_len = model_args['seq_length']
        self.forecast_fc = nn.Linear(hidden_dim, forecast_hidden_dim)
        self.model_args = model_args

        # CardGame新增: spectral dropout + LayerNorm
        self.spectral_drop = SpectralDropout(p=model_args.get('dropout', 0.1))
        self.ln_out = nn.LayerNorm(forecast_hidden_dim)

    def forward(self, gated_history_data, hidden_states_dif,
                localized_st_conv, dynamic_graph, static_graph):
        _dbg("input.gated", gated_history_data)
        _dbg("input.hidden_dif", hidden_states_dif)

        predict = []
        history = gated_history_data
        predict.append(hidden_states_dif[:, -1, :, :].unsqueeze(1))

        for _ in range(int(self.output_seq_len / self.model_args['gap']) - 1):
            _1 = predict[-self.k_t:]
            if len(_1) < self.k_t:
                sub = self.k_t - len(_1)
                _2 = history[:, -sub:, :, :]
                _1 = torch.cat([_2] + _1, dim=1)
            else:
                _1 = torch.cat(_1, dim=1)
            predict.append(localized_st_conv(
                _1, dynamic_graph, static_graph))

        predict = torch.cat(predict, dim=1)

        # CardGame: spectral dropout在FC之前
        predict = self.spectral_drop(predict)
        _dbg("after_spectral_drop", predict)

        predict = self.forecast_fc(predict)
        predict = self.ln_out(predict)
        _dbg("forecast_output", predict)
        return predict
