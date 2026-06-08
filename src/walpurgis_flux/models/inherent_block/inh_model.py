"""Flux inherent model: StreamGRU + 分块因果注意力(Chunked Causal Attention).
StreamGRU: GRU with forget bias — 初始forget gate偏置为正,
  使得模型初期更倾向保留历史信息, 适合流式推理场景.
分块因果注意力: 将序列分块, 块内做因果self-attention,
  块间不交互, 适合流式推理的O(chunk_size^2)复杂度.
与upstream(标准GRU + vanilla MHA)和vortex(MinGRU + 相对位置偏置)不同."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

_FX_DBG = os.environ.get('FLUX_DEBUG', '0') == '1'


class RNNLayer(nn.Module):
    """Flux RNN: GRU with forget bias for streaming."""
    def __init__(self, hidden_dim, dropout=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Flux: forget bias — 初始化bias使forget gate偏高
        # 让模型初期更保守地保留历史信息
        with torch.no_grad():
            # GRU的bias_ih和bias_hh的前hidden_dim维是reset gate
            # 中间hidden_dim维是update(forget) gate
            if hasattr(self.gru_cell, 'bias_ih'):
                self.gru_cell.bias_ih.data[
                    hidden_dim:2*hidden_dim].fill_(1.5)
            if hasattr(self.gru_cell, 'bias_hh'):
                self.gru_cell.bias_hh.data[
                    hidden_dim:2*hidden_dim].fill_(1.5)

    def forward(self, X):
        [batch_size, seq_len, num_nodes,
         hidden_dim] = X.shape
        X = X.transpose(1, 2).reshape(
            batch_size * num_nodes, seq_len, hidden_dim)
        hx = torch.zeros_like(X[:, 0, :])
        output = []
        for _ in range(X.shape[1]):
            hx = self.gru_cell(X[:, _, :], hx)
            output.append(hx)
        output = torch.stack(output, dim=0)
        output = self.dropout(output)
        if _FX_DBG:
            print(f"[FX:stream_gru@inh_model] "
                  f"h_mean={hx.mean().item():.4f} "
                  f"h_std={hx.std().item():.4f} "
                  f"seq_len={seq_len}",
                  file=sys.stderr)
        return output


class TransformerLayer(nn.Module):
    """Flux Transformer: 分块因果注意力(Chunked Causal Attention).
    将输入序列分成固定大小的chunk, 每个chunk内做因果self-attention,
    chunk间不交互. 复杂度从O(L^2)降到O(L*C), C=chunk_size.
    适合流式推理: 每个chunk可以独立处理."""
    def __init__(self, hidden_dim, num_heads=4,
                 dropout=None, bias=True):
        super().__init__()
        self.multi_head_self_attention = \
            nn.MultiheadAttention(
                hidden_dim, num_heads,
                dropout=dropout, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # Flux: chunk大小 — 控制因果注意力的局部范围
        self.chunk_size = 4
        self.hidden_dim = hidden_dim

    def _causal_mask(self, q_len, kv_len, device):
        """生成因果掩码: Q只能看K中位置<=自己的
        shape: (q_len, kv_len)"""
        # 每个Q位置只能attend到K中
        # index <= (kv_len - q_len + q_pos) 的位置
        mask = torch.zeros(
            q_len, kv_len, device=device)
        for i in range(q_len):
            # Q的第i个位置对应K的最后q_len-i个之前
            allowed_end = kv_len - q_len + i + 1
            if allowed_end < kv_len:
                mask[i, allowed_end:] = float('-inf')
        return mask

    def forward(self, X, K, V):
        S_q, BN, D = X.shape
        S_kv = K.shape[0]
        # Flux: 分块因果注意力
        if S_q == S_kv and S_q <= self.chunk_size:
            # 自注意力 + 短序列: 标准因果mask
            causal_mask = self._causal_mask(
                S_q, S_kv, X.device)
            hidden_states_MSA = \
                self.multi_head_self_attention(
                    X, K, V, attn_mask=causal_mask)[0]
        elif S_q != S_kv:
            # 交叉注意力(forecast中Q=1, K/V=window)
            # 不需要因果mask — Q只有1步, 可以看所有K/V
            hidden_states_MSA = \
                self.multi_head_self_attention(
                    X, K, V)[0]
        else:
            # 长序列分块处理
            outputs = []
            n_chunks = (S_q + self.chunk_size - 1) // \
                self.chunk_size
            for c in range(n_chunks):
                start = c * self.chunk_size
                end = min(start + self.chunk_size, S_q)
                chunk_len = end - start
                x_chunk = X[start:end]
                k_chunk = K[start:end]
                v_chunk = V[start:end]
                causal_mask = self._causal_mask(
                    chunk_len, chunk_len, X.device)
                chunk_out = \
                    self.multi_head_self_attention(
                        x_chunk, k_chunk, v_chunk,
                        attn_mask=causal_mask)[0]
                outputs.append(chunk_out)
            hidden_states_MSA = torch.cat(
                outputs, dim=0)
        hidden_states_MSA = self.dropout(
            hidden_states_MSA)
        if _FX_DBG:
            print(f"[FX:chunked_attn@inh_model] "
                  f"seq={S} chunks="
                  f"{(S+self.chunk_size-1)//self.chunk_size}"
                  f" chunk_size={self.chunk_size}",
                  file=sys.stderr)
        return hidden_states_MSA
