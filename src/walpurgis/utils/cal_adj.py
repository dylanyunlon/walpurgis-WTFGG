"""
Walpurgis v2 Adjacency Utilities — Graph Normalization with Condition Monitoring
==================================================================================
Delta: adds condition-number proxy (ratio of max/min nonzero degree) to
Laplacian computations, which flags ill-conditioned graphs early.
"""
import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch
import time
from contextlib import contextmanager


class AdjStats:
    """Centralized adjacency operation tracker. Call .report() from pdb."""
    _timings = {}
    _sanitize = {"nan": 0, "inf": 0, "calls": 0}
    _first_calls = set()

    @classmethod
    def record(cls, func_name, elapsed):
        cls._timings.setdefault(func_name, []).append(elapsed)

    @classmethod
    def is_first(cls, func_name):
        if func_name not in cls._first_calls:
            cls._first_calls.add(func_name)
            return True
        return False

    @classmethod
    def report(cls):
        print(f"\n{'═'*60}")
        print(f"  [AdjStats] Operation Summary")
        print(f"{'═'*60}")
        for name, times in cls._timings.items():
            total = sum(times)
            avg = total / len(times)
            print(f"  {name:>24s}: {len(times):3d} calls | total={total:.4f}s avg={avg:.4f}s")
        s = cls._sanitize
        print(f"  {'sanitize':>24s}: {s['calls']:3d} calls | nan={s['nan']} inf={s['inf']}")
        print(f"{'═'*60}\n")

    @classmethod
    def sanitize_stats(cls):
        return cls._sanitize.copy()


@contextmanager
def _timed(func_name):
    t0 = time.perf_counter()
    yield
    AdjStats.record(func_name, time.perf_counter() - t0)


def check_nan_inf(tensor, raise_ex=True):
    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()
    total = tensor.numel()
    diag = {
        "has_nan": n_nan > 0, "has_inf": n_inf > 0,
        "nan_count": n_nan, "inf_count": n_inf,
        "total_elements": total,
        "contamination_pct": (n_nan + n_inf) / max(total, 1) * 100,
    }
    if (n_nan or n_inf) and raise_ex:
        raise ValueError(f"Tensor contamination: {diag}")
    return diag, (n_nan > 0 or n_inf > 0)


def remove_nan_inf(tensor):
    stats = AdjStats._sanitize
    stats["calls"] += 1
    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()
    stats["nan"] += n_nan
    stats["inf"] += n_inf
    if (n_nan + n_inf > 0) and stats["calls"] <= 10:
        print(f"[sanitize #{stats['calls']}] nan={n_nan} inf={n_inf} shape={list(tensor.shape)}")
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
    return tensor


def _degree_condition(degrees):
    """Condition proxy: max/min nonzero degree. High values → ill-conditioned."""
    nz = degrees[degrees > 0]
    if len(nz) == 0:
        return float("inf")
    return float(nz.max() / nz.min())


def calculate_symmetric_normalized_laplacian(adj):
    with _timed("sym_norm_lap"):
        adj = sp.coo_matrix(adj)
        N = adj.shape[0]
        degrees = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(degrees, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_half = sp.diags(d_inv_sqrt)
        L_sym = sp.eye(N) - D_half.dot(adj).dot(D_half).tocoo()
        cond = _degree_condition(degrees)
        n_iso = int(np.sum(degrees == 0))
        print(f"[sym_norm_lap] N={N} nnz={L_sym.nnz} isolated={n_iso} cond={cond:.1f}")
    return L_sym


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    with _timed("scaled_lap"):
        if undirected:
            adj = np.maximum.reduce([adj, adj.T])
        L = calculate_symmetric_normalized_laplacian(adj)
        if lambda_max is None:
            lambda_max, _ = linalg.eigsh(L, 1, which="LM")
            lambda_max = float(lambda_max[0])
        L = sp.csr_matrix(L)
        M = L.shape[0]
        I = sp.identity(M, format="csr", dtype=L.dtype)
        L_scaled = (2.0 / lambda_max) * L - I
        print(f"[scaled_lap] N={M} λ_max={lambda_max} nnz={L_scaled.nnz}")
    return L_scaled


def symmetric_message_passing_adj(adj):
    with _timed("sym_mp_adj"):
        adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(row_sum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_half = sp.diags(d_inv_sqrt)
        mp_adj = D_half.dot(adj).transpose().dot(D_half).astype(np.float32).todense()
        print(f"[sym_mp_adj] {mp_adj.shape} ∈[{mp_adj.min():.4f},{mp_adj.max():.4f}]")
    return mp_adj


def transition_matrix(adj):
    with _timed("transition_mx"):
        adj = sp.coo_matrix(adj)
        N = adj.shape[0]
        row_sum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(row_sum, -1.0)
        d_inv[np.isinf(d_inv)] = 0.0
        D_inv = sp.diags(d_inv)
        P = D_inv.dot(adj).astype(np.float32).todense()
        n_iso = int(np.sum(row_sum == 0))
        if n_iso:
            print(f"[transition_mx] ⚠ {n_iso}/{N} isolated nodes")
        print(f"[transition_mx] N={N} edges={adj.nnz}")
    return P
