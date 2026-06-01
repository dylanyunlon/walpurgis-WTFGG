"""
Walpurgis v3 Adjacency Utilities — Regularized Graph Normalization & Spectral Diagnostics
============================================================================================
Delta vs v2:
  1. Tikhonov-regularized degree inversion (D + εI)^{-1/2} replaces raw D^{-1/2}
     to avoid catastrophic division for near-isolated nodes.
  2. Scaled Laplacian: optional Lanczos-estimated λ_max when lambda_max=None,
     with fallback to 2.0 on convergence failure (upstream only had bare eigsh).
  3. transition_matrix: row-stochasticity verification — prints max deviation
     from Σ_j P_{ij} = 1 and flags rows above tolerance.
  4. AdjStats gains spectral_snapshot(): top-K eigenvalue summary of the
     Laplacian, available from pdb for quick ill-conditioning diagnosis.
  5. All functions emit structured diagnostics suitable for automated parsing.
"""
import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import torch
import time
import warnings
from contextlib import contextmanager


# ════════════════════════════════════════════════════════════════
#  Diagnostic Hub
# ════════════════════════════════════════════════════════════════

class AdjStats:
    """Centralized adjacency operation tracker with spectral diagnostics.

    Usage from debugger:
        AdjStats.report()              # timing + sanitize summary
        AdjStats.spectral_snapshot(L)  # top eigenvalues of sparse Laplacian
        AdjStats.sparsity_trend()      # density of all processed adjacencies
    """
    _timings = {}
    _sanitize = {"nan": 0, "inf": 0, "calls": 0}
    _first_calls = set()
    _density_log = []          # (func_name, N, nnz, density)
    _spectral_cache = {}       # label → eigenvalues

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
    def log_density(cls, func_name, N, nnz):
        density = nnz / (N * N) if N > 0 else 0.0
        cls._density_log.append((func_name, N, nnz, density))

    @classmethod
    def report(cls):
        print(f"\n{'═'*65}")
        print(f"  [AdjStats] Operation Summary (v3)")
        print(f"{'═'*65}")
        for name, times in cls._timings.items():
            total = sum(times)
            avg = total / len(times)
            mx = max(times)
            print(f"  {name:>28s}: {len(times):3d} calls | "
                  f"total={total:.4f}s avg={avg:.4f}s max={mx:.4f}s")
        s = cls._sanitize
        print(f"  {'sanitize':>28s}: {s['calls']:3d} calls | "
              f"nan={s['nan']} inf={s['inf']}")
        if cls._density_log:
            densities = [d[3] for d in cls._density_log]
            print(f"  {'density':>28s}: {len(densities)} matrices | "
                  f"avg={np.mean(densities):.4f} "
                  f"range=[{min(densities):.4f},{max(densities):.4f}]")
        if cls._spectral_cache:
            print(f"  {'spectral_cache':>28s}: {list(cls._spectral_cache.keys())}")
        print(f"{'═'*65}\n")

    @classmethod
    def sanitize_stats(cls):
        return cls._sanitize.copy()

    @classmethod
    def spectral_snapshot(cls, L_sparse, label="laplacian", k=6):
        """Compute & cache top-k eigenvalues of a sparse matrix.
        Callable from pdb for quick spectral diagnosis."""
        try:
            k_actual = min(k, L_sparse.shape[0] - 2)
            if k_actual < 1:
                print(f"  [spectral] matrix too small for eigendecomposition")
                return None
            eigvals, _ = linalg.eigsh(L_sparse, k=k_actual, which="LM")
            eigvals = np.sort(eigvals)[::-1]
            cls._spectral_cache[label] = eigvals
            print(f"  [spectral:{label}] top-{k_actual} eigenvalues: "
                  f"{np.array2string(eigvals, precision=4, separator=', ')}")
            ratio = eigvals[0] / max(eigvals[-1], 1e-12)
            print(f"  [spectral:{label}] spectral condition ≈ {ratio:.1f}")
            return eigvals
        except Exception as e:
            print(f"  [spectral:{label}] eigsh failed: {e}")
            return None

    @classmethod
    def sparsity_trend(cls):
        """Print density of every adjacency processed so far."""
        if not cls._density_log:
            print("  [sparsity_trend] no matrices processed yet")
            return
        print(f"\n  [sparsity_trend] {len(cls._density_log)} matrices:")
        for func, N, nnz, dens in cls._density_log:
            bar_len = int(dens * 40)
            bar = '█' * bar_len + '░' * (40 - bar_len)
            print(f"    {func:>20s} N={N:4d} nnz={nnz:6d} [{bar}] {dens:.4f}")


@contextmanager
def _timed(func_name):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    AdjStats.record(func_name, elapsed)
    if elapsed > 1.0:
        print(f"  [perf_warn] {func_name} took {elapsed:.3f}s (>1s threshold)")


# ════════════════════════════════════════════════════════════════
#  Tensor Sanitizers
# ════════════════════════════════════════════════════════════════

# -- v3: Tikhonov regularisation constant for degree inversion --
_TIKHONOV_EPS = 1e-6


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
        raise ValueError(
            f"Tensor contamination: nan={n_nan} inf={n_inf} "
            f"shape={list(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device}"
        )
    return diag, (n_nan > 0 or n_inf > 0)


def remove_nan_inf(tensor):
    stats = AdjStats._sanitize
    stats["calls"] += 1
    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()
    stats["nan"] += n_nan
    stats["inf"] += n_inf
    if (n_nan + n_inf > 0) and stats["calls"] <= 10:
        print(f"[sanitize #{stats['calls']}] nan={n_nan} inf={n_inf} "
              f"shape={list(tensor.shape)} device={tensor.device}")
    # v3: use nan_to_num for single-pass replacement (fused, faster)
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor


def _degree_condition(degrees):
    """Condition proxy: max/min nonzero degree. High values → ill-conditioned."""
    nz = degrees[degrees > _TIKHONOV_EPS]
    if len(nz) == 0:
        return float("inf")
    return float(nz.max() / nz.min())


def _regularized_inv_sqrt(degrees, eps=_TIKHONOV_EPS):
    """Tikhonov-regularized D^{-1/2}: (d_i + ε)^{-1/2} instead of d_i^{-1/2}.

    This avoids catastrophic values for near-isolated nodes (d_i ≈ 0)
    without zeroing them out entirely (which silently disconnects them
    from message passing).
    """
    d_reg = degrees + eps
    d_inv_sqrt = np.power(d_reg, -0.5)
    # still clamp any residual inf (shouldn't happen with eps, but be safe)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    n_tiny = int(np.sum(degrees < eps))
    if n_tiny > 0:
        print(f"  [tikhonov] {n_tiny} nodes with degree < ε={eps:.1e}, "
              f"regularized instead of zeroed")
    return d_inv_sqrt


# ════════════════════════════════════════════════════════════════
#  Graph Normalization Functions
# ════════════════════════════════════════════════════════════════

def calculate_symmetric_normalized_laplacian(adj):
    """Compute L_sym = I - D_reg^{-1/2} A D_reg^{-1/2}.

    v3 delta: uses Tikhonov-regularized degree inversion.
    """
    with _timed("sym_norm_lap"):
        adj = sp.coo_matrix(adj)
        N = adj.shape[0]
        degrees = np.array(adj.sum(1)).flatten()

        # v3: regularized inversion
        d_inv_sqrt = _regularized_inv_sqrt(degrees)
        D_half = sp.diags(d_inv_sqrt)
        L_sym = sp.eye(N) - D_half.dot(adj).dot(D_half).tocoo()

        cond = _degree_condition(degrees)
        n_iso = int(np.sum(degrees == 0))
        deg_stats = (f"deg∈[{degrees.min():.2f},{degrees.max():.2f}] "
                     f"μ={degrees.mean():.2f} σ={degrees.std():.2f}")
        print(f"[sym_norm_lap] N={N} nnz={L_sym.nnz} isolated={n_iso} "
              f"cond={cond:.1f} {deg_stats}")
        AdjStats.log_density("sym_norm_lap", N, L_sym.nnz)
    return L_sym


def calculate_scaled_laplacian(adj, lambda_max=2, undirected=True):
    """Rescale Laplacian eigenvalues to [-1, 1] for Chebyshev polynomials.

    v3 delta: when lambda_max is None, uses Lanczos iteration (eigsh)
    with explicit fallback + warning on convergence failure, instead of
    silently propagating ARPACK exceptions.
    """
    with _timed("scaled_lap"):
        if undirected:
            adj = np.maximum.reduce([adj, adj.T])
        L = calculate_symmetric_normalized_laplacian(adj)

        if lambda_max is None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    eig_vals, _ = linalg.eigsh(L, 1, which="LM",
                                               maxiter=300, tol=1e-4)
                    lambda_max = float(eig_vals[0])
                    print(f"  [lanczos] converged: λ_max={lambda_max:.6f}")
            except linalg.ArpackNoConvergence as e:
                lambda_max = 2.0
                partial = getattr(e, 'eigenvalues', None)
                print(f"  [lanczos] ⚠ ARPACK did not converge, "
                      f"falling back to λ_max=2.0 "
                      f"(partial={partial})")

        L = sp.csr_matrix(L)
        M = L.shape[0]
        I = sp.identity(M, format="csr", dtype=L.dtype)
        L_scaled = (2.0 / lambda_max) * L - I

        # v3: validate spectral range of scaled laplacian
        diag_vals = L_scaled.diagonal()
        print(f"[scaled_lap] N={M} λ_max={lambda_max:.4f} nnz={L_scaled.nnz} "
              f"diag∈[{diag_vals.min():.4f},{diag_vals.max():.4f}]")
        AdjStats.log_density("scaled_lap", M, L_scaled.nnz)
    return L_scaled


def symmetric_message_passing_adj(adj):
    """Renormalized message passing adjacency (GCN-style).

    v3 delta: uses Tikhonov-regularized D^{-1/2}.
    """
    with _timed("sym_mp_adj"):
        adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1)).flatten()
        # v3: regularized inversion instead of raw power
        d_inv_sqrt = _regularized_inv_sqrt(row_sum)
        D_half = sp.diags(d_inv_sqrt)
        mp_adj = D_half.dot(adj).transpose().dot(D_half).astype(np.float32).todense()

        # v3: symmetry check — mp_adj should be symmetric for undirected graphs
        asym = np.abs(mp_adj - mp_adj.T).max()
        sym_tag = "symmetric" if asym < 1e-6 else f"asymmetric(Δ={asym:.2e})"
        print(f"[sym_mp_adj] {mp_adj.shape} ∈[{mp_adj.min():.4f},{mp_adj.max():.4f}] "
              f"{sym_tag}")
        AdjStats.log_density("sym_mp_adj", adj.shape[0], adj.nnz)
    return mp_adj


def transition_matrix(adj):
    """Row-stochastic transition matrix P = D^{-1}A.

    v3 delta: validates row-stochasticity after construction —
    prints max deviation from row-sum=1 and flags non-stochastic rows.
    """
    with _timed("transition_mx"):
        adj = sp.coo_matrix(adj)
        N = adj.shape[0]
        row_sum = np.array(adj.sum(1)).flatten()
        d_inv = np.power(row_sum, -1.0)
        d_inv[np.isinf(d_inv)] = 0.0
        D_inv = sp.diags(d_inv)
        P = D_inv.dot(adj).astype(np.float32).todense()

        # v3: row-stochasticity verification
        n_iso = int(np.sum(row_sum == 0))
        row_sums_P = np.array(P.sum(axis=1)).flatten()
        non_iso_mask = row_sum > 0
        if non_iso_mask.any():
            deviations = np.abs(row_sums_P[non_iso_mask] - 1.0)
            max_dev = deviations.max()
            n_bad = int(np.sum(deviations > 1e-5))
            stoch_tag = "✓ stochastic" if max_dev < 1e-5 else f"⚠ max_dev={max_dev:.2e}"
            print(f"[transition_mx] N={N} edges={adj.nnz} isolated={n_iso} "
                  f"{stoch_tag}")
            if n_bad > 0:
                print(f"  ⚠ {n_bad} non-isolated rows deviate from "
                      f"Σ=1 by >{1e-5:.0e}")
        else:
            print(f"[transition_mx] N={N} — all nodes isolated!")

        AdjStats.log_density("transition_mx", N, adj.nnz)
    return P
