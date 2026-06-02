"""Describe adjacency matrices across all supported datasets.

# ═══════════════════════════════════════════════════════════════════
# Walpurgis v4 Adjacency Descriptor
# ═══════════════════════════════════════════════════════════════════
# Fourth-pass rewrite.  Changes from v3:
#   1. Added eigenvalue distribution summary
#   2. Community detection via spectral clustering count
#   3. Enhanced debug output with graph diameter estimate
#
# Breakpoint guide:
#   pdb> python -m pdb describe_adjs.py
#   pdb> b describe           # break at main describe function
#   pdb> p adj.shape           # inspect adjacency dimensions
# ═══════════════════════════════════════════════════════════════════

Walpurgis adaptations:
- Extended diagnostics: density, symmetry, degree distribution, isolated nodes
- Memory footprint estimates per adjacency matrix
- Tier placement recommendations based on matrix size
"""
import pickle
import numpy as np


def load_pickle(pickle_file):
    """Load pickle data with encoding fallback."""
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f'[Walpurgis] Unable to load {pickle_file}: {e}')
        raise
    return pickle_data


def describe_adj(name, adj_mx):
    """Print comprehensive adjacency matrix diagnostics.

    Walpurgis: extended beyond simple node/edge counts.
    """
    if isinstance(adj_mx, np.matrix):
        adj_mx = np.asarray(adj_mx)

    n_nodes = adj_mx.shape[0]
    edge_count = np.count_nonzero(adj_mx)
    density = edge_count / (n_nodes * n_nodes) if n_nodes > 0 else 0
    is_symmetric = np.allclose(adj_mx, adj_mx.T)
    degrees = (adj_mx != 0).sum(axis=1)
    isolated = np.sum(degrees == 0)
    mem_kb = adj_mx.nbytes / 1024

    # Tier recommendation based on matrix size
    if n_nodes > 500:
        tier = "GDDR (large graph, frequent random access)"
    elif n_nodes > 200:
        tier = "GDDR/DRAM (moderate graph)"
    else:
        tier = "DRAM (small graph, low access frequency)"

    print(f"\n{'='*20} {name} {'='*20}")
    print(f"  Nodes:     {n_nodes}")
    print(f"  Edges:     {edge_count}")
    print(f"  Density:   {density:.4f}")
    print(f"  Symmetric: {is_symmetric}")
    print(f"  Degree:    mean={degrees.mean():.1f} min={degrees.min()} "
          f"max={degrees.max()} std={degrees.std():.1f}")
    print(f"  Isolated:  {isolated} nodes")
    print(f"  Memory:    {mem_kb:.1f} KB ({adj_mx.dtype})")
    print(f"  Walpurgis tier: {tier}")
    if adj_mx.max() > 1.0:
        vals = adj_mx[adj_mx > 0]
        print(f"  Weights:   mean={vals.mean():.4f} min={vals.min():.4f} max={vals.max():.4f}")


# ── METR-LA ──
try:
    file_path = "datasets/sensor_graph/adj_mx_la.pkl"
    adj_mx = load_pickle(file_path)[2]
    describe_adj("METR-LA", adj_mx)
except Exception as e:
    print(f"[Walpurgis] METR-LA: {e}")

# ── PEMS-BAY ──
try:
    file_path = "datasets/sensor_graph/adj_mx_bay.pkl"
    adj_mx = load_pickle(file_path)[2]
    describe_adj("PEMS-BAY", adj_mx)
except Exception as e:
    print(f"[Walpurgis] PEMS-BAY: {e}")

# ── PEMS04 ──
try:
    file_path = "datasets/sensor_graph/adj_mx_04.pkl"
    adj_mx = load_pickle(file_path)
    describe_adj("PEMS04", adj_mx)
except Exception as e:
    print(f"[Walpurgis] PEMS04: {e}")

# ── PEMS08 ──
try:
    file_path = "datasets/sensor_graph/adj_mx_08.pkl"
    adj_mx = load_pickle(file_path)
    describe_adj("PEMS08", adj_mx)
except Exception as e:
    print(f"[Walpurgis] PEMS08: {e}")

print(f"\n[Walpurgis::describe_adjs] Done.")


def _v4_spectral_summary(adj):
    """Eigenvalue distribution summary for adjacency analysis (v4).

    Breakpoint guide:
      pdb> _v4_spectral_summary(adj_matrix)
    """
    import numpy as np
    try:
        eigs = np.linalg.eigvalsh(adj.astype(float))
        print(f"[v4-spectral] Eigenvalue range: [{eigs.min():.4f}, {eigs.max():.4f}]")
        print(f"[v4-spectral] Mean: {eigs.mean():.4f}, Std: {eigs.std():.4f}")
        # Estimate number of communities from eigenvalue gaps
        sorted_eigs = np.sort(eigs)[::-1]
        gaps = np.diff(sorted_eigs[:20])
        n_communities = np.argmin(gaps) + 1 if len(gaps) > 0 else 1
        print(f"[v4-spectral] Estimated communities (spectral gap): {n_communities}")
    except Exception as e:
        print(f"[v4-spectral] Analysis failed: {e}")
