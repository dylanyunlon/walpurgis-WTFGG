import torch
import numpy as np
import pickle
import os
import sys
import csv

def _adbg(tag, val):
    if os.environ.get('SOLSTICE_DEBUG','0')!='1': return
    if isinstance(val, torch.Tensor):
        print(f"[SOL:adj:{tag}] shape={list(val.shape)} nnz={int((val.abs()>1e-6).sum().item())}", file=sys.stderr)
    else:
        print(f"[SOL:adj:{tag}] {val}", file=sys.stderr)


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None, direction=False):
    if id_filename:
        with open(id_filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            ids = [int(row[0]) for row in reader]
        id_map = {old_id: new_id for new_id, old_id in enumerate(ids)}
    else:
        id_map = None

    A = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)
    if distance_df_filename.endswith('.csv'):
        dist_df = csv.reader(open(distance_df_filename, 'r'))
        next(dist_df)
        for row in dist_df:
            if len(row) != 3:
                continue
            i, j, d = int(row[0]), int(row[1]), float(row[2])
            if id_map:
                i = id_map.get(i, -1)
                j = id_map.get(j, -1)
            if i >= 0 and j >= 0 and i < num_of_vertices and j < num_of_vertices:
                A[i, j] = 1.0
                if not direction:
                    A[j, i] = 1.0
    elif distance_df_filename.endswith('.pkl'):
        with open(distance_df_filename, 'rb') as f:
            sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
        A = adj_mx
    else:
        A = np.load(distance_df_filename)
        if A.shape[0] != num_of_vertices:
            A = A[:num_of_vertices, :num_of_vertices]

    result = torch.FloatTensor(A)
    _adbg("adjacency", result)
    return result


def calc_adj_dtw(data, n_nodes, top_k=10):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.ndim > 2:
        data = data.reshape(-1, n_nodes)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        dists = np.sqrt(np.sum((data[:, i:i+1] - data) ** 2, axis=0))
        top_indices = np.argsort(dists)[:top_k + 1]
        for j in top_indices:
            if j != i:
                adj[i, j] = np.exp(-dists[j])
    row_sums = adj.sum(axis=1, keepdims=True)
    adj = adj / np.maximum(row_sums, 1e-8)
    result = torch.FloatTensor(adj)
    _adbg("dtw_adj", result)
    return result


def calc_adj_correlation(data, n_nodes, threshold=0.3):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if data.ndim > 2:
        data = data.reshape(-1, n_nodes)
    corr = np.corrcoef(data.T)
    corr = np.nan_to_num(corr)
    adj = np.where(np.abs(corr) > threshold, np.abs(corr), 0).astype(np.float32)
    np.fill_diagonal(adj, 0)
    result = torch.FloatTensor(adj)
    _adbg("corr_adj", result)
    return result
