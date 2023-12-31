from enum import Enum

import numpy as np
from sparse_dot_mkl import dot_product_mkl

from multiscale_operator.operators.helpers.helpers_BFS import (
    _BFS_dist,
    _BFS_dist_all,
    _find_clusters,
)
from multiscale_operator.operators.helpers.helpers_convert import (
    _adj_mat_to_flat_edge,
    _flat_edge_to_adj_list,
    _flat_edge_to_adj_mat,
)


class SeedingHeuristic(Enum):
    MinAve = 1
    NearCenter = 2


_INF = 1 + 1e10


def _min_ave_seed(adj_list, clusters):
    seeds = []
    dist = _BFS_dist_all(adj_list, len(adj_list))
    for c in clusters:
        d_c = dist[c]
        d_c = d_c[:, c]
        d_sum = np.sum(d_c, axis=1)
        min_ave_depth_node = c[np.argmin(d_sum)]
        seeds.append(min_ave_depth_node)

    return seeds


def _nearest_center_seed(adj_list, clusters, pos_mesh):
    seeds = []
    for c in clusters:
        center = np.mean(pos_mesh[c], axis=0)
        dd = pos_mesh[c] - center[None, :]
        normd = np.linalg.norm(dd, 2, axis=-1)
        thresh_d = np.min(normd) * 1.2
        tmp = np.where(normd < thresh_d)[0].tolist()
        try_node = [c[i] for i in tmp]
        # print(try_node)
        min_node = try_node[0]
        d_min, _ = _BFS_dist(adj_list, len(adj_list), min_node)
        min_d_sum = np.sum(d_min)
        for i in range(1, len(try_node)):
            trial = try_node[i]
            d_trial, _ = _BFS_dist(adj_list, len(adj_list), trial)
            d_trial_sum = np.sum(d_trial)
            if d_trial_sum < min_d_sum:
                min_node = trial
                min_d_sum = d_trial_sum
        seeds.append(min_node)

    return seeds


def pool_edge(g, idx, num_nodes):
    # g in scipy sparse mat
    g = _adj_mat_to_flat_edge(g)  # now flat edge list
    # idx is list
    idx = np.array(idx, dtype=np.longlong)
    idx_new_valid = np.arange(len(idx)).astype(np.longlong)
    idx_new_all = -1 * np.ones(num_nodes).astype(np.longlong)
    idx_new_all[idx] = idx_new_valid
    new_g = -1 * np.ones_like(g).astype(np.longlong)
    new_g[0] = idx_new_all[g[0]]
    new_g[1] = idx_new_all[g[1]]
    both_valid = np.logical_and(new_g[0] >= 0, new_g[1] >= 0)
    e_idx = np.where(both_valid)[0]
    new_g = new_g[:, e_idx]

    return new_g


def bstride_selection(flat_edge, seed_heuristic, pos_mesh=None, n=None):
    combined_idx_kept = set()
    adj_list = _flat_edge_to_adj_list(flat_edge, n=n)
    adj_mat = _flat_edge_to_adj_mat(flat_edge, n=n)
    # adj mat enhance the diag
    adj_mat.setdiag(1)
    # 0. compute clusters, each of which should be deivded independantly
    clusters = _find_clusters(adj_list)
    # 1. seeding: by BFS_all for small graphs, or by seed_heuristic for larger graphs

    if seed_heuristic == SeedingHeuristic.NearCenter:
        seeds = _nearest_center_seed(adj_list, clusters, pos_mesh)
    else:
        seeds = _min_ave_seed(adj_list, clusters)

    for seed, _ in zip(seeds, clusters):
        odd = set()
        even = set()
        index_kept = set()
        dist_from_cental_node, _ = _BFS_dist(adj_list, len(adj_list), seed)
        for i in range(len(dist_from_cental_node)):
            if dist_from_cental_node[i] % 2 == 0 and dist_from_cental_node[i] != _INF:
                even.add(i)
            elif dist_from_cental_node[i] % 2 == 1 and dist_from_cental_node[i] != _INF:
                odd.add(i)

        # 4. enforce n//2 candidates
        if len(even) <= len(odd) or len(odd) == 0:
            index_kept = even
            index_rmvd = odd
            delta = len(index_rmvd) - len(index_kept)
        else:
            index_kept = odd
            index_rmvd = even
            delta = len(index_rmvd) - len(index_kept)

        if delta > 0:
            # sort the dist of idx rmvd
            # cal stride based on delta nodes to select
            # generate strided idx from rmvd idx
            # union
            index_rmvd = list(index_rmvd)
            dist_id_rmvd = np.array(dist_from_cental_node)[index_rmvd]
            sort_index = np.argsort(dist_id_rmvd)
            stride = len(index_rmvd) // delta + 1
            delta_idx = sort_index[0::stride]
            delta_idx = {index_rmvd[i] for i in delta_idx}
            index_kept = index_kept.union(delta_idx)

        combined_idx_kept = combined_idx_kept.union(index_kept)

    combined_idx_kept = list(combined_idx_kept)
    adj_mat = adj_mat.tocsr().astype(float)
    # todo
    adj_mat = dot_product_mkl(adj_mat, adj_mat)  # adj_mat @ adj_mat  #
    adj_mat.setdiag(0)
    adj_mat = pool_edge(adj_mat, combined_idx_kept, n)

    return combined_idx_kept, adj_mat


def generate_multi_layer_stride(flat_edge, num_l, seed_heuristic, n, pos_mesh=None):
    m_gs = [flat_edge]
    m_ids = []
    g = flat_edge
    index_to_keep = None
    for layer in range(num_l):
        n_l = n if layer == 0 else len(index_to_keep)
        index_to_keep, g = bstride_selection(
            g, seed_heuristic=seed_heuristic, pos_mesh=pos_mesh, n=n_l
        )
        pos_mesh = pos_mesh[index_to_keep]
        m_gs.append(g)
        m_ids.append(index_to_keep)

    return m_gs, m_ids
