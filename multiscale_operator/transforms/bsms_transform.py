import torch
from torch_geometric.data import Data
from torch_scatter import scatter

from multiscale_operator.operators.helpers.helpers_bistride import (
    SeedingHeuristic,
    generate_multi_layer_stride,
)
from multiscale_operator.transforms.cache import CachedTransform


class BSMSData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "m_gs":
            return self.x.size(0)
        if key == "m_ids":
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "m_gs":
            return 1

        if key == "m_ids":
            return 0

        return super().__cat_dim__(key, value, *args, **kwargs)


class BSMSTransform(CachedTransform):
    def __init__(self, layer_num: int = 2, seed_heuristic: str = "near_center"):
        super().__init__(check_attributes=["edge_index", "pos"])
        self.layer_num = layer_num

        if seed_heuristic == "near_center":
            self.seed_heuristic = SeedingHeuristic.NearCenter
        elif seed_heuristic == "min_ave":
            self.seed_heuristic = SeedingHeuristic.MinAve
        else:
            raise ValueError(f"Unknown seed heuristic: {seed_heuristic}")

    def rewrite_indices(self, m_gs, m_ids, org_indices):
        current_indices = org_indices

        m_gs_out = [m_gs[0]]
        m_ids_out = []

        for g, idx in zip(m_gs[1:], m_ids):
            # update current indices
            # -> the indices of the original graph in the current level
            current_indices = current_indices[idx]
            m_ids_out.append(current_indices)

            # rewrite indices to the original graph
            m_gs_out.append(current_indices[g])

        return m_gs_out, m_ids_out

    @staticmethod
    def get_indices(target, indices):
        # get indices in target
        # eq = indices.reshape(-1, 1) == target.reshape(1, -1)
        # return (eq.to_sparse() * torch.arange(target.shape[0]).to(eq.device)).sum(dim=-1).to_dense()
        target_reverse_idx = scatter(torch.arange(len(target), device=target.device), target)
        return target_reverse_idx[indices]

    @staticmethod
    def rewrite_indices_back(m_gs, m_ids, org_indices):
        current_indices = org_indices

        m_gs_out = [m_gs[0]]
        m_ids_out = []

        for g, idx in zip(m_gs[1:], m_ids):
            # rewrite indices relative to the last current_indices
            # find idx (the indices of the original graph in the current level) within current_indices (last level)
            idx_back = BSMSTransform.get_indices(current_indices, idx)
            m_ids_out.append(idx_back)

            # update current indices (the selected original indices)
            current_indices = org_indices[idx]

            # rewrite the graph
            i = g[0]
            j = g[1]

            # find the indices of the original graph (g) within the indices of the current level (idx)
            i = BSMSTransform.get_indices(idx, i)
            j = BSMSTransform.get_indices(idx, j)

            m_gs_out.append(torch.stack([i, j], dim=0))

        return m_gs_out, m_ids_out

    def postprocess(self, sample_in, sample_out):
        m_gs = sample_out["m_gs"]  # first edge index is for the whole graph
        m_ids = sample_out["m_ids"]

        sample_result = BSMSData.from_dict(sample_in.to_dict())
        sample_result.m_gs = m_gs
        sample_result.m_ids = m_ids

        return sample_result

    def transform(self, sample):
        edge_index = sample.edge_index

        m_gs, m_ids = generate_multi_layer_stride(
            edge_index.clone().cpu().numpy(),
            self.layer_num,
            seed_heuristic=self.seed_heuristic,
            n=sample.pos.shape[-2],
            pos_mesh=sample.pos.clone().cpu().numpy(),
        )

        org_indices = torch.arange(sample.pos.shape[0]).to(sample.pos.device)
        m_gs = [torch.from_numpy(el).to(device=edge_index.device) for el in m_gs]
        m_ids = [torch.tensor(el).to(device=edge_index.device) for el in m_ids]
        m_gs_out, m_ids_out = self.rewrite_indices(m_gs, m_ids, org_indices)

        return {
            "m_gs": m_gs_out,
            "m_ids": m_ids_out,
        }
