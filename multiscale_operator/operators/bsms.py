from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from multiscale_operator.operators.helpers.utils import MLP
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_scatter import scatter

from multiscale_operator.transforms.bsms_transform import BSMSTransform


class GMPStack(torch.nn.Module):
    """Stack of multiple GMP modules"""

    def __init__(self, n_mp, latent_dim, hidden_layer, pos_dim, lagrangian):
        super().__init__()
        self.mps = nn.ModuleList([GMP(latent_dim, hidden_layer, pos_dim, lagrangian) for _ in range(n_mp)])

    def forward(self, x, g, pos):
        for mp in self.mps:
            x = mp(x, g, pos)
        return x


class GMP(MessagePassing):
    """Default Message Passing Module for Bi-strided Operator.
    This is similar to the DeepMind operator.
    The distance in position between nodes is computed directly within the operator"""

    def __init__(self, latent_dim, hidden_layer, pos_dim, lagrangian):
        super().__init__(aggr="add", flow="target_to_source")
        self.mlp_node_delta = MLP(2 * latent_dim, latent_dim, latent_dim, hidden_layer, True)
        edge_info_in_len = 2 * latent_dim + 2 * pos_dim + 2 if lagrangian else 2 * latent_dim + pos_dim + 1
        self.mlp_edge_info = MLP(edge_info_in_len, latent_dim, latent_dim, hidden_layer, True)
        self.lagrangian = lagrangian
        self.pos_dim = pos_dim

    def forward(self, x, g, pos):
        i = g[0]
        j = g[1]
        if len(x.shape) == 3:
            T, _, _ = x.shape
            x_i = x[:, i]
            x_j = x[:, j]
        elif len(x.shape) == 2:
            x_i = x[i]
            x_j = x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        if len(pos.shape) == 3:
            pi = pos[:, i]
            pj = pos[:, j]
        elif len(pos.shape) == 2:
            pi = pos[i]
            pj = pos[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        dir = pi - pj  # in shape (T),N,dim
        if self.lagrangian:
            norm_w = torch.norm(dir[..., : self.pos_dim], dim=-1, keepdim=True)  # in shape (T),N,1
            norm_m = torch.norm(dir[..., self.pos_dim :], dim=-1, keepdim=True)  # in shape (T),N,1
            fiber = torch.cat([dir, norm_w, norm_m], dim=-1)
        else:
            norm = torch.norm(dir, dim=-1, keepdim=True)  # in shape (T),N,1
            fiber = torch.cat([dir, norm], dim=-1)

        if len(x.shape) == 3 and len(pos.shape) == 2:
            tmp = torch.cat([fiber.unsqueeze(0).repeat(T, 1, 1), x_i, x_j], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j], dim=-1)
        edge_embedding = self.mlp_edge_info(tmp)

        aggr_out = scatter(edge_embedding, j, dim=-2, dim_size=x.shape[-2], reduce="sum")

        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_node_delta(tmp) + x


class WeightedEdgeConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add", flow="target_to_source")

    def forward(self, x, g, ew, aggragating=True):
        # aggregating: False means returning
        i = g[0]
        j = g[1]
        if len(x.shape) == 3:
            weighted_info = x[:, i] if aggragating else x[:, j]
        elif len(x.shape) == 2:
            weighted_info = x[i] if aggragating else x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        weighted_info *= ew.unsqueeze(-1)
        target_index = j if aggragating else i
        aggr_out = scatter(weighted_info, target_index, dim=-2, dim_size=x.shape[-2], reduce="sum")
        return aggr_out

    @torch.no_grad()
    def cal_ew(self, w, g):
        deg = degree(g[0], dtype=torch.float, num_nodes=w.shape[0])
        normed_w = w.squeeze(-1) / deg
        i = g[0]
        j = g[1]
        w_to_send = normed_w[i]
        eps = 1e-12
        aggr_w = scatter(w_to_send, j, dim=-1, dim_size=normed_w.size(0), reduce="sum") + eps
        ec = w_to_send / aggr_w[j]
        return ec, aggr_w


class GMPEdgeAggregatedRes(MessagePassing):
    def __init__(self, in_dim, latent_dim, hidden_layer):
        super().__init__(aggr="add", flow="target_to_source")
        self.mlp_edge_info = MLP(in_dim, latent_dim, latent_dim, hidden_layer, True)

    def forward(self, x, g, pos, pos_w, use_mat=True, use_world=True):
        i = g[0]
        j = g[1]
        if len(x.shape) == 3:
            T, _, _ = x.shape
            x_i = x[:, i]
            x_j = x[:, j]
        elif len(x.shape) == 2:
            x_i = x[i]
            x_j = x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        if len(pos.shape) == 3:
            if use_mat:
                pi = pos[:, i]
                pj = pos[:, j]
            if use_world:
                pwi = pos_w[:, i]
                pwj = pos_w[:, j]
        elif len(pos.shape) == 2:
            if use_mat:
                pi = pos[i]
                pj = pos[j]
            if use_world:
                pwi = pos_w[i]
                pwj = pos_w[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        if use_mat:
            dir = pi - pj  # in shape (T),N,dim
            norm = torch.norm(dir, dim=-1, keepdim=True)  # in shape (T),N,1
        if use_world:
            dir_w = pwi - pwj  # in shape (T),N,dim
            norm_w = torch.norm(dir_w, dim=-1, keepdim=True)  # in shape (T),N,1

        if use_mat and use_world:
            fiber = torch.cat([dir, norm, dir_w, norm_w], dim=-1)
        elif not use_mat and use_world:
            fiber = torch.cat([dir_w, norm_w], dim=-1)
        elif use_mat and not use_world:
            fiber = torch.cat([dir, norm], dim=-1)
        else:
            raise NotImplementedError("at least one pos needs to cal fiber info")

        if len(x.shape) == 3 and len(pos.shape) == 2:
            tmp = torch.cat([fiber.unsqueeze(0).repeat(T, 1, 1), x_i, x_j], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j], dim=-1)
        edge_embedding = self.mlp_edge_info(tmp)

        aggr_out = scatter(edge_embedding, j, dim=-2, dim_size=x.shape[-2], reduce="sum")

        return aggr_out


class Unpool(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, h, pre_node_num, idx):
        if len(h.shape) == 2:
            new_h = h.new_zeros([pre_node_num, h.shape[-1]])
            new_h[idx] = h
        elif len(h.shape) == 3:
            new_h = h.new_zeros([h.shape[0], pre_node_num, h.shape[-1]])
            new_h[:, idx] = h
        return new_h


class BSGMP(nn.Module):
    """
    The main module for Bi-strided Graph Message Passing.
    It contains a U-Net-like architecture.
    The message passing"""

    def __init__(self, n_mp, l_n, ld, hidden_layer, pos_dim, lagrangian, MP_model=GMPStack, edge_set_num=1):
        super().__init__()
        self.bottom_gmp = MP_model(n_mp, ld, hidden_layer, pos_dim, lagrangian)
        self.down_gmps = nn.ModuleList()
        self.up_gmps = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = l_n
        self.edge_conv = WeightedEdgeConv()
        for _ in range(self.l_n):
            self.down_gmps.append(MP_model(n_mp, ld, hidden_layer, pos_dim, lagrangian))
            self.up_gmps.append(MP_model(n_mp, ld, hidden_layer, pos_dim, lagrangian))
            self.unpools.append(Unpool())
        self.esn = edge_set_num
        self.lagrangian = lagrangian

    def forward(self, h, m_ids, m_gs, pos, weights=None):
        # h is in shape of (T), N, F
        # if edge_set_num>1, then m_g is in shape: Level,(Set),2,Edges, the 0th Set is main/material graph
        # pos is in (T),N,D
        down_outs = []
        down_ps = []
        cts = []
        w = pos.new_ones((pos.shape[-2], 1)) if weights is None else weights
        # down pass
        for i in range(self.l_n):
            h = self.down_gmps[i](h, m_gs[i], pos)
            if i == 0 and self.lagrangian:
                h = self.down_gmps[i](h, m_gs[i], pos)
            # record the infor before aggregation
            down_outs.append(h)
            down_ps.append(pos)
            # aggregate then pooling
            # cal edge_weights
            tmp_g = m_gs[i][0] if self.esn > 1 else m_gs[i]
            ew, w = self.edge_conv.cal_ew(w, tmp_g)
            h = self.edge_conv(h, tmp_g, ew)
            pos = self.edge_conv(pos, tmp_g, ew)
            cts.append(ew)
            # pooling
            if len(h.shape) == 3:
                h = h[:, m_ids[i]]
            elif len(h.shape) == 2:
                h = h[m_ids[i]]
            if len(pos.shape) == 3:
                pos = pos[:, m_ids[i]]
            elif len(pos.shape) == 2:
                pos = pos[m_ids[i]]
            w = w[m_ids[i]]
        # bottom pass
        h = self.bottom_gmp(h, m_gs[self.l_n], pos)
        if self.lagrangian:
            h = self.bottom_gmp(h, m_gs[self.l_n], pos)
        # up pass
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = m_gs[up_idx], m_ids[up_idx]
            h = self.unpools[i](h, down_outs[up_idx].shape[-2], idx)
            tmp_g = g[0] if self.esn > 1 else g
            h = self.edge_conv(h, tmp_g, cts[up_idx], aggragating=False)
            h = self.up_gmps[i](h, g, down_ps[up_idx])
            if up_idx == 0 and self.lagrangian:
                h = self.up_gmps[i](h, g, down_ps[up_idx])
            h = h.add(down_outs[up_idx])

        return h


class BSMSOperator(torch.nn.Module):
    """
    The Bi-strided Multi-scale Operator"""

    def __init__(self, cfg: DictConfig | dict, MP_times=1, layer_num=2, n_mp=1):
        super().__init__()
        self.MP_times = MP_times
        self.layer_num = layer_num
        self.n_mp = n_mp

    def init_shapes(self, sample):
        in_dim = sample.x.shape[-1]
        out_dim = sample.y.shape[-1]
        ld = 64
        mlp_hidden_layer = 3
        pos_dim = 2
        lagrangian = False

        self.encode = MLP(in_dim, ld, ld, mlp_hidden_layer, True)
        self.process = BSGMP(self.n_mp, self.layer_num, ld, mlp_hidden_layer, pos_dim, lagrangian)
        self.decode = MLP(ld, ld, out_dim, mlp_hidden_layer, False)

    def forward(self, batch):
        m_gs, m_ids = batch.m_gs, batch.m_ids
        pos = batch.pos
        x = batch.x

        # revert the indices to bsms style
        m_gs, m_ids = BSMSTransform.rewrite_indices_back(m_gs, m_ids, torch.arange(pos.shape[0]).to(pos.device))

        x = self.encode(x)
        for _ in range(self.MP_times):
            x = self.process(x, m_ids, m_gs, pos)
        x = self.decode(x)

        return x.reshape(-1, x.shape[-1])
