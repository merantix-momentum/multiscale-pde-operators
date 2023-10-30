import enum

import torch
from torch_geometric.data import Data
from torch.nn import LayerNorm, Linear, ReLU
from torch.nn import Sequential as Seq


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def decompose_graph(graph):
    # graph: torch_geometric.data.data.Data
    # TODO: make it more robust
    x, edge_index, edge_attr, global_attr = None, None, None, None
    for key in graph.keys:
        if key == "x":
            x = graph.x
        elif key == "edge_index":
            edge_index = graph.edge_index
        elif key == "edge_attr":
            edge_attr = graph.edge_attr
        elif key == "global_attr":
            global_attr = graph.global_attr
        else:
            pass
    return (x, edge_index, edge_attr, global_attr)


# see https://github.com/sungyongs/dpgn/blob/master/utils.py
def copy_geometric_data(graph):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    node_attr, edge_index, edge_attr, global_attr = decompose_graph(graph)

    ret = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
    ret.global_attr = global_attr

    return ret


class MLP(torch.nn.Module):
    """Basic MLP module."""

    def __init__(self, input_dim, latent_dim, output_dim, hidden_layers, layer_normalized=True):
        super().__init__()
        modules = []
        for layer in range(hidden_layers):
            if layer == 0:
                modules.append(Linear(input_dim, latent_dim))
            else:
                modules.append(Linear(latent_dim, latent_dim))
            modules.append(ReLU())
        modules.append(Linear(latent_dim, output_dim))
        if layer_normalized:
            modules.append(LayerNorm(output_dim, elementwise_affine=False))

        self.seq = Seq(*modules)

    def forward(self, x):
        return self.seq(x)
