import torch.nn as nn
from torch_geometric.data import Data

from multiscale_operator.operators.helpers.utils import (
    copy_geometric_data,
    decompose_graph,
    MLP
)

from multiscale_operator.operators.helpers.blocks import EdgeBlock, NodeBlock

def build_mlp(in_size, hidden_size, out_size, lay_norm=True, n_layers=3):
    return MLP(in_size, hidden_size, out_size, n_layers, layer_normalized=lay_norm)


class Encoder(nn.Module):
    def __init__(self, edge_input_size=128, node_input_size=128, hidden_size=128):
        super().__init__()

        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)

    def forward(self, graph):
        node_attr, _, edge_attr, _ = decompose_graph(graph)
        node_ = self.nb_encoder(node_attr)
        if edge_attr is None:
            edge_ = None
        else:
            edge_ = self.eb_encoder(edge_attr)

        return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)


class GnBlock(nn.Module):
    def __init__(self, hidden_size=None):
        assert hidden_size is not None, "hidden_size is not defined"
        super().__init__()

        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)

        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):
        graph_last = copy_geometric_data(graph)
        graph = self.eb_module(graph)
        graph = self.nb_module(graph)

        edge_attr = graph_last.edge_attr + graph.edge_attr
        x = graph_last.x + graph.x
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)


class Decoder(nn.Module):
    def __init__(self, hidden_size=128, output_size=2):
        super().__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.x)


class EncoderProcessorDecoder(nn.Module):
    def __init__(self, cfg, num_message_passing=None, hidden_size=None):
        super().__init__()
        assert num_message_passing is not None, "num_message_passing is not defined"
        assert hidden_size is not None, "hidden_size is not defined"
        self.num_message_passing = num_message_passing
        self.hidden_size = hidden_size

    def init_shapes(self, sample: Data):
        # Edge input maybe distance
        node_input_size = sample.x.shape[-1]
        # Edge input = distance of nodes
        edge_input_size = 0 if sample.edge_attr is None else sample.edge_attr.shape[-1]

        self.encoder = Encoder(
            edge_input_size=edge_input_size,
            node_input_size=node_input_size,
            hidden_size=self.hidden_size,
        )

        processer_list = []
        for _ in range(self.num_message_passing):
            processer_list.append(GnBlock(hidden_size=self.hidden_size))
        self.processer_list = nn.ModuleList(processer_list)

        self.decoder = Decoder(hidden_size=self.hidden_size, output_size=sample.y.shape[-1])

    def forward(self, graph: Data):
        assert self.encoder is not None, "Model is not initialized"

        graph = Data(
            x=graph.x,
            y=graph.y,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            pos=graph.pos,
        )
        graph = self.encoder(graph)
        for model in self.processer_list:
            graph = model(graph)
        decoded = self.decoder(graph)

        return decoded
