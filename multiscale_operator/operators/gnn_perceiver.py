from __future__ import annotations

import torch
from omegaconf import DictConfig
from transformers import PerceiverConfig, PerceiverModel
from multiscale_operator.operators.bsms import GMPStack
from multiscale_operator.operators.helpers.utils import MLP

from multiscale_operator.operators.perceiver import CustomPerceiverDecoder



class GraphPerceiverProcess(torch.nn.Module):
    def __init__(self, ld, hidden_layer, pos_dim, lagrangian, n_mp, d_latents):
        super().__init__()
        self.gmp_stack_encode = GMPStack(n_mp, ld, hidden_layer, pos_dim, lagrangian)
        self.gmp_stack_decode = GMPStack(n_mp, ld, hidden_layer, pos_dim, lagrangian)
        self.config = PerceiverConfig(d_model=ld, d_latents=d_latents)
        self.perceiver = PerceiverModel(
            config=self.config, decoder=CustomPerceiverDecoder(self.config, num_channels=ld, output_num_channels=ld)
        )
        self.ld = ld

    def _reshape_batches(self, batch):
        data_list = batch.to_data_list()
        max_len = max([e.x.shape[0] for e in data_list])
        attention_mask = torch.ones(len(data_list), max_len).to(batch.x.device)

        # pad with zeros
        for idx, e in enumerate(data_list):
            attention_mask[idx][e.x.shape[0] :] = 0
            e.x = torch.cat([e.x, torch.zeros(max_len - e.x.shape[0], e.x.shape[1]).to(batch.x.device)])

        batch = batch.from_data_list(data_list)

        b = batch.num_graphs
        n = batch.x.shape[0] // b
        x = batch.x.reshape(b, n, -1)
        return x, attention_mask

    def _postprocess_output(self, outputs, attention_mask):
        # remove padding
        outputs = outputs.reshape(-1, self.ld)
        attention_mask = attention_mask.reshape(-1)
        outputs = outputs[attention_mask == 1]

        return outputs

    def forward(self, x, edge_index, pos, batch):
        x_encode = self.gmp_stack_encode(x, edge_index, pos)
        batch.x = x_encode
        x, attention_mask = self._reshape_batches(batch)
        x = self.perceiver(x, attention_mask=attention_mask)
        x = self._postprocess_output(x.logits, attention_mask)
        x_decode = self.gmp_stack_decode(x + x_encode, edge_index, pos)
        return x_decode


class GraphPeceiverOperator(torch.nn.Module):
    def __init__(self, cfg: DictConfig, d_model: int = 64, d_latents: int = 128) -> None:
        super().__init__()

        self.cfg = cfg
        self.d_latents = d_latents
        self.d_model = d_model

    def init_shapes(self, sample):
        in_dim = sample.x.shape[-1]
        out_dim = sample.y.shape[-1]
        pos_dim = sample.pos.shape[-1]
        lagrangian = False

        self.encode = MLP(in_dim, self.d_model, self.d_model, self.cfg.mlp_hidden_layer, True)
        self.process = GraphPerceiverProcess(
            self.d_model,
            self.cfg.mlp_hidden_layer,
            pos_dim,
            lagrangian,
            self.cfg.num_message_passing,
            self.d_latents,
        )
        self.decode = MLP(self.d_model, self.d_model, out_dim, self.cfg.mlp_hidden_layer, False)

    def forward(self, batch):
        edge_index = batch.edge_index
        pos = batch.pos
        x = batch.x

        x_encode = self.encode(x)
        batch_copy = batch.clone()
        x = self.process(x_encode, edge_index, pos, batch_copy)
        return self.decode(x + x_encode)
