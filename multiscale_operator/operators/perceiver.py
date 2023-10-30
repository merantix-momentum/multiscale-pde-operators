from __future__ import annotations

import torch
from omegaconf import DictConfig
from transformers import PerceiverConfig, PerceiverModel
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverAbstractDecoder,
    PerceiverBasicDecoder,
    PerceiverDecoderOutput,
)
from multiscale_operator.operators.helpers.utils import MLP


class CustomPerceiverDecoder(PerceiverAbstractDecoder):
    """Cross-attention based decoder module for graph nodes."""

    def __init__(self, config, output_num_channels=1, **decoder_kwargs):
        super().__init__()
        self.output_num_channels = output_num_channels
        self.decoder = PerceiverBasicDecoder(
            config,
            output_num_channels=output_num_channels,
            position_encoding_type="none",
            **decoder_kwargs,
        )

    @property
    def num_query_channels(self) -> int:
        return self.decoder.num_query_channels

    def decoder_query(
        self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None
    ):
        return inputs

    def forward(
        self,
        query: torch.Tensor,
        z: torch.FloatTensor,
        query_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = False,
    ) -> PerceiverDecoderOutput:
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)
        preds = decoder_outputs.logits
        return PerceiverDecoderOutput(
            logits=preds, cross_attentions=decoder_outputs.cross_attentions
        )


class PerceiverOperator(torch.nn.Module):
    def __init__(self, cfg: DictConfig, d_model: int = 64, d_latents: int = 128):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.config = PerceiverConfig(d_model=d_model, d_latents=d_latents)

    def _reshape_batches(self, batch):
        data_list = batch.to_data_list()
        max_len = max([e.x.shape[0] for e in data_list])
        # pad with zeros
        for e in data_list:
            e.attention_mask = torch.ones(max_len).to(batch.x.device)
            e.attention_mask[e.x.shape[0] :] = 0
            e.x = torch.cat([e.x, torch.zeros(max_len - e.x.shape[0], e.x.shape[1]).to(batch.x.device)])
            e.pos = torch.cat([e.pos, torch.zeros(max_len - e.pos.shape[0], e.pos.shape[1]).to(batch.x.device)])
        batch_new = batch.from_data_list(data_list)

        b = batch_new.num_graphs
        n = batch_new.x.shape[0] // b
        x = batch_new.x.reshape(b, n, -1)
        pos = batch_new.pos.reshape(b, n, -1)
        attention_mask = batch_new.attention_mask.reshape(b, n)

        return torch.cat([x, pos], dim=-1), attention_mask

    def _postprocess_output(self, outputs, attention_mask):
        # remove padding
        outputs = outputs.reshape(-1, self.cfg.model_cfg.output_channels)
        attention_mask = attention_mask.reshape(-1)
        outputs = outputs[attention_mask == 1]

        return outputs

    def init_shapes(self, sample):
        inputs, _ = self._reshape_batches(sample)
        self.encoder = MLP(inputs.shape[-1], self.d_model, self.d_model, 3)
        if self.cfg.model_cfg.mlp_decoder:
            self.decoder = MLP(
                self.d_model,
                self.d_model,
                self.cfg.model_cfg.output_channels,
                3,
                layer_normalized=False,
            )
            output_num_channels = self.d_model
        else:
            output_num_channels = self.cfg.model_cfg.output_channels
        self.model = PerceiverModel(
            config=self.config,
            decoder=CustomPerceiverDecoder(
                self.config, num_channels=self.d_model, output_num_channels=output_num_channels
            ),
        )

    def forward(self, batch):
        inputs, attention_mask = self._reshape_batches(batch)
        x_encode = self.encoder(inputs)
        outputs = self.model(x_encode, attention_mask=attention_mask).logits
        if self.cfg.model_cfg.mlp_decoder:
            # including skip connection
            outputs = self.decoder(outputs + x_encode)

        return self._postprocess_output(outputs, attention_mask)
