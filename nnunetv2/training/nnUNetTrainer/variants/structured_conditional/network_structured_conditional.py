from __future__ import annotations

import os
from typing import List, Optional, Sequence, Union

import torch
from torch import nn

from .condition_encoding import FiLMModulation, GroupConditionEncoder


class StructuredConditionalUNet(nn.Module):
    """
    nnUNet backbone wrapper with decoder/head-side conditioning.

    Design goals:
    - shared encoder for all conditions
    - fixed output head with 11 channels
    - condition injection mainly in decoder stages (FiLM)
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_groups: int,
        num_output_channels: int,
        cond_dim: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = backbone.encoder
        self.decoder = backbone.decoder
        self.num_groups = int(num_groups)
        self.num_output_channels = int(num_output_channels)
        default_group = int(os.environ.get("NNUNET_STRUCTCOND_INFER_GROUP_ID", "0"))
        self.default_infer_group_id = int(max(0, min(default_group, self.num_groups - 1)))

        self.condition_encoder = GroupConditionEncoder(num_groups=self.num_groups, embedding_dim=int(cond_dim))

        # Segmentation layers consume decoder feature maps. We modulate those features
        # right before they are projected into logits.
        decoder_stage_channels = [int(seg_layer.in_channels) for seg_layer in self.decoder.seg_layers]
        self.decoder_film = nn.ModuleList(
            [FiLMModulation(cond_dim=self.condition_encoder.output_dim, channels=c) for c in decoder_stage_channels]
        )

    def _normalize_group_ids(self, x: torch.Tensor, group_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Normalize optional group IDs for forward compatibility with generic nnUNet
        inference entrypoints that only call `network(x)`.
        """
        if group_ids is None:
            return torch.full(
                (x.shape[0],),
                int(self.default_infer_group_id),
                dtype=torch.long,
                device=x.device,
            )
        return group_ids.reshape(-1).to(device=x.device, dtype=torch.long)

    def _forward_conditioned(
        self,
        x: torch.Tensor,
        group_ids: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        group_ids = self._normalize_group_ids(x, group_ids)
        skips = self.encode(x)
        return self.decode_from_skips(skips, group_ids)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        skips: List[torch.Tensor] = []
        for stage in self.encoder.stages:
            x = stage(x)
            skips.append(x)
        return skips

    def decode_from_skips(
        self,
        skips: List[torch.Tensor],
        group_ids: Optional[torch.Tensor],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if group_ids is None:
            group_ids = torch.full(
                (skips[0].shape[0],),
                int(self.default_infer_group_id),
                dtype=torch.long,
                device=skips[0].device,
            )
        else:
            group_ids = group_ids.reshape(-1).to(device=skips[0].device, dtype=torch.long)
        cond_vec = self.condition_encoder(group_ids, batch_size=skips[0].shape[0], device=skips[0].device)

        lres_input = skips[-1]
        seg_outputs: List[torch.Tensor] = []

        for stage_idx in range(len(self.decoder.stages)):
            x = self.decoder.transpconvs[stage_idx](lres_input)
            x = torch.cat((x, skips[-(stage_idx + 2)]), dim=1)
            x = self.decoder.stages[stage_idx](x)
            x = self.decoder_film[stage_idx](x, cond_vec)

            if self.decoder.deep_supervision:
                seg_outputs.append(self.decoder.seg_layers[stage_idx](x))
            elif stage_idx == (len(self.decoder.stages) - 1):
                seg_outputs.append(self.decoder.seg_layers[-1](x))

            lres_input = x

        seg_outputs = seg_outputs[::-1]
        if not self.decoder.deep_supervision:
            return seg_outputs[0]
        return seg_outputs

    def forward(
        self,
        x: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self._forward_conditioned(x, group_ids)


def get_main_output(output: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[0]
    return output
