from __future__ import annotations

from typing import List, Union

import torch

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.conditional_film_unet import (
    ConditionalFiLMUNet,
)


class ConditionalFiLMUNetDecoderOnly(ConditionalFiLMUNet):
    """
    Conditional FiLM UNet with decoder-only FiLM injection.

    Differences from ConditionalFiLMUNet:
    - encoder path: no FiLM modulation
    - decoder path: FiLM modulation remains enabled
    """

    def _binary_forward(self, x: torch.Tensor, condition: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        cond_repr = self._prepare_condition(condition, batch_size=x.shape[0], device=x.device)
        if cond_repr.ndim == 2:
            cond_vec = torch.matmul(cond_repr, self.cond_emb.weight)
        else:
            cond_vec = self.cond_emb(cond_repr)
        cond_vec = self.cond_mlp(cond_vec)

        # Encoder without FiLM modulation.
        skips = []
        for stage in self.encoder.stages:
            x = stage(x)
            skips.append(x)

        # Decoder with FiLM modulation (same as base model).
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.decoder.stages)):
            x = self.decoder.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.decoder.stages[s](x)
            x = self.film_decoder[s](x, cond_vec)
            if self.decoder.deep_supervision:
                seg_outputs.append(self.decoder.seg_layers[s](x))
            elif s == (len(self.decoder.stages) - 1):
                seg_outputs.append(self.decoder.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]
        if not self.decoder.deep_supervision:
            return seg_outputs[0]
        return seg_outputs

