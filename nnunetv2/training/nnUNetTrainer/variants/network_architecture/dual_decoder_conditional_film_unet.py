from __future__ import annotations

from typing import List, Sequence, Union

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.conditional_film_unet import FiLMStage


class DualDecoderConditionalFiLMUNet(nn.Module):
    """
    Shared encoder + two decoders:
    - multiclass decoder: direct C-way logits (for inference/metrics)
    - conditional binary decoder: bg/fg logits with FiLM conditioning
    """

    def __init__(
        self,
        backbone_multiclass: nn.Module,
        backbone_binary: nn.Module,
        num_conditions: int,
        num_output_channels: int,
        cond_dim: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = backbone_multiclass.encoder
        self.decoder_multi = backbone_multiclass.decoder
        self.decoder_binary = backbone_binary.decoder
        # Compatibility with nnUNetTrainer utilities that access `model.decoder`.
        self.decoder = self.decoder_multi

        self.num_conditions = int(num_conditions)
        self.num_output_channels = int(num_output_channels)
        if self.num_conditions <= 0:
            raise ValueError("num_conditions must be > 0")
        if self.num_output_channels < 2:
            raise ValueError("num_output_channels must be >= 2")

        enc_channels = list(getattr(self.encoder, "output_channels", []))
        if len(enc_channels) == 0:
            raise RuntimeError("Backbone encoder does not expose output_channels.")
        dec_bin_channels = [int(seg.in_channels) for seg in self.decoder_binary.seg_layers]

        self.cond_emb = nn.Embedding(self.num_conditions, int(cond_dim))
        self.cond_mlp = nn.Sequential(
            nn.Linear(int(cond_dim), int(cond_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(cond_dim), int(cond_dim)),
        )
        self.film_encoder = nn.ModuleList([FiLMStage(cond_dim=int(cond_dim), channels=int(c)) for c in enc_channels])
        self.film_decoder_binary = nn.ModuleList(
            [FiLMStage(cond_dim=int(cond_dim), channels=int(c)) for c in dec_bin_channels]
        )

        default_labels = torch.arange(1, self.num_conditions + 1, dtype=torch.long)
        self.register_buffer("condition_label_values", default_labels, persistent=True)

    @torch.no_grad()
    def set_condition_label_values(self, label_values: Union[Sequence[int], torch.Tensor]) -> None:
        vals = torch.as_tensor(label_values, dtype=torch.long, device=self.condition_label_values.device).reshape(-1)
        if vals.numel() != self.num_conditions:
            raise ValueError(
                f"condition_label_values length mismatch: expected {self.num_conditions}, got {vals.numel()}"
            )
        if (vals < 0).any():
            raise ValueError("condition_label_values must be non-negative")
        self.condition_label_values.copy_(vals)

    def _prepare_condition(
        self, condition: torch.Tensor | None, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if condition is None:
            return torch.zeros((batch_size,), dtype=torch.long, device=device)
        if not torch.is_tensor(condition):
            cond = torch.as_tensor(condition, device=device)
        else:
            cond = condition.to(device=device)

        if cond.ndim == 2:
            if cond.shape[1] != self.num_conditions:
                raise ValueError(
                    f"condition width mismatch: got {cond.shape[1]}, expected {self.num_conditions}"
                )
            if cond.shape[0] == 1 and batch_size > 1:
                cond = cond.expand(batch_size, -1)
            if cond.shape[0] != batch_size:
                raise ValueError(f"condition batch mismatch: got {cond.shape[0]}, expected {batch_size}")
            cond = (cond > 0).float()
            empty = cond.sum(dim=1) <= 0
            if empty.any():
                cond = cond.clone()
                cond[empty, 0] = 1.0
            cond = cond / cond.sum(dim=1, keepdim=True).clamp_min(1.0)
            return cond

        cond = cond.reshape(-1).long()
        if cond.numel() == 1 and batch_size > 1:
            cond = cond.expand(batch_size)
        if cond.numel() != batch_size:
            raise ValueError(f"condition batch mismatch: got {cond.numel()}, expected {batch_size}")
        return cond.clamp(min=0, max=self.num_conditions - 1)

    def _run_encoder(self, x: torch.Tensor) -> List[torch.Tensor]:
        skips = []
        for stage in self.encoder.stages:
            x = stage(x)
            skips.append(x)
        return skips

    @staticmethod
    def _decoder_forward_plain(decoder: nn.Module, skips: List[torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(decoder.stages)):
            x = decoder.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = decoder.stages[s](x)
            if decoder.deep_supervision:
                seg_outputs.append(decoder.seg_layers[s](x))
            elif s == (len(decoder.stages) - 1):
                seg_outputs.append(decoder.seg_layers[-1](x))
            lres_input = x
        seg_outputs = seg_outputs[::-1]
        if not decoder.deep_supervision:
            return seg_outputs[0]
        return seg_outputs

    def _decoder_forward_binary_film(
        self,
        skips: List[torch.Tensor],
        cond_vec: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips_bin = [film(skip, cond_vec) for skip, film in zip(skips, self.film_encoder)]
        lres_input = skips_bin[-1]
        seg_outputs = []

        for s in range(len(self.decoder_binary.stages)):
            x = self.decoder_binary.transpconvs[s](lres_input)
            x = torch.cat((x, skips_bin[-(s + 2)]), 1)
            x = self.decoder_binary.stages[s](x)
            x = self.film_decoder_binary[s](x, cond_vec)
            if self.decoder_binary.deep_supervision:
                seg_outputs.append(self.decoder_binary.seg_layers[s](x))
            elif s == (len(self.decoder_binary.stages) - 1):
                seg_outputs.append(self.decoder_binary.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]
        if not self.decoder_binary.deep_supervision:
            return seg_outputs[0]
        return seg_outputs

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        return_binary: bool = False,
        return_multiclass: bool = True,
    ):
        # Inference compatibility with nnUNetPredictor: network(x) -> multiclass logits.
        if condition is None:
            skips = self._run_encoder(x)
            return self._decoder_forward_plain(self.decoder_multi, skips)

        if not return_binary and not return_multiclass:
            raise ValueError("At least one of return_binary/return_multiclass must be True.")

        skips = self._run_encoder(x)
        out_multi = None
        out_binary = None

        if return_multiclass:
            out_multi = self._decoder_forward_plain(self.decoder_multi, skips)

        if return_binary:
            cond_repr = self._prepare_condition(condition, batch_size=x.shape[0], device=x.device)
            if cond_repr.ndim == 2:
                cond_vec = torch.matmul(cond_repr, self.cond_emb.weight)
            else:
                cond_vec = self.cond_emb(cond_repr)
            cond_vec = self.cond_mlp(cond_vec)
            out_binary = self._decoder_forward_binary_film(skips, cond_vec)

        if return_binary and return_multiclass:
            return {"multi": out_multi, "binary": out_binary}
        if return_binary:
            return out_binary
        return out_multi
