from __future__ import annotations

from typing import List, Sequence, Union

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.conditional_film_unet import FiLMStage


class ConditionalFiLMTernaryUNet(nn.Module):
    """
    Conditional FiLM UNet with ternary conditioned output:
    - channel 0: background
    - channel 1: current condition class (or union of selected condition classes in multi-hot mode)
    - channel 2: all other foreground classes (except condition class(es))

    Inference compatibility:
    - condition provided: returns ternary logits (3 channels)
    - condition omitted: sweeps all conditions and returns multiclass logits where each class channel
      receives a condition-specific foreground score derived from ternary logits.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_conditions: int,
        num_output_channels: int,
        cond_dim: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = backbone.encoder
        self.decoder = backbone.decoder
        self.num_conditions = int(num_conditions)
        self.num_output_channels = int(num_output_channels)
        if self.num_conditions <= 0:
            raise ValueError("num_conditions must be > 0")
        if self.num_output_channels < 2:
            raise ValueError("num_output_channels must be >= 2")

        self.cond_emb = nn.Embedding(self.num_conditions, int(cond_dim))
        self.cond_mlp = nn.Sequential(
            nn.Linear(int(cond_dim), int(cond_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(cond_dim), int(cond_dim)),
        )

        enc_channels = list(getattr(self.encoder, "output_channels", []))
        if len(enc_channels) == 0:
            raise RuntimeError("Backbone encoder does not expose output_channels.")
        dec_channels = [int(seg.in_channels) for seg in self.decoder.seg_layers]

        self.film_encoder = nn.ModuleList([FiLMStage(cond_dim=int(cond_dim), channels=int(c)) for c in enc_channels])
        self.film_decoder = nn.ModuleList([FiLMStage(cond_dim=int(cond_dim), channels=int(c)) for c in dec_channels])

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

        # Multi-class conditioning: cond can be [B, num_conditions] multi-hot.
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

    def _conditioned_forward(self, x: torch.Tensor, condition: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        cond_repr = self._prepare_condition(condition, batch_size=x.shape[0], device=x.device)
        if cond_repr.ndim == 2:
            cond_vec = torch.matmul(cond_repr, self.cond_emb.weight)
        else:
            cond_vec = self.cond_emb(cond_repr)
        cond_vec = self.cond_mlp(cond_vec)

        skips = []
        for stage, film in zip(self.encoder.stages, self.film_encoder):
            x = stage(x)
            x = film(x, cond_vec)
            skips.append(x)

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

    @staticmethod
    def _fg_score_from_ternary(logits_ternary: torch.Tensor) -> torch.Tensor:
        # score(current_condition) against both background and "other foreground"
        return logits_ternary[:, 1] - torch.maximum(logits_ternary[:, 0], logits_ternary[:, 2])

    def _fg_to_multiclass_logits(self, fg_logits: torch.Tensor) -> torch.Tensor:
        # fg_logits shape: [B, num_conditions, ...]
        b = fg_logits.shape[0]
        spatial = fg_logits.shape[2:]
        out = fg_logits.new_zeros((b, self.num_output_channels, *spatial))
        for cond_idx in range(self.num_conditions):
            class_label = int(self.condition_label_values[cond_idx].item())
            if 0 <= class_label < self.num_output_channels:
                out[:, class_label] = fg_logits[:, cond_idx]
        return out

    def _multiclass_forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        per_condition_outputs = []
        for cond_idx in range(self.num_conditions):
            cond = torch.full((x.shape[0],), cond_idx, dtype=torch.long, device=x.device)
            out_cond = self._conditioned_forward(x, cond)
            per_condition_outputs.append(out_cond)

        if isinstance(per_condition_outputs[0], list):
            n_scales = len(per_condition_outputs[0])
            out_multi_scales: List[torch.Tensor] = []
            for s in range(n_scales):
                fg_list = []
                for out_cond in per_condition_outputs:
                    logits_cond = out_cond[s]
                    fg_list.append(self._fg_score_from_ternary(logits_cond))
                fg_logits = torch.stack(fg_list, dim=1)
                out_multi_scales.append(self._fg_to_multiclass_logits(fg_logits))
            return out_multi_scales

        fg_list = []
        for out_cond in per_condition_outputs:
            fg_list.append(self._fg_score_from_ternary(out_cond))
        fg_logits = torch.stack(fg_list, dim=1)
        return self._fg_to_multiclass_logits(fg_logits)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if condition is None:
            return self._multiclass_forward(x)
        return self._conditioned_forward(x, condition)

