from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.conditional_film_unet_ternary import (
    ConditionalFiLMTernaryUNet,
)
from nnunetv2.training.nnUNetTrainer.variants.partial_label.nnUNetTrainerConditionalFiLM import (
    nnUNetTrainerConditionalFiLM,
)
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class nnUNetTrainerConditionalFiLMTernary(nnUNetTrainerConditionalFiLM):
    """
    Conditional FiLM trainer with 3-way conditioned target:
    0: background
    1: condition class (or union of selected condition classes)
    2: all other foreground classes except condition class(es)
    """

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        # Conditioned branch predicts 3 channels (bg / cond / other-fg).
        conditioned_output_channels = 3
        num_conditions = max(1, int(num_output_channels) - 1)
        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            conditioned_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return ConditionalFiLMTernaryUNet(
            backbone=backbone,
            num_conditions=num_conditions,
            num_output_channels=int(num_output_channels),
            cond_dim=64,
        )

    def _binaryize_target(self, target: torch.Tensor, cond_mask: torch.Tensor) -> torch.Tensor:
        """
        Override binary target construction with ternary labels:
        0: background
        1: selected condition class(es)
        2: other foreground classes
        """
        b = int(target.shape[0])
        all_fg_labels = self._cond_label_tensor_cpu.to(target.device)
        ternary = torch.zeros_like(target, dtype=torch.long)
        for i in range(b):
            idx = torch.nonzero(cond_mask[i] > 0, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            sel_labels = all_fg_labels[idx]
            is_sel = torch.isin(target[i], sel_labels)
            is_any_fg = torch.isin(target[i], all_fg_labels)
            ternary[i][is_sel] = 1
            ternary[i][(~is_sel) & is_any_fg] = 2

        ignore_label = self.label_manager.ignore_label
        if ignore_label is not None:
            ignore = target == int(ignore_label)
            ternary = torch.where(ignore, torch.full_like(ternary, int(ignore_label)), ternary)
        return ternary

    def _compute_classwise_stats_condition_sampled(
        self,
        output_bin: torch.Tensor,
        target_for_metric: torch.Tensor,
        valid_mask: torch.Tensor,
        cond_mask: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Here output_bin has 3 channels; "positive for condition" is argmax==1.
        num_fg = self.num_conditions
        tp_hard = np.zeros((num_fg,), dtype=np.float64)
        fp_hard = np.zeros((num_fg,), dtype=np.float64)
        fn_hard = np.zeros((num_fg,), dtype=np.float64)

        pred_fg = output_bin.argmax(dim=1, keepdim=True) == 1
        cond_labels = self._cond_label_tensor_cpu.to(target_for_metric.device)

        for b in range(int(target_for_metric.shape[0])):
            cond_idx = torch.nonzero(cond_mask[b] > 0, as_tuple=False).flatten()
            if cond_idx.numel() == 0:
                continue
            for idx in cond_idx.tolist():
                cls_label = int(cond_labels[int(idx)].item())
                pred_cls = pred_fg[b : b + 1] & valid_mask[b : b + 1]
                gt_cls = (target_for_metric[b : b + 1] == cls_label) & valid_mask[b : b + 1]
                tp_hard[int(idx)] += (pred_cls & gt_cls).sum().item()
                fp_hard[int(idx)] += (pred_cls & (~gt_cls)).sum().item()
                fn_hard[int(idx)] += ((~pred_cls) & gt_cls).sum().item()
        return tp_hard, fp_hard, fn_hard

