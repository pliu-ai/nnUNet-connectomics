from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nnunetv2.utilities.ddp_allgather import AllGatherGrad

from .label_mapping_no_slot3 import (
    COND_SLOT_1_CHANNEL,
    COND_SLOT_2_CHANNEL,
    MITO_RIBO_CHANNEL,
    NUM_OUTPUT_CHANNELS,
    OTHER_CHANNEL,
)


@dataclass
class StructuredLossConfig:
    """Configuration for structured conditional loss components."""

    lambda_ce: float = 1.0
    lambda_dice: float = 1.0
    lambda_cond: float = 0.25
    lambda_suppress: float = 0.1

    enable_conditional_focus: bool = True
    enable_suppression: bool = True

    batch_dice: bool = True
    smooth: float = 1e-5
    ddp: bool = False


class StructuredConditionalLoss(nn.Module):
    """
    Structured loss for fixed 11-channel output with dynamic conditional slots.

    Supports:
    - masked CE
    - masked Soft Dice over active foreground channels
    - optional conditional-slot focus loss
    - optional conditional suppression loss
    """

    def __init__(self, config: Optional[StructuredLossConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else StructuredLossConfig()

    @staticmethod
    def _expand_per_class_mask(class_mask: torch.Tensor, ndim: int) -> torch.Tensor:
        view_shape = [class_mask.shape[0], class_mask.shape[1]] + [1] * (ndim - 2)
        return class_mask.view(*view_shape)

    @staticmethod
    def _build_foreground_class_mask(active_conditional_slots: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Foreground mask for Dice: fixed classes + active cond slots + other."""
        b = int(active_conditional_slots.shape[0])
        class_mask = torch.zeros((b, NUM_OUTPUT_CHANNELS), dtype=torch.bool, device=device)

        # Fixed foreground channels.
        class_mask[:, 1:7] = True
        class_mask[:, MITO_RIBO_CHANNEL] = True

        # Conditional slots.
        class_mask[:, COND_SLOT_1_CHANNEL] = active_conditional_slots[:, 0]
        class_mask[:, COND_SLOT_2_CHANNEL] = active_conditional_slots[:, 1]

        # Other channel is part of supervision.
        class_mask[:, OTHER_CHANNEL] = True
        return class_mask

    @staticmethod
    def _build_conditional_class_mask(active_conditional_slots: torch.Tensor, device: torch.device) -> torch.Tensor:
        b = int(active_conditional_slots.shape[0])
        class_mask = torch.zeros((b, NUM_OUTPUT_CHANNELS), dtype=torch.bool, device=device)
        class_mask[:, COND_SLOT_1_CHANNEL] = active_conditional_slots[:, 0]
        class_mask[:, COND_SLOT_2_CHANNEL] = active_conditional_slots[:, 1]
        return class_mask

    @staticmethod
    def _safe_zero(logits: torch.Tensor) -> torch.Tensor:
        return logits.sum() * 0.0

    def _masked_ce(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        active_conditional_slots: torch.Tensor,
    ) -> torch.Tensor:
        del active_conditional_slots
        target_labels = target[:, 0].long().clamp(min=0, max=NUM_OUTPUT_CHANNELS - 1)
        valid = valid_mask[:, 0].float()

        denom = valid.sum()
        if float(denom.item()) <= 0.0:
            return self._safe_zero(logits)

        ce_map = F.cross_entropy(logits, target_labels, reduction="none")
        return (ce_map * valid).sum() / denom.clamp_min(1.0)

    def _masked_soft_dice_from_probs(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        class_mask: torch.Tensor,
        zero_ref: torch.Tensor,
    ) -> torch.Tensor:
        target_labels = target[:, 0].long().clamp(min=0, max=NUM_OUTPUT_CHANNELS - 1)
        target_onehot = F.one_hot(target_labels, num_classes=NUM_OUTPUT_CHANNELS)
        target_onehot = target_onehot.permute(0, -1, *range(1, target_onehot.ndim - 1)).float()

        valid = valid_mask.float()
        class_mask_float = self._expand_per_class_mask(class_mask.float(), probs.ndim)

        masked_probs = probs * class_mask_float
        target_onehot = target_onehot * class_mask_float

        axes = tuple(range(2, probs.ndim))
        intersect = (masked_probs * target_onehot * valid).sum(dim=axes)
        pred_sum = (masked_probs * valid).sum(dim=axes)
        gt_sum = (target_onehot * valid).sum(dim=axes)

        if self.config.batch_dice:
            if self.config.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(dim=0)
                pred_sum = AllGatherGrad.apply(pred_sum).sum(dim=0)
                gt_sum = AllGatherGrad.apply(gt_sum).sum(dim=0)

            intersect = intersect.sum(dim=0)
            pred_sum = pred_sum.sum(dim=0)
            gt_sum = gt_sum.sum(dim=0)

            dice = (2.0 * intersect + self.config.smooth) / (pred_sum + gt_sum + self.config.smooth).clamp_min(1e-8)
            valid_classes = class_mask.any(dim=0).to(torch.int64)

            if self.config.ddp and dist.is_available() and dist.is_initialized():
                dist.all_reduce(valid_classes, op=dist.ReduceOp.MAX)
            valid_classes = valid_classes.bool()

            if not torch.any(valid_classes):
                return self._safe_zero(zero_ref)
            return 1.0 - dice[valid_classes].mean()

        dice = (2.0 * intersect + self.config.smooth) / (pred_sum + gt_sum + self.config.smooth).clamp_min(1e-8)
        valid_entries = class_mask
        if not torch.any(valid_entries):
            return self._safe_zero(zero_ref)
        return 1.0 - dice[valid_entries].mean()

    def _conditional_focus_loss(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        active_conditional_slots: torch.Tensor,
        zero_ref: torch.Tensor,
    ) -> torch.Tensor:
        cond_class_mask = self._build_conditional_class_mask(active_conditional_slots, probs.device)
        if not torch.any(cond_class_mask):
            return self._safe_zero(zero_ref)
        return self._masked_soft_dice_from_probs(probs, target, valid_mask, cond_class_mask, zero_ref)

    def _suppression_loss(
        self,
        probs: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        active_conditional_slots: torch.Tensor,
    ) -> torch.Tensor:
        """
        Suppress active conditional slots in non-conditional target regions.
        """
        cond_probs = torch.stack(
            (
                probs[:, COND_SLOT_1_CHANNEL],
                probs[:, COND_SLOT_2_CHANNEL],
            ),
            dim=1,
        )

        active = active_conditional_slots.float()
        active = active.view(active.shape[0], active.shape[1], *([1] * (probs.ndim - 2)))

        cond_region = ((target == COND_SLOT_1_CHANNEL) | (target == COND_SLOT_2_CHANNEL)).float()
        suppression_region = valid_mask.float() * (1.0 - cond_region)

        numer = (cond_probs * active * suppression_region).sum()
        denom = (active * suppression_region).sum().clamp_min(1.0)
        return numer / denom

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        active_conditional_slots: torch.Tensor,
        return_components: bool = False,
    ):
        ce_loss = self._masked_ce(logits, target, valid_mask, active_conditional_slots)
        probs = torch.softmax(logits, dim=1)

        foreground_class_mask = self._build_foreground_class_mask(active_conditional_slots, logits.device)
        dice_loss = self._masked_soft_dice_from_probs(probs, target, valid_mask, foreground_class_mask, logits)

        cond_loss = self._safe_zero(logits)
        if self.config.enable_conditional_focus and self.config.lambda_cond > 0:
            cond_loss = self._conditional_focus_loss(probs, target, valid_mask, active_conditional_slots, logits)

        suppress_loss = self._safe_zero(logits)
        if self.config.enable_suppression and self.config.lambda_suppress > 0:
            suppress_loss = self._suppression_loss(probs, target, valid_mask, active_conditional_slots)

        total = (
            self.config.lambda_ce * ce_loss
            + self.config.lambda_dice * dice_loss
            + self.config.lambda_cond * cond_loss
            + self.config.lambda_suppress * suppress_loss
        )

        if not return_components:
            return total

        components: Dict[str, torch.Tensor] = {
            "total": total,
            "ce": ce_loss,
            "dice": dice_loss,
            "cond_focus": cond_loss,
            "suppress": suppress_loss,
        }
        return total, components
