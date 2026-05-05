from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import autocast

from nnunetv2.utilities.helpers import dummy_context

from .label_mapping_no_slot3 import (
    BACKGROUND_CHANNEL,
    COND_SLOT_1_CHANNEL,
    COND_SLOT_2_CHANNEL,
    FIXED_ORIGINAL_TO_OUTPUT,
    get_group_spec,
)
from .network_structured_conditional import get_main_output
from .trainer_structured_conditional_no_slot3 import nnUNetTrainerStructuredConditionalNoSlot3


MEM_LUM_PAIRS: Tuple[Tuple[int, int], ...] = (
    #(4, 5),
    #(7, 8),
    (9, 10),
    # (11, 12),
    # (13, 14),
    # (15, 16),
    # (17, 18),
    # (19, 20),
    # (21, 22),
    (30, 31),
)


def _binary_soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    axes = tuple(range(2, pred.ndim))
    intersect = (pred * target).sum(dim=axes)
    denom = pred.sum(dim=axes) + target.sum(dim=axes)
    dice = (2.0 * intersect + smooth) / (denom + smooth).clamp_min(1e-8)
    return 1.0 - dice.mean()


def _binary_bce_from_probs(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    BCE over probability predictions.

    This avoids calling F.binary_cross_entropy under autocast, which is disallowed.
    """
    pred = pred.float().clamp(min=eps, max=1.0 - eps)
    target = target.float()
    bce = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
    return bce.mean()


def _dilate_3d(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    if radius <= 0:
        return mask
    k = int(2 * radius + 1)
    return F.max_pool3d(mask, kernel_size=k, stride=1, padding=radius)


def paired_mem_lum_consistency_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pairs: Sequence[Tuple[int, int]],
    radius: int = 1,
) -> torch.Tensor:
    """
    Paired membrane-lumen consistency loss.

    Args:
        logits: [B, C, D, H, W]
        target: [B, D, H, W] or [B, 1, D, H, W], label IDs aligned with `logits` channels.
        pairs: membrane-lumen class ID pairs.
        radius: 3D dilation radius.
    """
    if logits.ndim != 5:
        raise ValueError(f"logits must be 5D [B, C, D, H, W], got shape={tuple(logits.shape)}")

    if target.ndim == 4:
        target = target.unsqueeze(1)
    elif target.ndim == 5 and target.shape[1] != 1:
        target = target[:, :1]
    elif target.ndim != 5:
        raise ValueError(f"target must be [B, D, H, W] or [B, 1, D, H, W], got shape={tuple(target.shape)}")

    probs = torch.softmax(logits.float(), dim=1)
    c = int(probs.shape[1])
    total = probs.sum() * 0.0
    n_valid = 0

    target_long = target.long()
    for mem_id, lum_id in pairs:
        if int(mem_id) < 0 or int(lum_id) < 0 or int(mem_id) >= c or int(lum_id) >= c:
            continue

        lum_gt = (target_long == int(lum_id)).float()
        mem_gt = (target_long == int(mem_id)).float()
        if not (torch.any(lum_gt > 0.5) or torch.any(mem_gt > 0.5)):
            continue

        lum_dilated = _dilate_3d(lum_gt, radius=radius)
        expected_mem_region = (lum_dilated - lum_gt).clamp(0.0, 1.0)

        mem_dilated = _dilate_3d(mem_gt, radius=radius)
        expected_lum_region = (lum_gt * mem_dilated).clamp(0.0, 1.0)

        pred_mem = probs[:, int(mem_id) : int(mem_id) + 1]
        pred_lum = probs[:, int(lum_id) : int(lum_id) + 1]

        bce_mem = _binary_bce_from_probs(pred_mem, expected_mem_region)
        bce_lum = _binary_bce_from_probs(pred_lum, expected_lum_region)
        dice_mem = _binary_soft_dice_loss(pred_mem, expected_mem_region)
        dice_lum = _binary_soft_dice_loss(pred_lum, expected_lum_region)

        total = total + 0.5 * (bce_mem + dice_mem) + 0.5 * (bce_lum + dice_lum)
        n_valid += 1

    if n_valid == 0:
        return total
    return total / float(n_valid)


class nnUNetTrainerMemLumConsistency(nnUNetTrainerStructuredConditionalNoSlot3):
    """
    no_slot3 structured conditional trainer with auxiliary mem-lum consistency loss.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.mem_lum_pairs: Tuple[Tuple[int, int], ...] = MEM_LUM_PAIRS
        self.mem_lum_radius: int = 1
        self.mem_lum_warmup_epochs: int = 0
        self.mem_lum_lambda: float = 0.01
        self._mem_lum_dynamic_pairs = {tuple(p) for p in self.mem_lum_pairs if tuple(p) != (17, 18)}

    @staticmethod
    def _safe_zero(x: torch.Tensor) -> torch.Tensor:
        return x.sum() * 0.0

    def _build_pairs_for_group(self, group_id: int) -> List[Tuple[int, int]]:
        pairs = [(17, 18)]
        spec = get_group_spec(int(group_id))
        if tuple(spec.original_labels) in self._mem_lum_dynamic_pairs:
            pairs.append((int(spec.original_labels[0]), int(spec.original_labels[1])))
        return pairs

    def _project_structured_logits_to_original_space(
        self,
        logits_structured: torch.Tensor,
        group_id: int,
    ) -> torch.Tensor:
        """
        Build per-sample logits in original-label ID space for pair-loss channels.
        Unavailable classes stay at a large negative logit.
        """
        max_label_id = 31
        out = torch.full(
            (1, max_label_id + 1, *logits_structured.shape[2:]),
            fill_value=-20.0,
            dtype=logits_structured.dtype,
            device=logits_structured.device,
        )

        out[:, 0] = logits_structured[:, BACKGROUND_CHANNEL]
        for original_label, output_channel in FIXED_ORIGINAL_TO_OUTPUT.items():
            out[:, int(original_label)] = logits_structured[:, int(output_channel)]

        spec = get_group_spec(int(group_id))
        if len(spec.original_labels) >= 1:
            out[:, int(spec.original_labels[0])] = logits_structured[:, COND_SLOT_1_CHANNEL]
        if len(spec.original_labels) >= 2:
            out[:, int(spec.original_labels[1])] = logits_structured[:, COND_SLOT_2_CHANNEL]
        return out

    def _compute_mem_lum_pair_loss(
        self,
        output_topo: torch.Tensor,
        target_topo: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        if target_topo.ndim == 4:
            target_topo = target_topo.unsqueeze(1)
        elif target_topo.ndim == 5 and target_topo.shape[1] != 1:
            target_topo = target_topo[:, :1]

        batch_size = int(output_topo.shape[0])
        losses: List[torch.Tensor] = []
        for b in range(batch_size):
            group_id = int(group_ids[b].item())
            pairs_b = self._build_pairs_for_group(group_id)
            logits_b_struct = output_topo[b : b + 1]
            target_b = target_topo[b : b + 1].long()
            logits_b_orig = self._project_structured_logits_to_original_space(logits_b_struct, group_id)
            loss_b = paired_mem_lum_consistency_loss(
                logits=logits_b_orig,
                target=target_b,
                pairs=pairs_b,
                radius=self.mem_lum_radius,
            )
            losses.append(loss_b)

        if len(losses) == 0:
            return self._safe_zero(output_topo)
        return torch.stack(losses).mean()

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_target(batch["target"], self.device)
        target_high = target[0] if isinstance(target, list) else target

        group_ids = self._extract_group_ids_for_batch(batch, target_high)
        bincount = torch.bincount(group_ids.detach().cpu(), minlength=self.num_dynamic_groups).numpy()
        self._group_sample_counter_epoch += bincount.astype("int64")

        remapped_target, valid_mask, active_slots = self._remap_target_for_group(target, group_ids)

        self.optimizer.zero_grad(set_to_none=True)
        amp_context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with amp_context:
            output = self.network(data, group_ids)
            output = self._sanitize_output(output)
            loss_seg = self._compute_structured_loss(output, remapped_target, valid_mask, active_slots)

            lambda_pair = 0.0 if int(self.current_epoch) < self.mem_lum_warmup_epochs else float(self.mem_lum_lambda)
            if lambda_pair > 0.0:
                output_topo = get_main_output(output)
                target_topo = target[0] if isinstance(target, list) else target
                loss_pair = self._compute_mem_lum_pair_loss(output_topo, target_topo, group_ids)
                loss = loss_seg + lambda_pair * loss_pair
            else:
                loss = loss_seg

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": loss.detach().cpu().numpy()}


class nnUNetTrainerStructuredConditionalNoSlot3MemLumConsistency(nnUNetTrainerMemLumConsistency):
    """Alias with explicit structured-conditional naming style."""
