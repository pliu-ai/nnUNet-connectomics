from __future__ import annotations

from typing import Optional, Tuple

import torch

from .label_mapping_no_slot3 import (
    BACKGROUND_CHANNEL,
    COND_SLOT_1_CHANNEL,
    COND_SLOT_2_CHANNEL,
    FIXED_ORIGINAL_TO_OUTPUT,
    NUM_DYNAMIC_GROUPS,
    OTHER_CHANNEL,
    get_group_spec,
)


def _normalize_group_condition(
    group_condition: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not torch.is_tensor(group_condition):
        group_condition = torch.as_tensor(group_condition, device=device)
    group_condition = group_condition.to(device=device)

    if group_condition.ndim == 2:
        if group_condition.shape[1] != NUM_DYNAMIC_GROUPS:
            raise ValueError(
                f"group_condition width mismatch: got {group_condition.shape[1]}, expected {NUM_DYNAMIC_GROUPS}"
            )
        if group_condition.shape[0] == 1 and batch_size > 1:
            group_condition = group_condition.expand(batch_size, -1)
        if group_condition.shape[0] != batch_size:
            raise ValueError(
                f"group_condition batch mismatch: got {group_condition.shape[0]}, expected {batch_size}"
            )
        cond_mask = group_condition > 0
    else:
        cond_ids = group_condition.reshape(-1).long()
        if cond_ids.numel() == 1 and batch_size > 1:
            cond_ids = cond_ids.expand(batch_size)
        if cond_ids.numel() != batch_size:
            raise ValueError(f"group_condition batch mismatch: got {cond_ids.numel()}, expected {batch_size}")
        cond_ids = cond_ids.clamp(min=0, max=NUM_DYNAMIC_GROUPS - 1)
        cond_mask = torch.zeros((batch_size, NUM_DYNAMIC_GROUPS), dtype=torch.bool, device=device)
        cond_mask.scatter_(1, cond_ids[:, None], True)

    empty = cond_mask.sum(dim=1) <= 0
    if empty.any():
        cond_mask = cond_mask.clone()
        cond_mask[empty, 0] = True
    return cond_mask


def remap_original_to_structured_multi_condition(
    segmentation: torch.Tensor,
    group_condition: torch.Tensor,
    ignore_label: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remap original labels to fixed 11-channel target under multi-condition semantics.

    For each sample:
    - fixed classes remain fixed
    - slot1 is the union of slot1 labels from all selected groups
    - slot2 is the union of slot2 labels from all selected groups
    """
    if segmentation.ndim < 2:
        raise ValueError("segmentation must have shape [B, 1, ...] or [B, ...].")
    if segmentation.ndim >= 2 and segmentation.shape[1] != 1:
        segmentation = segmentation[:, :1]

    b = int(segmentation.shape[0])
    cond_mask = _normalize_group_condition(group_condition, batch_size=b, device=segmentation.device)

    seg = segmentation.long()
    remapped = torch.full_like(seg, fill_value=OTHER_CHANNEL, dtype=torch.long)

    invalid_mask = seg < 0
    if ignore_label is not None:
        invalid_mask = invalid_mask | (seg == int(ignore_label))
    valid_mask = ~invalid_mask

    remapped[valid_mask & (seg == 0)] = BACKGROUND_CHANNEL
    for original_label, output_channel in FIXED_ORIGINAL_TO_OUTPUT.items():
        remapped[valid_mask & (seg == int(original_label))] = int(output_channel)

    active_conditional_slots = torch.zeros((b, 2), dtype=torch.bool, device=seg.device)

    for i in range(b):
        selected = torch.nonzero(cond_mask[i], as_tuple=False).flatten().tolist()
        seg_i = seg[i, 0]
        valid_i = valid_mask[i, 0]
        remapped_i = remapped[i, 0]

        slot1_labels = []
        slot2_labels = []
        for gid in selected:
            spec = get_group_spec(int(gid))
            if spec.num_slots > 0:
                slot1_labels.append(int(spec.original_labels[0]))
                active_conditional_slots[i, 0] = True
            if spec.num_slots > 1:
                slot2_labels.append(int(spec.original_labels[1]))
                active_conditional_slots[i, 1] = True

        if len(slot1_labels) > 0:
            slot1_tensor = torch.as_tensor(sorted(set(slot1_labels)), dtype=seg_i.dtype, device=seg_i.device)
            remapped_i[valid_i & torch.isin(seg_i, slot1_tensor)] = COND_SLOT_1_CHANNEL

        if len(slot2_labels) > 0:
            slot2_tensor = torch.as_tensor(sorted(set(slot2_labels)), dtype=seg_i.dtype, device=seg_i.device)
            remapped_i[valid_i & torch.isin(seg_i, slot2_tensor)] = COND_SLOT_2_CHANNEL

    remapped[~valid_mask] = BACKGROUND_CHANNEL
    return remapped, valid_mask, active_conditional_slots
