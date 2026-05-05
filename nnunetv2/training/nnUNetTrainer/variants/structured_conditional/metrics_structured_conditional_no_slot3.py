from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from .label_mapping_no_slot3 import (
    COND_SLOT_1_CHANNEL,
    COND_SLOT_2_CHANNEL,
    DYNAMIC_GROUP_SPECS,
    ECS_CHANNEL,
    ER_LUM_CHANNEL,
    ER_MEM_CHANNEL,
    MITO_RIBO_CHANNEL,
    NUM_DYNAMIC_GROUPS,
    NUM_OUTPUT_CHANNELS,
    OTHER_CHANNEL,
    PM_CHANNEL,
    CYTO_CHANNEL,
    NUCPL_CHANNEL,
    get_active_conditional_output_channels,
)

FIXED_CHANNEL_NAME_MAP = {
    ECS_CHANNEL: "ecs",
    PM_CHANNEL: "pm",
    CYTO_CHANNEL: "cyto",
    ER_MEM_CHANNEL: "er_mem",
    ER_LUM_CHANNEL: "er_lum",
    NUCPL_CHANNEL: "nucpl",
    MITO_RIBO_CHANNEL: "mito_ribo",
}

# Fixed original labels in CellMap.
FIXED_ORIGINAL_LABEL_TO_CHANNEL = {
    1: ECS_CHANNEL,
    2: PM_CHANNEL,
    3: CYTO_CHANNEL,
    6: MITO_RIBO_CHANNEL,
    17: ER_MEM_CHANNEL,
    18: ER_LUM_CHANNEL,
    27: NUCPL_CHANNEL,
}


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def dice_from_stats(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    return safe_divide(2.0 * tp, 2.0 * tp + fp + fn)


def iou_from_stats(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    return safe_divide(tp, tp + fp + fn)


def _binary_counts(pred_mask: torch.Tensor, gt_mask: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[float, float, float]:
    pred_valid = pred_mask & valid_mask
    gt_valid = gt_mask & valid_mask
    tp = float((pred_valid & gt_valid).sum().item())
    fp = float((pred_valid & (~gt_valid)).sum().item())
    fn = float(((~pred_valid) & gt_valid).sum().item())
    return tp, fp, fn


def compute_group_confusion_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    group_id: int,
    present_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute confusion statistics for one forward pass with a specific dynamic group.

    Returns:
        class_tp/fp/fn: [11]
        cond_slot_tp/fp/fn: [2]
        merged_cond_tp/fp/fn: [1]
    """
    pred = logits.argmax(dim=1, keepdim=True)
    valid = valid_mask.bool()

    class_tp = np.zeros((NUM_OUTPUT_CHANNELS,), dtype=np.float64)
    class_fp = np.zeros((NUM_OUTPUT_CHANNELS,), dtype=np.float64)
    class_fn = np.zeros((NUM_OUTPUT_CHANNELS,), dtype=np.float64)

    active_cond_channels = get_active_conditional_output_channels(group_id)
    active_eval_channels = [ECS_CHANNEL, PM_CHANNEL, CYTO_CHANNEL, ER_MEM_CHANNEL, ER_LUM_CHANNEL, NUCPL_CHANNEL, MITO_RIBO_CHANNEL]
    active_eval_channels += list(active_cond_channels)
    active_eval_channels += [OTHER_CHANNEL]

    for channel_idx in active_eval_channels:
        pred_mask = pred == int(channel_idx)
        gt_mask = target == int(channel_idx)
        if present_only and not torch.any(gt_mask & valid):
            continue
        tp, fp, fn = _binary_counts(pred_mask, gt_mask, valid)
        class_tp[int(channel_idx)] += tp
        class_fp[int(channel_idx)] += fp
        class_fn[int(channel_idx)] += fn

    cond_tp = np.zeros((2,), dtype=np.float64)
    cond_fp = np.zeros((2,), dtype=np.float64)
    cond_fn = np.zeros((2,), dtype=np.float64)
    for slot_idx, channel_idx in enumerate((COND_SLOT_1_CHANNEL, COND_SLOT_2_CHANNEL)):
        if int(channel_idx) not in active_cond_channels:
            continue
        pred_mask = pred == int(channel_idx)
        gt_mask = target == int(channel_idx)
        if present_only and not torch.any(gt_mask & valid):
            continue
        tp, fp, fn = _binary_counts(pred_mask, gt_mask, valid)
        cond_tp[slot_idx] += tp
        cond_fp[slot_idx] += fp
        cond_fn[slot_idx] += fn

    pred_cond_merged = torch.zeros_like(valid, dtype=torch.bool)
    gt_cond_merged = torch.zeros_like(valid, dtype=torch.bool)
    for channel_idx in active_cond_channels:
        pred_cond_merged |= pred == int(channel_idx)
        gt_cond_merged |= target == int(channel_idx)

    if present_only and not torch.any(gt_cond_merged & valid):
        merged_tp, merged_fp, merged_fn = 0.0, 0.0, 0.0
    else:
        merged_tp, merged_fp, merged_fn = _binary_counts(pred_cond_merged, gt_cond_merged, valid)
    merged_tp_arr = np.asarray([merged_tp], dtype=np.float64)
    merged_fp_arr = np.asarray([merged_fp], dtype=np.float64)
    merged_fn_arr = np.asarray([merged_fn], dtype=np.float64)

    return (
        class_tp,
        class_fp,
        class_fn,
        cond_tp,
        cond_fp,
        cond_fn,
        merged_tp_arr,
        merged_fp_arr,
        merged_fn_arr,
    )


def build_validation_report(
    class_tp: np.ndarray,
    class_fp: np.ndarray,
    class_fn: np.ndarray,
    cond_tp: np.ndarray,
    cond_fp: np.ndarray,
    cond_fn: np.ndarray,
    merged_cond_tp: np.ndarray,
    merged_cond_fp: np.ndarray,
    merged_cond_fn: np.ndarray,
) -> Dict[str, object]:
    """Build a structured metrics dictionary for logging/inspection."""
    class_dice = dice_from_stats(class_tp, class_fp, class_fn)
    class_iou = iou_from_stats(class_tp, class_fp, class_fn)

    fixed_class_dice = {
        name: float(class_dice[channel])
        for channel, name in FIXED_CHANNEL_NAME_MAP.items()
    }
    fixed_class_iou = {
        name: float(class_iou[channel])
        for channel, name in FIXED_CHANNEL_NAME_MAP.items()
    }

    cond_subclass_dice: Dict[str, float] = {}
    cond_subclass_precision: Dict[str, float] = {}
    cond_subclass_recall: Dict[str, float] = {}

    for spec in DYNAMIC_GROUP_SPECS:
        for slot_idx, subclass_name in enumerate(spec.subclass_names):
            key = f"{spec.short_name}_{subclass_name}"
            tp = cond_tp[spec.group_id, slot_idx]
            fp = cond_fp[spec.group_id, slot_idx]
            fn = cond_fn[spec.group_id, slot_idx]

            denom_dice = 2.0 * tp + fp + fn
            denom_precision = tp + fp
            denom_recall = tp + fn

            cond_subclass_dice[key] = float((2.0 * tp / denom_dice) if denom_dice > 0 else 0.0)
            cond_subclass_precision[key] = float((tp / denom_precision) if denom_precision > 0 else 0.0)
            cond_subclass_recall[key] = float((tp / denom_recall) if denom_recall > 0 else 0.0)

    # Original CellMap foreground labels (1..31), ordered by original ID.
    original31_dice: List[float] = []
    original31_gt_present: List[bool] = []
    for original_label in range(1, 32):
        if original_label in FIXED_ORIGINAL_LABEL_TO_CHANNEL:
            channel = FIXED_ORIGINAL_LABEL_TO_CHANNEL[original_label]
            original31_dice.append(float(class_dice[channel]))
            original31_gt_present.append(bool((class_tp[channel] + class_fn[channel]) > 0))
            continue

        value = 0.0
        gt_present = False
        found = False
        for spec in DYNAMIC_GROUP_SPECS:
            if original_label in spec.original_labels:
                slot_idx = spec.original_labels.index(original_label)
                tp = cond_tp[spec.group_id, slot_idx]
                fp = cond_fp[spec.group_id, slot_idx]
                fn = cond_fn[spec.group_id, slot_idx]
                denom = 2.0 * tp + fp + fn
                value = float((2.0 * tp / denom) if denom > 0 else 0.0)
                gt_present = bool((tp + fn) > 0)
                found = True
                break
        if not found:
            # Should not happen for CellMap IDs 1..31, but keep robust behavior.
            value = 0.0
        original31_dice.append(value)
        original31_gt_present.append(gt_present)

    merged_conditional_dice = {
        spec.short_name: float(
            (2.0 * merged_cond_tp[spec.group_id] / (2.0 * merged_cond_tp[spec.group_id] + merged_cond_fp[spec.group_id] + merged_cond_fn[spec.group_id]))
            if (2.0 * merged_cond_tp[spec.group_id] + merged_cond_fp[spec.group_id] + merged_cond_fn[spec.group_id]) > 0
            else 0.0
        )
        for spec in DYNAMIC_GROUP_SPECS
    }

    active_values: List[float] = []
    active_values.extend(fixed_class_dice.values())
    active_values.append(float(class_dice[OTHER_CHANNEL]))
    active_values.extend(cond_subclass_dice.values())
    original31_present_dice = [
        dice_value
        for dice_value, is_present in zip(original31_dice, original31_gt_present)
        if is_present
    ]

    report: Dict[str, object] = {
        "fixed_class_dice": fixed_class_dice,
        "fixed_class_iou": fixed_class_iou,
        "other_dice": float(class_dice[OTHER_CHANNEL]),
        "other_iou": float(class_iou[OTHER_CHANNEL]),
        "conditional_subclass_dice": cond_subclass_dice,
        "conditional_subclass_precision": cond_subclass_precision,
        "conditional_subclass_recall": cond_subclass_recall,
        "merged_conditional_dice": merged_conditional_dice,
        "output_channel_dice_foreground": [float(class_dice[i]) for i in range(1, NUM_OUTPUT_CHANNELS)],
        "summary": {
            "mean_active_dice": float(np.mean(active_values)) if len(active_values) > 0 else 0.0,
            "mean_fixed_dice": float(np.mean(list(fixed_class_dice.values()))) if len(fixed_class_dice) > 0 else 0.0,
            "mean_conditional_subclass_dice": float(np.mean(list(cond_subclass_dice.values()))) if len(cond_subclass_dice) > 0 else 0.0,
            "mean_merged_conditional_dice": float(np.mean(list(merged_conditional_dice.values()))) if len(merged_conditional_dice) > 0 else 0.0,
            "mean_original31_dice": float(np.mean(original31_dice)) if len(original31_dice) > 0 else 0.0,
            "mean_original31_present_dice": float(np.mean(original31_present_dice)) if len(original31_present_dice) > 0 else 0.0,
            "num_original31_present": int(len(original31_present_dice)),
        },
        "original31_dice": original31_dice,
        "original31_gt_present": original31_gt_present,
    }
    return report


def empty_validation_accumulators() -> Dict[str, np.ndarray]:
    """Helper for consistent accumulator initialization."""
    return {
        "class_tp": np.zeros((NUM_OUTPUT_CHANNELS,), dtype=np.float64),
        "class_fp": np.zeros((NUM_OUTPUT_CHANNELS,), dtype=np.float64),
        "class_fn": np.zeros((NUM_OUTPUT_CHANNELS,), dtype=np.float64),
        "cond_tp": np.zeros((NUM_DYNAMIC_GROUPS, 2), dtype=np.float64),
        "cond_fp": np.zeros((NUM_DYNAMIC_GROUPS, 2), dtype=np.float64),
        "cond_fn": np.zeros((NUM_DYNAMIC_GROUPS, 2), dtype=np.float64),
        "merged_cond_tp": np.zeros((NUM_DYNAMIC_GROUPS,), dtype=np.float64),
        "merged_cond_fp": np.zeros((NUM_DYNAMIC_GROUPS,), dtype=np.float64),
        "merged_cond_fn": np.zeros((NUM_DYNAMIC_GROUPS,), dtype=np.float64),
    }
