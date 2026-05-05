from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import autocast

from nnunetv2.utilities.helpers import dummy_context

from .label_mapping_no_slot3 import (
    FIXED_OUTPUT_TO_ORIGINAL,
    NUM_DYNAMIC_GROUPS,
    get_conditional_channel_to_original_label,
    structured_prediction_to_original_labels,
)
from .network_structured_conditional import get_main_output


@torch.no_grad()
def predict_logits_for_group(
    network: torch.nn.Module,
    image: torch.Tensor,
    group_id: int,
    use_amp: bool = True,
) -> torch.Tensor:
    """Run one conditioned forward pass and return highest-resolution logits."""
    device = image.device
    group_ids = torch.full((image.shape[0],), int(group_id), dtype=torch.long, device=device)
    amp_context = autocast(device.type, enabled=use_amp) if device.type == "cuda" else dummy_context()
    with amp_context:
        output = network(image, group_ids)
    return get_main_output(output)


@torch.no_grad()
def predict_logits_all_groups(
    network: torch.nn.Module,
    image: torch.Tensor,
    use_amp: bool = True,
) -> Dict[int, torch.Tensor]:
    """Run all dynamic groups (0..11) and return logits by group ID."""
    out: Dict[int, torch.Tensor] = {}
    for group_id in range(NUM_DYNAMIC_GROUPS):
        out[group_id] = predict_logits_for_group(network, image, group_id=group_id, use_amp=use_amp)
    return out


@torch.no_grad()
def predict_structured_labels_for_group(
    network: torch.nn.Module,
    image: torch.Tensor,
    group_id: int,
    use_amp: bool = True,
) -> torch.Tensor:
    """Option A: predict structured 11-class labels for a selected dynamic group."""
    logits = predict_logits_for_group(network, image, group_id=group_id, use_amp=use_amp)
    return logits.argmax(dim=1, keepdim=True)


@torch.no_grad()
def reconstruct_original_labels_from_group_prediction(
    structured_prediction: torch.Tensor,
    group_id: int,
) -> torch.Tensor:
    """
    Reconstruct one conditioned prediction to original CellMap IDs.

    `other` is intentionally not converted to a semantic label and maps to background.
    """
    return structured_prediction_to_original_labels(structured_prediction, group_id=group_id, background_value=0)


@torch.no_grad()
def reconstruct_original_labels_from_all_groups(
    logits_by_group: Dict[int, torch.Tensor],
    fixed_merge_mode: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Option B: combine all group runs back into original CellMap label space.

    Conflict resolution strategy:
    - compute per-original-label confidence maps from group-specific probabilities
    - fixed classes use mean/max confidence across all group runs (configurable)
    - conditional classes use max confidence across group runs
    - final label is argmax over original-label confidence maps

    The `other` channel is never projected to an original label, so it cannot
    overwrite fixed/conditional classes.

    Returns:
        merged_labels: [B, 1, ...] original CellMap label IDs
        label_scores: [B, 32, ...] confidence per original label
    """
    if len(logits_by_group) == 0:
        raise ValueError("logits_by_group is empty")
    fixed_merge_mode = str(fixed_merge_mode).lower().strip()
    if fixed_merge_mode not in {"mean", "max"}:
        raise ValueError(f"fixed_merge_mode must be 'mean' or 'max', got {fixed_merge_mode!r}")

    first_logits = next(iter(logits_by_group.values()))
    b = int(first_logits.shape[0])
    spatial = first_logits.shape[2:]
    score_map = first_logits.new_full((b, 32, *spatial), fill_value=-1e4)

    # Aggregate background from all runs.
    bg_scores = []
    # Aggregate fixed classes from all runs with configurable merge mode.
    fixed_scores = {int(v): [] for v in FIXED_OUTPUT_TO_ORIGINAL.values()}

    for group_id, logits in logits_by_group.items():
        probs = torch.softmax(logits, dim=1)
        bg_scores.append(probs[:, 0])

        # Fixed classes are shared across all runs and merged with mean later.
        for output_channel, original_label in FIXED_OUTPUT_TO_ORIGINAL.items():
            fixed_scores[int(original_label)].append(probs[:, int(output_channel)])

        # Conditional slots map group-specifically to original labels and keep max.
        cond_map = get_conditional_channel_to_original_label(int(group_id))
        for output_channel, original_label in cond_map.items():
            score_map[:, int(original_label)] = torch.maximum(
                score_map[:, int(original_label)],
                probs[:, int(output_channel)],
            )

    for original_label, score_list in fixed_scores.items():
        if len(score_list) > 0:
            stacked = torch.stack(score_list, dim=0)
            if fixed_merge_mode == "mean":
                score_map[:, int(original_label)] = stacked.mean(dim=0)
            else:
                score_map[:, int(original_label)] = stacked.max(dim=0).values

    if len(bg_scores) > 0:
        bg_score = torch.stack(bg_scores, dim=0).mean(dim=0)
        score_map[:, 0] = bg_score

    merged = score_map.argmax(dim=1, keepdim=True)
    return merged.long(), score_map
