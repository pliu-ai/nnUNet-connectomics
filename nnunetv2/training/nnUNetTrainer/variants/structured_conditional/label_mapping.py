from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

# Fixed output head channels.
BACKGROUND_CHANNEL = 0
ECS_CHANNEL = 1
PM_CHANNEL = 2
CYTO_CHANNEL = 3
ER_MEM_CHANNEL = 4
ER_LUM_CHANNEL = 5
NUCPL_CHANNEL = 6
COND_SLOT_1_CHANNEL = 7
COND_SLOT_2_CHANNEL = 8
COND_SLOT_3_CHANNEL = 9
OTHER_CHANNEL = 10
NUM_OUTPUT_CHANNELS = 11
NUM_DYNAMIC_GROUPS = 12

OUTPUT_CHANNEL_NAMES: Tuple[str, ...] = (
    "background",
    "ecs",
    "pm",
    "cyto",
    "er_mem",
    "er_lum",
    "nucpl",
    "cond_slot_1",
    "cond_slot_2",
    "cond_slot_3",
    "other",
)

# Original CellMap label IDs for fixed classes.
FIXED_ORIGINAL_TO_OUTPUT: Dict[int, int] = {
    1: ECS_CHANNEL,
    2: PM_CHANNEL,
    3: CYTO_CHANNEL,
    17: ER_MEM_CHANNEL,
    18: ER_LUM_CHANNEL,
    27: NUCPL_CHANNEL,
}

FIXED_OUTPUT_TO_ORIGINAL: Dict[int, int] = {
    ECS_CHANNEL: 1,
    PM_CHANNEL: 2,
    CYTO_CHANNEL: 3,
    ER_MEM_CHANNEL: 17,
    ER_LUM_CHANNEL: 18,
    NUCPL_CHANNEL: 27,
}


@dataclass(frozen=True)
class DynamicGroupSpec:
    """Describes one dynamic group and its ordered conditional subclasses."""

    group_id: int
    short_name: str
    display_name: str
    original_labels: Tuple[int, ...]
    subclass_names: Tuple[str, ...]

    @property
    def num_slots(self) -> int:
        return len(self.original_labels)


DYNAMIC_GROUP_SPECS: Tuple[DynamicGroupSpec, ...] = (
    DynamicGroupSpec(0, "G1", "Mito", (4, 5, 6), ("mito_mem", "mito_lum", "mito_ribo")),
    DynamicGroupSpec(1, "G2", "Golgi", (7, 8), ("golgi_mem", "golgi_lum")),
    DynamicGroupSpec(2, "G3", "Vesicle", (9, 10), ("ves_mem", "ves_lum")),
    DynamicGroupSpec(3, "G4", "Endosome", (11, 12), ("endo_mem", "endo_lum")),
    DynamicGroupSpec(4, "G5", "Lysosome", (13, 14), ("lyso_mem", "lyso_lum")),
    DynamicGroupSpec(5, "G6", "LipidDroplet", (15, 16), ("ld_mem", "ld_lum")),
    DynamicGroupSpec(6, "G7", "ERES", (19, 20), ("eres_mem", "eres_lum")),
    DynamicGroupSpec(7, "G8", "Chromatin", (25, 26), ("hchrom", "echrom")),
    DynamicGroupSpec(8, "G9", "NuclearEnvelope", (21, 22), ("ne_mem", "ne_lum")),
    DynamicGroupSpec(9, "G10", "NuclearPore", (23, 24), ("np_out", "np_in")),
    DynamicGroupSpec(10, "G11", "Microtubule", (28, 29), ("mt_out", "mt_in")),
    DynamicGroupSpec(11, "G12", "Peroxisome", (30, 31), ("perox_mem", "perox_lum")),
)

GROUP_ID_TO_SPEC: Dict[int, DynamicGroupSpec] = {spec.group_id: spec for spec in DYNAMIC_GROUP_SPECS}
ORIGINAL_LABEL_TO_GROUP_ID: Dict[int, int] = {
    original_label: spec.group_id
    for spec in DYNAMIC_GROUP_SPECS
    for original_label in spec.original_labels
}


def get_group_spec(group_id: int) -> DynamicGroupSpec:
    if int(group_id) not in GROUP_ID_TO_SPEC:
        raise ValueError(f"Unknown dynamic group_id={group_id}. Expected range [0, {NUM_DYNAMIC_GROUPS - 1}].")
    return GROUP_ID_TO_SPEC[int(group_id)]


def get_active_conditional_output_channels(group_id: int) -> Tuple[int, ...]:
    spec = get_group_spec(group_id)
    return tuple(COND_SLOT_1_CHANNEL + i for i in range(spec.num_slots))


def get_conditional_channel_to_original_label(group_id: int) -> Dict[int, int]:
    spec = get_group_spec(group_id)
    return {
        COND_SLOT_1_CHANNEL + i: int(original_label)
        for i, original_label in enumerate(spec.original_labels)
    }


def build_active_conditional_slot_mask(group_ids: torch.Tensor) -> torch.Tensor:
    """
    Build per-sample active slot mask.

    Returns:
        Tensor with shape [B, 3], where True means the conditional slot is active
        for the selected group and should participate in slot-specific losses/metrics.
    """
    if group_ids.ndim != 1:
        group_ids = group_ids.reshape(-1)
    b = int(group_ids.shape[0])
    active = torch.zeros((b, 3), dtype=torch.bool, device=group_ids.device)
    for i in range(b):
        spec = get_group_spec(int(group_ids[i].item()))
        active[i, : spec.num_slots] = True
    return active


def infer_present_groups_from_segmentation(
    segmentation: torch.Tensor,
    ignore_label: Optional[int] = None,
) -> Set[int]:
    """Infer which dynamic groups are present in one segmentation tensor."""
    if segmentation.ndim > 0 and segmentation.shape[0] == 1:
        segmentation = segmentation[0]
    labels = torch.unique(segmentation).tolist()
    present: Set[int] = set()
    for label in labels:
        label_i = int(label)
        if label_i < 0:
            continue
        if ignore_label is not None and label_i == int(ignore_label):
            continue
        if label_i in ORIGINAL_LABEL_TO_GROUP_ID:
            present.add(int(ORIGINAL_LABEL_TO_GROUP_ID[label_i]))
    return present


def infer_present_groups_from_class_locations(properties: Mapping) -> Set[int]:
    """
    Infer dynamic group presence from nnUNet case properties['class_locations'].

    This is used by the dataloader to sample conditions at case level, and can be
    cached because properties are static per case.
    """
    class_locations = properties.get("class_locations", {})
    if not isinstance(class_locations, Mapping):
        return set()

    present: Set[int] = set()
    for spec in DYNAMIC_GROUP_SPECS:
        for original_label in spec.original_labels:
            coords = class_locations.get(int(original_label), None)
            if isinstance(coords, np.ndarray):
                if coords.size > 0:
                    present.add(spec.group_id)
                    break
            elif isinstance(coords, (list, tuple)):
                if len(coords) > 0:
                    present.add(spec.group_id)
                    break
    return present


def sample_group_id_for_case(
    present_group_ids: Sequence[int],
    p_present_group: float = 0.8,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Sample one dynamic group ID for one case.

    - With probability p_present_group, sample from present groups when possible.
    - Otherwise sample from absent groups when possible.
    """
    if rng is None:
        rng = np.random.default_rng()

    p_present_group = float(np.clip(p_present_group, 0.0, 1.0))
    present_sorted = sorted({int(i) for i in present_group_ids if 0 <= int(i) < NUM_DYNAMIC_GROUPS})
    absent = [i for i in range(NUM_DYNAMIC_GROUPS) if i not in present_sorted]

    if len(present_sorted) == 0:
        return int(rng.integers(0, NUM_DYNAMIC_GROUPS))

    draw_present = bool(rng.random() < p_present_group)
    if draw_present:
        return int(rng.choice(np.asarray(present_sorted, dtype=np.int64)))

    if len(absent) > 0:
        return int(rng.choice(np.asarray(absent, dtype=np.int64)))

    return int(rng.choice(np.asarray(present_sorted, dtype=np.int64)))


def remap_original_to_structured(
    segmentation: torch.Tensor,
    group_ids: torch.Tensor,
    ignore_label: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remap original CellMap labels into the fixed 11-channel structured target space.

    Args:
        segmentation: [B, 1, ...] (or [B, ...]) tensor of original label IDs.
        group_ids: [B] dynamic group IDs in [0, 11].
        ignore_label: Optional dataset ignore label.

    Returns:
        remapped_target: [B, 1, ...], long, values in [0, 10] on valid voxels.
        valid_mask: [B, 1, ...], bool, False on ignored voxels.
        active_conditional_slots: [B, 3], bool, active slot mask.
    """
    if segmentation.ndim < 2:
        raise ValueError("segmentation must have shape [B, 1, ...] or [B, ...].")
    if segmentation.ndim >= 2 and segmentation.shape[1] != 1:
        segmentation = segmentation[:, :1]
    if group_ids.ndim != 1:
        group_ids = group_ids.reshape(-1)

    if segmentation.shape[0] != group_ids.shape[0]:
        raise ValueError(
            f"Batch size mismatch between segmentation ({segmentation.shape[0]}) and group_ids ({group_ids.shape[0]})."
        )

    seg = segmentation.long()
    remapped = torch.full_like(seg, fill_value=OTHER_CHANNEL, dtype=torch.long)
    invalid_mask = seg < 0
    if ignore_label is not None:
        invalid_mask = invalid_mask | (seg == int(ignore_label))
    valid_mask = ~invalid_mask

    # Background and fixed classes are globally defined.
    remapped[valid_mask & (seg == 0)] = BACKGROUND_CHANNEL
    for original_label, output_channel in FIXED_ORIGINAL_TO_OUTPUT.items():
        remapped[valid_mask & (seg == int(original_label))] = int(output_channel)

    # Group-dependent conditional slots.
    for b in range(seg.shape[0]):
        spec = get_group_spec(int(group_ids[b].item()))
        seg_b = seg[b, 0]
        valid_b = valid_mask[b, 0]
        remapped_b = remapped[b, 0]
        for slot_idx, original_label in enumerate(spec.original_labels):
            output_channel = COND_SLOT_1_CHANNEL + slot_idx
            remapped_b[valid_b & (seg_b == int(original_label))] = int(output_channel)

    # Invalid voxels are set to background but masked out by valid_mask.
    remapped[~valid_mask] = BACKGROUND_CHANNEL
    active_conditional_slots = build_active_conditional_slot_mask(group_ids)
    return remapped, valid_mask, active_conditional_slots


def structured_prediction_to_original_labels(
    structured_prediction: torch.Tensor,
    group_id: int,
    background_value: int = 0,
) -> torch.Tensor:
    """
    Convert one structured prediction map back to original CellMap label IDs.

    The `other` channel is intentionally not mapped to an original semantic class.
    It is mapped to `background_value` in the reconstructed output.
    """
    pred = structured_prediction.long()
    out = torch.full_like(pred, fill_value=int(background_value), dtype=torch.long)

    for output_channel, original_label in FIXED_OUTPUT_TO_ORIGINAL.items():
        out[pred == int(output_channel)] = int(original_label)

    cond_mapping = get_conditional_channel_to_original_label(group_id)
    for output_channel, original_label in cond_mapping.items():
        out[pred == int(output_channel)] = int(original_label)

    return out


def original_label_name_lookup() -> Dict[int, str]:
    """Optional helper for readable logs/reports."""
    names = {
        0: "background",
        1: "ecs",
        2: "pm",
        3: "cyto",
        4: "mito_mem",
        5: "mito_lum",
        6: "mito_ribo",
        7: "golgi_mem",
        8: "golgi_lum",
        9: "ves_mem",
        10: "ves_lum",
        11: "endo_mem",
        12: "endo_lum",
        13: "lyso_mem",
        14: "lyso_lum",
        15: "ld_mem",
        16: "ld_lum",
        17: "er_mem",
        18: "er_lum",
        19: "eres_mem",
        20: "eres_lum",
        21: "ne_mem",
        22: "ne_lum",
        23: "np_out",
        24: "np_in",
        25: "hchrom",
        26: "echrom",
        27: "nucpl",
        28: "mt_out",
        29: "mt_in",
        30: "perox_mem",
        31: "perox_lum",
    }
    return names


def flatten_active_conditional_label_names() -> List[str]:
    """Returns all active dynamic subclass names in group order for reporting."""
    names: List[str] = []
    for spec in DYNAMIC_GROUP_SPECS:
        for slot_idx, subclass_name in enumerate(spec.subclass_names):
            names.append(f"{spec.short_name}:{subclass_name}:slot{slot_idx + 1}")
    return names


def iter_group_specs() -> Iterable[DynamicGroupSpec]:
    return iter(DYNAMIC_GROUP_SPECS)
