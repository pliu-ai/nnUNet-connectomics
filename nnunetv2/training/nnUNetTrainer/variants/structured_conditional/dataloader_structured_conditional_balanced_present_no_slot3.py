from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase

from .label_mapping_no_slot3 import (
    DYNAMIC_GROUP_SPECS,
    NUM_DYNAMIC_GROUPS,
    infer_present_groups_from_segmentation,
)


def _non_empty_locations(class_locations: Mapping, label_id: int) -> bool:
    coords = class_locations.get(int(label_id), None)
    if isinstance(coords, np.ndarray):
        return bool(coords.size > 0)
    if isinstance(coords, (list, tuple)):
        return bool(len(coords) > 0)
    return False


class StructuredConditionalBalancedPresentDataLoader3D(nnUNetDataLoaderBase):
    """
    3D dataloader for no_slot3 structured conditional training with:
    - epoch-level near-uniform group schedule
    - patch-level present guarantee via overwrite_class foreground sampling
    """

    def __init__(
        self,
        *args,
        num_iterations_per_epoch: int,
        balance_seed: Optional[int] = None,
        verify_present_after_transforms: bool = True,
        force_fg_for_group: bool = True,
        **kwargs,
    ):
        if "label_manager" in kwargs:
            label_manager = kwargs["label_manager"]
        else:
            label_manager = args[4] if len(args) > 4 else None
        self.ignore_label = getattr(label_manager, "ignore_label", None)

        super().__init__(*args, **kwargs)
        self.num_iterations_per_epoch = int(num_iterations_per_epoch)
        if self.num_iterations_per_epoch < 1:
            raise ValueError("num_iterations_per_epoch must be >= 1")

        self.rng = np.random.default_rng(balance_seed)
        self.verify_present_after_transforms = bool(verify_present_after_transforms)
        self.force_fg_for_group = bool(force_fg_for_group)

        self._group_to_cases: Dict[int, List[str]] = {g: [] for g in range(NUM_DYNAMIC_GROUPS)}
        self._case_group_to_labels: Dict[Tuple[str, int], Tuple[int, ...]] = {}
        self._valid_groups: List[int] = []
        self._group_case_order: Dict[int, np.ndarray] = {}
        self._group_case_cursor: Dict[int, int] = {}

        self._epoch = -1
        self._epoch_group_schedule: np.ndarray = np.zeros((0,), dtype=np.int64)
        self._epoch_schedule_cursor = 0
        self._fallback_group_cursor = 0
        self._presence_fix_counter = 0

        self._build_group_case_index()
        self.set_epoch(0)

    def _build_group_case_index(self) -> None:
        for case_key in self.list_of_keys:
            case_info = self._data[case_key]
            properties = case_info.get("properties", {})
            class_locations = properties.get("class_locations", {})
            if not isinstance(class_locations, Mapping):
                continue

            for spec in DYNAMIC_GROUP_SPECS:
                labels = [int(lbl) for lbl in spec.original_labels if _non_empty_locations(class_locations, int(lbl))]
                if len(labels) == 0:
                    continue
                self._group_to_cases[spec.group_id].append(case_key)
                self._case_group_to_labels[(case_key, spec.group_id)] = tuple(labels)

        self._valid_groups = [g for g in range(NUM_DYNAMIC_GROUPS) if len(self._group_to_cases[g]) > 0]
        if len(self._valid_groups) == 0:
            raise RuntimeError("No valid dynamic groups found in training data.")

    def _reset_group_case_orders(self) -> None:
        self._group_case_order.clear()
        self._group_case_cursor.clear()
        for g in self._valid_groups:
            arr = np.asarray(self._group_to_cases[g], dtype=object)
            if arr.size > 1:
                self.rng.shuffle(arr)
            self._group_case_order[g] = arr
            self._group_case_cursor[g] = 0

    def _build_epoch_group_schedule(self, epoch: int) -> None:
        total_samples = int(self.num_iterations_per_epoch) * int(self.batch_size)
        num_groups = len(self._valid_groups)
        base = total_samples // num_groups
        rem = total_samples % num_groups

        counts = np.full((num_groups,), base, dtype=np.int64)
        if rem > 0:
            counts[:rem] += 1

        schedule = np.repeat(np.asarray(self._valid_groups, dtype=np.int64), counts)
        if schedule.size > 1:
            self.rng.shuffle(schedule)

        self._epoch = int(epoch)
        self._epoch_group_schedule = schedule
        self._epoch_schedule_cursor = 0
        self._fallback_group_cursor = 0
        self._presence_fix_counter = 0
        self._reset_group_case_orders()

    def set_epoch(self, epoch: int) -> None:
        self._build_epoch_group_schedule(int(epoch))

    def _next_group_id(self) -> int:
        if self._epoch_schedule_cursor >= self._epoch_group_schedule.size:
            g = int(self._valid_groups[self._fallback_group_cursor % len(self._valid_groups)])
            self._fallback_group_cursor += 1
            return g
        g = int(self._epoch_group_schedule[self._epoch_schedule_cursor])
        self._epoch_schedule_cursor += 1
        return g

    def _draw_case_for_group(self, group_id: int) -> str:
        order = self._group_case_order[group_id]
        if order.size == 0:
            return str(self.list_of_keys[int(self.rng.integers(0, len(self.list_of_keys)))])
        cursor = self._group_case_cursor[group_id]
        if cursor >= int(order.size):
            cursor = 0
            if order.size > 1:
                self.rng.shuffle(order)
            self._group_case_order[group_id] = order
        case_key = str(order[cursor])
        self._group_case_cursor[group_id] = cursor + 1
        return case_key

    def _crop_case(self, data: np.ndarray, seg: np.ndarray, properties: dict, force_fg: bool, overwrite_class: int):
        shape = data.shape[1:]
        dim = len(shape)
        bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties["class_locations"], overwrite_class=overwrite_class)

        valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
        valid_bbox_ubs = np.minimum(shape, bbox_ubs)

        data_slice = tuple([slice(0, data.shape[0])] + [slice(i, k) for i, k in zip(valid_bbox_lbs, valid_bbox_ubs)])
        seg_slice = tuple([slice(0, seg.shape[0])] + [slice(i, k) for i, k in zip(valid_bbox_lbs, valid_bbox_ubs)])
        data_crop = data[data_slice]
        seg_crop = seg[seg_slice]

        padding = [(-min(0, bbox_lbs[d]), max(bbox_ubs[d] - shape[d], 0)) for d in range(dim)]
        padding = ((0, 0), *padding)
        data_pad = np.pad(data_crop, padding, "constant", constant_values=0)
        seg_pad = np.pad(seg_crop, padding, "constant", constant_values=-1)
        return data_pad, seg_pad

    @staticmethod
    def _extract_highres_seg(seg_value, batch_idx: int) -> torch.Tensor:
        if isinstance(seg_value, list):
            return seg_value[0][batch_idx]
        return seg_value[batch_idx]

    def generate_train_batch(self):
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        group_ids = np.zeros((self.batch_size,), dtype=np.int64)
        chosen_keys: List[str] = []

        for j in range(self.batch_size):
            group_id = self._next_group_id()
            case_key = self._draw_case_for_group(group_id)
            data, seg, properties = self._data.load_case(case_key)

            label_choices = self._case_group_to_labels.get((case_key, group_id), None)
            if label_choices is None or len(label_choices) == 0:
                group_id = int(self._valid_groups[self._fallback_group_cursor % len(self._valid_groups)])
                self._fallback_group_cursor += 1
                case_key = self._draw_case_for_group(group_id)
                data, seg, properties = self._data.load_case(case_key)
                label_choices = self._case_group_to_labels[(case_key, group_id)]

            overwrite_class = int(label_choices[int(self.rng.integers(0, len(label_choices)))])
            force_fg = True if self.force_fg_for_group else self.get_do_oversample(j)
            data_crop, seg_crop = self._crop_case(
                data,
                seg,
                properties,
                force_fg=force_fg,
                overwrite_class=overwrite_class,
            )

            data_all[j] = data_crop
            seg_all[j] = seg_crop
            group_ids[j] = int(group_id)
            chosen_keys.append(case_key)

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all_t = torch.from_numpy(data_all).float()
                    seg_all_t = torch.from_numpy(seg_all).to(torch.int16)
                    group_ids_t = torch.from_numpy(group_ids).long()

                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        transformed = self.transforms(**{"image": data_all_t[b], "segmentation": seg_all_t[b]})
                        images.append(transformed["image"])
                        segs.append(transformed["segmentation"])

                    data_all_t = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all_t = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all_t = torch.stack(segs)

                    if self.verify_present_after_transforms:
                        for b in range(self.batch_size):
                            seg_high = self._extract_highres_seg(seg_all_t, b)
                            present_groups = infer_present_groups_from_segmentation(
                                seg_high,
                                ignore_label=self.ignore_label,
                            )
                            gid = int(group_ids_t[b].item())
                            if gid not in present_groups and len(present_groups) > 0:
                                group_ids_t[b] = int(sorted(present_groups)[0])
                                self._presence_fix_counter += 1

                    del images, segs

            return {
                "data": data_all_t,
                "target": seg_all_t,
                "keys": chosen_keys,
                "group_id": group_ids_t,
            }

        return {
            "data": data_all,
            "target": seg_all,
            "keys": chosen_keys,
            "group_id": group_ids,
        }
