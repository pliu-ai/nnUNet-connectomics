from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase


class ConditionAwareDataLoader3D(nnUNetDataLoaderBase):
    """
    3D dataloader for conditional training that can sample patches according to
    the current condition labels.
    """

    def __init__(
        self,
        *args,
        num_conditions: int,
        condition_label_values: Sequence[int],
        label_to_condition_index: Dict[int, int],
        condition_sampling_strategy: str = "legacy",
        condition_positive_prob: float = 0.7,
        multi_condition_enable: bool = False,
        multi_condition_prob: float = 0.5,
        multi_condition_min_k: int = 2,
        multi_condition_max_k: int = 4,
        class_aware_resample_tries: int = 8,
        class_aware_fallback: str = "legacy_fg",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_conditions = int(num_conditions)
        self.condition_label_values = np.asarray(condition_label_values, dtype=np.int64).reshape(-1)
        self.label_to_condition_index = {int(k): int(v) for k, v in label_to_condition_index.items()}
        self.condition_sampling_strategy = str(condition_sampling_strategy).strip().lower()
        self.condition_positive_prob = float(np.clip(condition_positive_prob, 0.0, 1.0))
        self.multi_condition_enable = bool(multi_condition_enable)
        self.multi_condition_prob = float(np.clip(multi_condition_prob, 0.0, 1.0))
        self.multi_condition_min_k = int(max(2, multi_condition_min_k))
        self.multi_condition_max_k = int(max(self.multi_condition_min_k, multi_condition_max_k))
        self.class_aware_resample_tries = int(max(0, class_aware_resample_tries))
        self.class_aware_fallback = str(class_aware_fallback).strip().lower()
        if self.condition_sampling_strategy not in {"legacy", "uniform_cycle"}:
            raise ValueError(
                f"Unknown condition_sampling_strategy={self.condition_sampling_strategy}. "
                f"Supported: legacy, uniform_cycle"
            )
        if self.class_aware_fallback not in {"legacy_fg", "random"}:
            raise ValueError(
                f"Unknown class_aware_fallback={self.class_aware_fallback}. "
                f"Supported: legacy_fg, random"
            )
        self._uniform_cond_cursor = 0
        self._all_indices = list(range(self.num_conditions))

    @staticmethod
    def _extract_class_locations(properties: dict) -> dict:
        cl = properties.get("class_locations", {})
        return cl if isinstance(cl, dict) else {}

    def _present_condition_indices(self, properties: dict) -> List[int]:
        class_locations = self._extract_class_locations(properties)
        present: List[int] = []
        for k, v in class_locations.items():
            if not isinstance(v, (list, tuple, np.ndarray)) or len(v) == 0:
                continue
            if isinstance(k, (int, np.integer)):
                lbl = int(k)
                if lbl in self.label_to_condition_index:
                    present.append(self.label_to_condition_index[lbl])
        if len(present) == 0:
            return []
        return sorted(set(present))

    def _draw_uniform_cycle_indices(self, k: int) -> List[int]:
        if self.num_conditions <= 0:
            return []
        k = int(max(1, min(k, self.num_conditions)))
        start = int(self._uniform_cond_cursor)
        out = [int((start + j) % self.num_conditions) for j in range(k)]
        self._uniform_cond_cursor = int((start + k) % self.num_conditions)
        return out

    def _sample_condition_mask_for_case(self, present_idx: List[int]) -> np.ndarray:
        cond_mask = np.zeros((self.num_conditions,), dtype=np.float32)

        do_multi = (
            self.multi_condition_enable
            and self.num_conditions > 1
            and (np.random.rand() < self.multi_condition_prob)
        )

        if self.condition_sampling_strategy == "uniform_cycle":
            if do_multi:
                max_k = min(self.multi_condition_max_k, self.num_conditions)
                min_k = min(max(self.multi_condition_min_k, 2), max_k)
                if min_k > max_k:
                    min_k = max_k
                k = int(min_k) if min_k == max_k else int(np.random.randint(min_k, max_k + 1))
            else:
                k = 1
            sel = self._draw_uniform_cycle_indices(k)
            cond_mask[sel] = 1.0
            return cond_mask

        # legacy
        present_set = set(present_idx)
        negatives = [i for i in self._all_indices if i not in present_set]
        if do_multi:
            do_positive = bool(np.random.rand() < self.condition_positive_prob)
            if do_positive and len(present_idx) > 0:
                pool = present_idx
            else:
                pool = negatives if len(negatives) > 0 else (present_idx if len(present_idx) > 0 else self._all_indices)
            max_k = min(self.multi_condition_max_k, len(pool), self.num_conditions)
            min_k = min(max(self.multi_condition_min_k, 2), max_k)
            if max_k <= 0:
                cond_mask[int(np.random.randint(self.num_conditions))] = 1.0
                return cond_mask
            if min_k > max_k:
                min_k = max_k
            k = int(min_k) if min_k == max_k else int(np.random.randint(min_k, max_k + 1))
            sel = np.random.choice(np.asarray(pool, dtype=np.int64), size=k, replace=False)
            cond_mask[sel] = 1.0
            return cond_mask

        # single condition
        if len(present_idx) > 0:
            do_positive = bool(np.random.rand() < self.condition_positive_prob)
            if do_positive:
                choice = int(np.random.choice(np.asarray(present_idx, dtype=np.int64)))
            else:
                pool = negatives if len(negatives) > 0 else present_idx
                choice = int(np.random.choice(np.asarray(pool, dtype=np.int64)))
        else:
            choice = int(np.random.randint(self.num_conditions))
        cond_mask[choice] = 1.0
        return cond_mask

    def _condition_label_values_from_mask(self, cond_mask: np.ndarray) -> List[int]:
        idx = np.where(cond_mask > 0)[0].astype(np.int64).tolist()
        if len(idx) == 0:
            return []
        return [int(self.condition_label_values[i]) for i in idx]

    def _pick_anchor_label(self, properties: dict, cond_label_values: List[int]) -> int | None:
        if len(cond_label_values) == 0:
            return None
        class_locations = self._extract_class_locations(properties)
        candidates: List[int] = []
        for lbl in cond_label_values:
            loc = class_locations.get(int(lbl), None)
            if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) > 0:
                candidates.append(int(lbl))
        if len(candidates) == 0:
            return None
        return int(np.random.choice(np.asarray(candidates, dtype=np.int64)))

    def _case_has_any_condition_label(self, properties: dict, cond_label_values: List[int]) -> bool:
        if len(cond_label_values) == 0:
            return False
        class_locations = self._extract_class_locations(properties)
        for lbl in cond_label_values:
            loc = class_locations.get(int(lbl), None)
            if isinstance(loc, (list, tuple, np.ndarray)) and len(loc) > 0:
                return True
        return False

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        cond_all = np.zeros((self.batch_size, self.num_conditions), dtype=np.float32)
        chosen_keys: List[str] = []

        for j, key in enumerate(selected_keys):
            data, seg, properties = self._data.load_case(key)
            present_idx = self._present_condition_indices(properties)
            cond_mask = self._sample_condition_mask_for_case(present_idx)
            cond_all[j] = cond_mask

            cond_label_values = self._condition_label_values_from_mask(cond_mask)
            anchor_label = self._pick_anchor_label(properties, cond_label_values)

            if anchor_label is None and self.class_aware_resample_tries > 0 and len(cond_label_values) > 0:
                for _ in range(self.class_aware_resample_tries):
                    cand_key = self.list_of_keys[int(np.random.randint(len(self.list_of_keys)))]
                    cand_props = self._data[cand_key]["properties"]
                    if self._case_has_any_condition_label(cand_props, cond_label_values):
                        key = cand_key
                        data, seg, properties = self._data.load_case(key)
                        anchor_label = self._pick_anchor_label(properties, cond_label_values)
                        break

            shape = data.shape[1:]
            dim = len(shape)
            if anchor_label is not None:
                bbox_lbs, bbox_ubs = self.get_bbox(shape, True, properties["class_locations"], overwrite_class=anchor_label)
            else:
                force_fg = self.get_do_oversample(j) if self.class_aware_fallback == "legacy_fg" else False
                bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties["class_locations"])

            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, k) for i, k in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]
            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, k) for i, k in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[d]), max(bbox_ubs[d] - shape[d], 0)) for d in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, "constant", constant_values=0)
            seg_all[j] = np.pad(seg, padding, "constant", constant_values=-1)
            chosen_keys.append(key)

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    cond_all_t = torch.from_numpy(cond_all).float()
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{"image": data_all[b], "segmentation": seg_all[b]})
                        images.append(tmp["image"])
                        segs.append(tmp["segmentation"])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images
            return {"data": data_all, "target": seg_all, "keys": chosen_keys, "condition_mask": cond_all_t}

        return {"data": data_all, "target": seg_all, "keys": chosen_keys, "condition_mask": cond_all}
