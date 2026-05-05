from __future__ import annotations

import os
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import autocast

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context

from .label_mapping_no_slot3 import (
    NUM_DYNAMIC_GROUPS,
    NUM_OUTPUT_CHANNELS,
    infer_present_groups_from_segmentation,
    sample_group_id_for_case,
)
from .label_mapping_no_slot3_multi_condition import remap_original_to_structured_multi_condition
from .network_structured_conditional_multi_condition import StructuredConditionalUNetMultiCondition
from .trainer_structured_conditional_no_slot3 import nnUNetTrainerStructuredConditionalNoSlot3


class nnUNetTrainerStructuredConditionalNoSlot3MultiCondition(nnUNetTrainerStructuredConditionalNoSlot3):
    """
    Multi-condition variant of no_slot3 trainer.

    Differences from base no_slot3 trainer:
    - conditioning signal is multi-hot group mask [B, NUM_DYNAMIC_GROUPS]
    - slot targets are remapped with union semantics across selected groups
    - old trainer behavior remains untouched in its original class
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

        self.multi_condition_enable = (
            str(os.environ.get("NNUNET_STRUCTCOND_MULTI_ENABLE", "1")).lower() in {"1", "true", "yes", "y"}
        )
        self.multi_condition_prob = float(os.environ.get("NNUNET_STRUCTCOND_MULTI_PROB", "0.5"))
        self.multi_condition_min_k = int(os.environ.get("NNUNET_STRUCTCOND_MULTI_MIN_K", "2"))
        self.multi_condition_max_k = int(
            os.environ.get("NNUNET_STRUCTCOND_MULTI_MAX_K", str(NUM_DYNAMIC_GROUPS))
        )

        self.multi_condition_prob = float(np.clip(self.multi_condition_prob, 0.0, 1.0))
        self.multi_condition_max_k = int(max(1, min(NUM_DYNAMIC_GROUPS, self.multi_condition_max_k)))
        if self.multi_condition_max_k > 1:
            self.multi_condition_min_k = int(max(2, self.multi_condition_min_k))
        else:
            self.multi_condition_min_k = 1
        self.multi_condition_min_k = int(min(self.multi_condition_min_k, self.multi_condition_max_k))

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        del num_output_channels
        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            NUM_OUTPUT_CHANNELS,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return StructuredConditionalUNetMultiCondition(
            backbone=backbone,
            num_groups=NUM_DYNAMIC_GROUPS,
            num_output_channels=NUM_OUTPUT_CHANNELS,
            cond_dim=64,
        )

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            "[StructuredConditionalMulti] "
            f"enabled={self.multi_condition_enable}, "
            f"prob={self.multi_condition_prob:.3f}, "
            f"k=[{self.multi_condition_min_k},{self.multi_condition_max_k}]"
        )

    def _sample_group_mask_from_target(self, target_high: torch.Tensor) -> torch.Tensor:
        b = int(target_high.shape[0])
        group_mask = torch.zeros((b, self.num_dynamic_groups), dtype=torch.float32, device=self.device)

        all_indices = list(range(self.num_dynamic_groups))
        for i in range(b):
            present = sorted(
                infer_present_groups_from_segmentation(
                    target_high[i],
                    ignore_label=self.label_manager.ignore_label,
                )
            )
            present_set = set(int(g) for g in present)
            negatives = [idx for idx in all_indices if idx not in present_set]

            do_multi = (
                self.multi_condition_enable
                and self.num_dynamic_groups > 1
                and (self.group_sampling_rng.random() < self.multi_condition_prob)
                and self.multi_condition_max_k > 1
            )

            if do_multi:
                draw_present = bool(self.group_sampling_rng.random() < self.p_present_group)
                if draw_present and len(present) > 0:
                    pool = present
                else:
                    pool = negatives if len(negatives) > 0 else (present if len(present) > 0 else all_indices)

                max_k = int(min(self.multi_condition_max_k, len(pool), self.num_dynamic_groups))
                min_k = int(min(max(self.multi_condition_min_k, 2), max_k))

                if max_k > 0:
                    if min_k > max_k:
                        min_k = max_k
                    if min_k == max_k:
                        k = int(min_k)
                    else:
                        k = int(self.group_sampling_rng.integers(min_k, max_k + 1))

                    sel = self.group_sampling_rng.choice(
                        np.asarray(pool, dtype=np.int64),
                        size=int(k),
                        replace=False,
                    )
                    group_mask[i, torch.as_tensor(sel, dtype=torch.long, device=self.device)] = 1.0
                    continue

            group_id = sample_group_id_for_case(
                present_group_ids=present,
                p_present_group=self.p_present_group,
                rng=self.group_sampling_rng,
            )
            group_mask[i, int(group_id)] = 1.0

        return group_mask

    def _extract_group_condition_for_batch(self, batch: dict, target_high: torch.Tensor) -> torch.Tensor:
        condition = batch.get("group_mask", None)
        if condition is None:
            condition = batch.get("condition_mask", None)

        if condition is None:
            group_ids = batch.get("group_id", None)
            if group_ids is None:
                group_mask = self._sample_group_mask_from_target(target_high)
            else:
                if not torch.is_tensor(group_ids):
                    group_ids = torch.as_tensor(group_ids, dtype=torch.long)
                group_ids = group_ids.to(self.device, non_blocking=True).reshape(-1).long()
                if group_ids.numel() != int(target_high.shape[0]):
                    raise ValueError(
                        f"group_id batch mismatch: got {group_ids.numel()}, expected {int(target_high.shape[0])}"
                    )
                group_ids = group_ids.clamp(min=0, max=self.num_dynamic_groups - 1)
                group_mask = torch.zeros(
                    (int(target_high.shape[0]), self.num_dynamic_groups),
                    dtype=torch.float32,
                    device=self.device,
                )
                group_mask.scatter_(1, group_ids[:, None], 1.0)
        else:
            if not torch.is_tensor(condition):
                condition = torch.as_tensor(condition)
            condition = condition.to(self.device, non_blocking=True)

            if condition.ndim == 1:
                group_ids = condition.reshape(-1).long().clamp(min=0, max=self.num_dynamic_groups - 1)
                if group_ids.numel() != int(target_high.shape[0]):
                    raise ValueError(
                        f"condition batch mismatch: got {group_ids.numel()}, expected {int(target_high.shape[0])}"
                    )
                group_mask = torch.zeros(
                    (int(target_high.shape[0]), self.num_dynamic_groups),
                    dtype=torch.float32,
                    device=self.device,
                )
                group_mask.scatter_(1, group_ids[:, None], 1.0)
            elif condition.ndim == 2:
                if condition.shape[0] == 1 and int(target_high.shape[0]) > 1:
                    condition = condition.expand(int(target_high.shape[0]), -1)
                if condition.shape[0] != int(target_high.shape[0]):
                    raise ValueError(
                        f"condition batch mismatch: got {condition.shape[0]}, expected {int(target_high.shape[0])}"
                    )
                if condition.shape[1] != self.num_dynamic_groups:
                    raise ValueError(
                        f"condition width mismatch: got {condition.shape[1]}, expected {self.num_dynamic_groups}"
                    )
                group_mask = (condition > 0).float()
            else:
                raise ValueError(f"Unsupported condition ndim={condition.ndim}, expected 1 or 2")

        empty = group_mask.sum(dim=1) <= 0
        if empty.any():
            group_mask = group_mask.clone()
            group_mask[empty, 0] = 1.0

        return group_mask

    def _remap_target_for_group_condition(
        self,
        target,
        group_condition: torch.Tensor,
    ):
        if isinstance(target, list):
            remapped_targets = []
            valid_masks = []
            active_slots = None
            for t in target:
                remapped_t, valid_t, active_t = remap_original_to_structured_multi_condition(
                    t,
                    group_condition=group_condition,
                    ignore_label=self.label_manager.ignore_label,
                )
                remapped_targets.append(remapped_t)
                valid_masks.append(valid_t)
                if active_slots is None:
                    active_slots = active_t
            assert active_slots is not None
            return remapped_targets, valid_masks, active_slots

        remapped, valid, active_slots = remap_original_to_structured_multi_condition(
            target,
            group_condition=group_condition,
            ignore_label=self.label_manager.ignore_label,
        )
        return remapped, valid, active_slots

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_target(batch["target"], self.device)
        target_high = target[0] if isinstance(target, list) else target

        group_condition = self._extract_group_condition_for_batch(batch, target_high)
        picked = group_condition.detach().sum(dim=0).cpu().numpy()
        self._group_sample_counter_epoch += np.rint(picked).astype(np.int64)

        remapped_target, valid_mask, active_slots = self._remap_target_for_group_condition(target, group_condition)

        self.optimizer.zero_grad(set_to_none=True)
        amp_context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with amp_context:
            output = self.network(data, group_condition)
            output = self._sanitize_output(output)
            loss = self._compute_structured_loss(output, remapped_target, valid_mask, active_slots)

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
