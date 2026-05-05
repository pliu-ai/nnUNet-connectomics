from __future__ import annotations

import os
from typing import List, Tuple, Union

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.dual_decoder_conditional_film_unet import (
    DualDecoderConditionalFiLMUNet,
)
from nnunetv2.training.nnUNetTrainer.variants.partial_label.nnUNetTrainerConditionalFiLM import (
    nnUNetTrainerConditionalFiLM,
)
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class _PolyPerGroupLRScheduler:
    """Poly LR scheduler with per-parameter-group base learning rates."""

    def __init__(self, optimizer: torch.optim.Optimizer, initial_lrs: List[float], max_steps: int, exponent: float = 0.9):
        self.optimizer = optimizer
        self.initial_lrs = [float(i) for i in initial_lrs]
        self.max_steps = int(max_steps)
        self.exponent = float(exponent)
        self.ctr = 0

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        factor = (1 - float(current_step) / float(self.max_steps)) ** self.exponent
        for base_lr, param_group in zip(self.initial_lrs, self.optimizer.param_groups):
            param_group["lr"] = base_lr * factor


class nnUNetTrainerConditionalFiLMDualDecoder(nnUNetTrainerConditionalFiLM):
    """
    Shared encoder + two decoders:
    - multiclass decoder for direct C-way segmentation
    - conditional FiLM binary decoder for condition-guided segmentation

    Training loss:
        total = w_multi * L_multi + w_binary * L_binary
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
        self.loss_weight_multiclass = float(os.environ.get("NNUNET_DUAL_LOSS_W_MULTI", "1.0"))
        self.loss_weight_binary = float(os.environ.get("NNUNET_DUAL_LOSS_W_BINARY", "1.0"))

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        num_conditions = max(1, int(num_output_channels) - 1)
        backbone_multiclass = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            int(num_output_channels),
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        backbone_binary = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            2,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return DualDecoderConditionalFiLMUNet(
            backbone_multiclass=backbone_multiclass,
            backbone_binary=backbone_binary,
            num_conditions=num_conditions,
            num_output_channels=int(num_output_channels),
            cond_dim=64,
        )

    def initialize(self):
        super().initialize()
        mod = self._unwrap_network()
        if isinstance(mod, DualDecoderConditionalFiLMUNet):
            mod.set_condition_label_values(self.condition_label_values)

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self._unwrap_network()
        if isinstance(mod, DualDecoderConditionalFiLMUNet):
            mod.decoder_multi.deep_supervision = enabled
            mod.decoder_binary.deep_supervision = enabled
            return
        super().set_deep_supervision_enabled(enabled)

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            f"[DualDecoder] loss_weight_multiclass={self.loss_weight_multiclass}, "
            f"loss_weight_binary={self.loss_weight_binary}"
        )

    def configure_optimizers(self):
        """
        Support separate initial LRs for dual decoders:
        - NNUNET_DUAL_LR_MULTI: decoder_multi LR
        - NNUNET_DUAL_LR_BINARY: decoder_binary LR
        - NNUNET_DUAL_LR_SHARED: shared/other params LR (optional, defaults to self.initial_lr)
        """
        mod = self._unwrap_network()
        if not isinstance(mod, DualDecoderConditionalFiLMUNet):
            return super().configure_optimizers()

        lr_multi = float(os.environ.get("NNUNET_DUAL_LR_MULTI", str(self.initial_lr)))
        lr_binary = float(os.environ.get("NNUNET_DUAL_LR_BINARY", str(self.initial_lr)))
        lr_shared = float(os.environ.get("NNUNET_DUAL_LR_SHARED", str(self.initial_lr)))
        if min(lr_multi, lr_binary, lr_shared) <= 0:
            raise ValueError("Dual decoder learning rates must be > 0")

        multi_param_ids = {id(p) for p in mod.decoder_multi.parameters() if p.requires_grad}
        binary_param_ids = {id(p) for p in mod.decoder_binary.parameters() if p.requires_grad}
        if len(multi_param_ids & binary_param_ids) > 0:
            raise RuntimeError("decoder_multi and decoder_binary unexpectedly share parameters")

        params_multi: List[torch.nn.Parameter] = []
        params_binary: List[torch.nn.Parameter] = []
        params_shared: List[torch.nn.Parameter] = []
        for p in mod.parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in multi_param_ids:
                params_multi.append(p)
            elif pid in binary_param_ids:
                params_binary.append(p)
            else:
                params_shared.append(p)

        param_groups = []
        group_lrs = []
        if len(params_shared) > 0:
            param_groups.append(
                {
                    "params": params_shared,
                    "lr": lr_shared,
                    "weight_decay": self.weight_decay,
                    "group_name": "shared",
                }
            )
            group_lrs.append(lr_shared)
        if len(params_multi) > 0:
            param_groups.append(
                {
                    "params": params_multi,
                    "lr": lr_multi,
                    "weight_decay": self.weight_decay,
                    "group_name": "decoder_multi",
                }
            )
            group_lrs.append(lr_multi)
        if len(params_binary) > 0:
            param_groups.append(
                {
                    "params": params_binary,
                    "lr": lr_binary,
                    "weight_decay": self.weight_decay,
                    "group_name": "decoder_binary",
                }
            )
            group_lrs.append(lr_binary)
        if len(param_groups) == 0:
            raise RuntimeError("No trainable parameters found for optimizer")

        optimizer = torch.optim.SGD(param_groups, momentum=0.99, nesterov=True)
        lr_scheduler = _PolyPerGroupLRScheduler(optimizer, group_lrs, self.num_epochs)

        self.print_to_log_file(
            f"[DualDecoderLR] shared={lr_shared}, multi={lr_multi}, binary={lr_binary}, "
            f"groups={len(param_groups)}"
        )
        return optimizer, lr_scheduler

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_target(batch["target"], self.device)

        target_high = target[0] if isinstance(target, list) else target
        cond_mask = self._extract_condition_mask_from_batch(batch, target_high)
        if isinstance(target, list):
            target_bin = [self._binaryize_target(t, cond_mask) for t in target]
        else:
            target_bin = self._binaryize_target(target, cond_mask)

        self.optimizer.zero_grad(set_to_none=True)
        with self._train_autocast():
            out = self.network(data, cond_mask, return_binary=True, return_multiclass=True)
            output_multi = self._sanitize_logits(out["multi"])
            output_bin = self._sanitize_logits(out["binary"])
            l_multi = self.loss(output_multi, target)
            l_bin = self.loss(output_bin, target_bin)
            l = (self.loss_weight_multiclass * l_multi) + (self.loss_weight_binary * l_bin)

        if not torch.isfinite(l).item():
            self._nonfinite_skip_count_train += 1
            if self._nonfinite_skip_count_train <= 20 or self._nonfinite_skip_count_train % 50 == 0:
                self.print_to_log_file(
                    f"[NonFinite][train] skipping step at epoch={self.current_epoch}, "
                    f"count={self._nonfinite_skip_count_train}"
                )
            self.optimizer.zero_grad(set_to_none=True)
            return {"loss": np.float32(0.0)}

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_target(batch["target"], self.device)

        target_high = target[0] if isinstance(target, list) else target
        cond_mask = self._extract_condition_mask_from_batch(batch, target_high)
        if isinstance(target, list):
            target_bin = [self._binaryize_target(t, cond_mask) for t in target]
        else:
            target_bin = self._binaryize_target(target, cond_mask)

        with self._train_autocast():
            out = self.network(data, cond_mask, return_binary=True, return_multiclass=True)
            output_multi = self._sanitize_logits(out["multi"])
            output_bin = self._sanitize_logits(out["binary"])
            l_multi = self.loss(output_multi, target)
            l_bin = self.loss(output_bin, target_bin)
            l = (self.loss_weight_multiclass * l_multi) + (self.loss_weight_binary * l_bin)

        if not torch.isfinite(l).item():
            self._nonfinite_skip_count_val += 1
            if self._nonfinite_skip_count_val <= 20 or self._nonfinite_skip_count_val % 50 == 0:
                self.print_to_log_file(
                    f"[NonFinite][val] forcing val loss to 0 at epoch={self.current_epoch}, "
                    f"count={self._nonfinite_skip_count_val}"
                )
            l = torch.zeros((), dtype=torch.float32, device=self.device)

        output_for_metric = self._get_main_output(output_multi) if self.enable_deep_supervision else output_multi
        target_for_metric = target[0] if isinstance(target, list) else target

        if self.label_manager.has_ignore_label:
            ignore_label = int(self.label_manager.ignore_label)
            valid_mask = target_for_metric != ignore_label
            target_for_metric = target_for_metric.clone()
            target_for_metric[~valid_mask] = 0
        else:
            valid_mask = torch.ones_like(target_for_metric, dtype=torch.bool)

        tp_hard, fp_hard, fn_hard = self._compute_classwise_stats_full_sweep(
            output_for_metric,
            target_for_metric,
            valid_mask,
        )
        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }
