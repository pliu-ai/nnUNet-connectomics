from __future__ import annotations

import json
import os
from typing import List, Tuple, Union

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from torch import autocast
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

from .dataloader_structured_conditional import StructuredConditionalDataLoader3D
from .inference_structured_conditional import (
    predict_logits_all_groups,
    predict_logits_for_group,
    reconstruct_original_labels_from_all_groups,
)
from .label_mapping import (
    NUM_DYNAMIC_GROUPS,
    NUM_OUTPUT_CHANNELS,
    infer_present_groups_from_segmentation,
    remap_original_to_structured,
    sample_group_id_for_case,
)
from .metrics_structured_conditional import (
    build_validation_report,
    compute_group_confusion_from_logits,
    empty_validation_accumulators,
)
from .network_structured_conditional import StructuredConditionalUNet, get_main_output
from .structured_loss import StructuredConditionalLoss, StructuredLossConfig


class nnUNetTrainerStructuredConditional(nnUNetTrainer):
    """
    Structured conditional trainer for CellMap with:
    - one shared model
    - fixed 11-channel output head
    - dynamic group-conditioned remapping
    - unified train/val/test workflow
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

        # Keep one fixed head independent of dataset label count.
        self.fixed_num_output_channels = NUM_OUTPUT_CHANNELS
        self.num_dynamic_groups = NUM_DYNAMIC_GROUPS

        # Group sampling during training.
        self.p_present_group = float(np.clip(float(os.environ.get("NNUNET_STRUCTCOND_P_PRESENT_GROUP", "0.8")), 0.0, 1.0))
        self.group_sampling_seed = int(os.environ.get("NNUNET_STRUCTCOND_GROUP_SEED", "1234"))
        self.group_sampling_rng = np.random.default_rng(self.group_sampling_seed)

        # Optimizer defaults: keep close to previously stable conditional training.
        self.initial_lr = float(os.environ.get("NNUNET_STRUCTCOND_INITIAL_LR", "0.001"))
        self.weight_decay = float(os.environ.get("NNUNET_STRUCTCOND_WEIGHT_DECAY", str(self.weight_decay)))
        if self.initial_lr <= 0:
            raise ValueError("NNUNET_STRUCTCOND_INITIAL_LR must be > 0.")
        if self.weight_decay < 0:
            raise ValueError("NNUNET_STRUCTCOND_WEIGHT_DECAY must be >= 0.")

        # Loss defaults are practical and can be tuned by env vars if needed.
        self.loss_cfg = StructuredLossConfig(
            lambda_ce=float(os.environ.get("NNUNET_STRUCTCOND_LAMBDA_CE", "1.0")),
            lambda_dice=float(os.environ.get("NNUNET_STRUCTCOND_LAMBDA_DICE", "1.0")),
            lambda_cond=float(os.environ.get("NNUNET_STRUCTCOND_LAMBDA_COND", "0.25")),
            lambda_suppress=float(os.environ.get("NNUNET_STRUCTCOND_LAMBDA_SUPPRESS", "0.1")),
            enable_conditional_focus=str(os.environ.get("NNUNET_STRUCTCOND_ENABLE_COND", "1")).lower() in {"1", "true", "yes", "y"},
            enable_suppression=str(os.environ.get("NNUNET_STRUCTCOND_ENABLE_SUPPRESS", "1")).lower() in {"1", "true", "yes", "y"},
            batch_dice=self.configuration_manager.batch_dice,
            smooth=1e-5,
            ddp=self.is_ddp,
        )

        self._ds_loss_weights = np.asarray([1.0], dtype=np.float32)
        self._latest_structured_val_report = {}
        self._group_sample_counter_epoch = np.zeros((self.num_dynamic_groups,), dtype=np.int64)
        self._val_group_cursor = int(self.local_rank) % max(1, self.num_dynamic_groups)
        self._val_step_counter = 0
        self._wandb_module = None
        self._wandb_run = None
        self._wandb_enabled = str(os.environ.get("NNUNET_USE_WANDB", "0")).lower() in {"1", "true", "yes", "y"}

        # Validation speed controls.
        self.val_full_sweep_every = int(os.environ.get("NNUNET_STRUCTCOND_VAL_FULL_EVERY", "1"))
        self.val_full_sweep_batches = int(os.environ.get("NNUNET_STRUCTCOND_VAL_FULL_SWEEP_BATCHES", "0"))
        self.val_full_sweep_epochs = int(os.environ.get("NNUNET_STRUCTCOND_VAL_FULL_SWEEP_EPOCHS", "0"))
        self.val_groups_per_epoch = int(os.environ.get("NNUNET_STRUCTCOND_VAL_GROUPS_PER_EPOCH", "12"))
        self.val_reuse_encoder = str(os.environ.get("NNUNET_STRUCTCOND_VAL_REUSE_ENCODER", "1")).lower() in {"1", "true", "yes", "y"}
        self.val_loss_mode = str(os.environ.get("NNUNET_STRUCTCOND_VAL_LOSS_MODE", "main_only")).strip().lower()
        if self.val_loss_mode not in {"full", "main_only", "none"}:
            raise ValueError("NNUNET_STRUCTCOND_VAL_LOSS_MODE must be one of: full, main_only, none")
        if self.val_groups_per_epoch < 1:
            raise ValueError("NNUNET_STRUCTCOND_VAL_GROUPS_PER_EPOCH must be >= 1")
        if self.val_full_sweep_batches < 0:
            raise ValueError("NNUNET_STRUCTCOND_VAL_FULL_SWEEP_BATCHES must be >= 0")
        if self.val_full_sweep_epochs < 0:
            raise ValueError("NNUNET_STRUCTCOND_VAL_FULL_SWEEP_EPOCHS must be >= 0")

    def _do_i_compile(self):
        enable = str(os.environ.get("NNUNET_STRUCTCOND_COMPILE", "0")).lower() in {"1", "true", "yes", "y"}
        if not enable:
            return False
        return super()._do_i_compile()

    def _setup_wandb(self) -> None:
        if not self._wandb_enabled or self.local_rank != 0:
            return
        try:
            import wandb  # type: ignore
        except Exception as e:
            self.print_to_log_file(f"[W&B] disabled because import failed: {e}")
            return

        run_id_file = os.path.join(self.output_folder, "wandb_run_id.txt")
        run_id = str(os.environ.get("WANDB_RUN_ID", "")).strip()
        if run_id == "" and os.path.isfile(run_id_file):
            try:
                with open(run_id_file, "r", encoding="utf-8") as f:
                    run_id = f.read().strip()
            except OSError:
                run_id = ""

        project = str(os.environ.get("WANDB_PROJECT", "nnUNet")).strip() or "nnUNet"
        entity = str(os.environ.get("WANDB_ENTITY", "")).strip() or None
        name = str(os.environ.get("WANDB_RUN_NAME", "")).strip()
        if name == "":
            name = (
                f"{self.__class__.__name__}_"
                f"{self.plans_manager.dataset_name}_"
                f"{self.configuration_name}_fold{self.fold}"
            )
        mode = str(os.environ.get("WANDB_MODE", "online")).strip() or "online"
        tags_env = str(os.environ.get("WANDB_TAGS", "")).strip()
        tags = [i.strip() for i in tags_env.split(",") if i.strip()] if tags_env else None

        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags,
            mode=mode,
            dir=self.output_folder,
            id=run_id if run_id != "" else None,
            resume="allow",
            config={
                "trainer": self.__class__.__name__,
                "dataset_name": self.plans_manager.dataset_name,
                "configuration": self.configuration_name,
                "fold": self.fold,
                "batch_size": self.batch_size,
                "num_iterations_per_epoch": self.num_iterations_per_epoch,
                "num_val_iterations_per_epoch": self.num_val_iterations_per_epoch,
                "num_epochs": self.num_epochs,
                "initial_lr": self.initial_lr,
                "weight_decay": self.weight_decay,
                "p_present_group": self.p_present_group,
                "val_full_every": self.val_full_sweep_every,
                "val_full_sweep_batches": self.val_full_sweep_batches,
                "val_full_sweep_epochs": self.val_full_sweep_epochs,
                "val_groups_per_epoch": self.val_groups_per_epoch,
                "val_reuse_encoder": self.val_reuse_encoder,
                "val_loss_mode": self.val_loss_mode,
            },
        )
        self._wandb_module = wandb
        self._wandb_run = run
        try:
            with open(run_id_file, "w", encoding="utf-8") as f:
                f.write(str(run.id))
        except OSError:
            pass
        run_url = getattr(run, "url", None)
        if run_url:
            self.print_to_log_file(f"[W&B] enabled. run_id={run.id}, url={run_url}")
        else:
            self.print_to_log_file(f"[W&B] enabled. run_id={run.id}")

    def _log_wandb_epoch(self) -> None:
        if self._wandb_run is None or self.local_rank != 0:
            return
        logs = self.logger.my_fantastic_logging
        if len(logs.get("train_losses", [])) == 0 or len(logs.get("val_losses", [])) == 0:
            return
        epoch_idx = int(self.current_epoch) - 1
        payload = {
            "epoch": epoch_idx,
            "train/loss": float(logs["train_losses"][-1]),
            "val/loss": float(logs["val_losses"][-1]),
            "val/mean_original31_dice": float(logs["mean_fg_dice"][-1]),
            "lr": float(logs["lrs"][-1]),
            "val/summary_mean_original31_dice": float(
                self._latest_structured_val_report.get("summary", {}).get("mean_original31_dice", logs["mean_fg_dice"][-1])
            ),
        }
        if (
            len(logs.get("epoch_start_timestamps", [])) > 0
            and len(logs.get("epoch_end_timestamps", [])) > 0
        ):
            payload["time/epoch_sec"] = float(
                logs["epoch_end_timestamps"][-1] - logs["epoch_start_timestamps"][-1]
            )
        self._wandb_run.log(payload, step=epoch_idx)

    def _finish_wandb(self) -> None:
        if self._wandb_run is not None and self.local_rank == 0:
            self._wandb_run.finish()
            self._wandb_run = None

    def initialize(self):
        """Custom initialize to enforce fixed output channels."""
        if self.was_initialized:
            raise RuntimeError("initialize() called more than once.")

        self.num_input_channels = determine_num_input_channels(
            self.plans_manager,
            self.configuration_manager,
            self.dataset_json,
        )

        self.network = self.build_network_architecture(
            self.configuration_manager.network_arch_class_name,
            self.configuration_manager.network_arch_init_kwargs,
            self.configuration_manager.network_arch_init_kwargs_req_import,
            self.num_input_channels,
            self.fixed_num_output_channels,
            self.enable_deep_supervision,
        ).to(self.device)

        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        if self.is_ddp:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            self.network = DDP(self.network, device_ids=[self.local_rank])

        self.loss = self._build_loss()
        self._ds_loss_weights = self._compute_deep_supervision_weights()
        self.was_initialized = True

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        # Enforce fixed output channels regardless of dataset labels.
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
        return StructuredConditionalUNet(
            backbone=backbone,
            num_groups=NUM_DYNAMIC_GROUPS,
            num_output_channels=NUM_OUTPUT_CHANNELS,
            cond_dim=64,
        )

    def _compute_deep_supervision_weights(self) -> np.ndarray:
        if not self.enable_deep_supervision:
            return np.asarray([1.0], dtype=np.float32)

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))], dtype=np.float32)

        if self.is_ddp:
            weights[-1] = 1e-6
        else:
            weights[-1] = 0.0

        weights = weights / np.clip(weights.sum(), a_min=1e-12, a_max=None)
        return weights.astype(np.float32)

    def _build_loss(self):
        return StructuredConditionalLoss(self.loss_cfg)

    @staticmethod
    def _to_device_target(target, device: torch.device):
        if isinstance(target, list):
            return [i.to(device, non_blocking=True) for i in target]
        return target.to(device, non_blocking=True)

    @staticmethod
    def _sanitize_output(output):
        if isinstance(output, (tuple, list)):
            return [torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4) for x in output]
        return torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

    def _sample_group_ids_from_target(self, target_high: torch.Tensor) -> torch.Tensor:
        sampled: List[int] = []
        for b in range(int(target_high.shape[0])):
            present = infer_present_groups_from_segmentation(
                target_high[b],
                ignore_label=self.label_manager.ignore_label,
            )
            group_id = sample_group_id_for_case(
                present_group_ids=sorted(present),
                p_present_group=self.p_present_group,
                rng=self.group_sampling_rng,
            )
            sampled.append(int(group_id))
        return torch.as_tensor(sampled, dtype=torch.long, device=self.device)

    def _extract_group_ids_for_batch(self, batch: dict, target_high: torch.Tensor) -> torch.Tensor:
        group_ids = batch.get("group_id", None)
        if group_ids is None:
            return self._sample_group_ids_from_target(target_high)

        if not torch.is_tensor(group_ids):
            group_ids = torch.as_tensor(group_ids, dtype=torch.long)
        group_ids = group_ids.to(self.device, non_blocking=True).reshape(-1).long()

        if group_ids.numel() != int(target_high.shape[0]):
            raise ValueError(
                f"group_id batch mismatch: got {group_ids.numel()}, expected {int(target_high.shape[0])}"
            )
        return group_ids.clamp(min=0, max=self.num_dynamic_groups - 1)

    def _remap_target_for_group(
        self,
        target,
        group_ids: torch.Tensor,
    ):
        if isinstance(target, list):
            remapped_targets = []
            valid_masks = []
            active_slots = None
            for t in target:
                remapped_t, valid_t, active_t = remap_original_to_structured(
                    t,
                    group_ids=group_ids,
                    ignore_label=self.label_manager.ignore_label,
                )
                remapped_targets.append(remapped_t)
                valid_masks.append(valid_t)
                if active_slots is None:
                    active_slots = active_t
            assert active_slots is not None
            return remapped_targets, valid_masks, active_slots

        remapped, valid, active_slots = remap_original_to_structured(
            target,
            group_ids=group_ids,
            ignore_label=self.label_manager.ignore_label,
        )
        return remapped, valid, active_slots

    def _compute_structured_loss(
        self,
        output,
        remapped_target,
        valid_mask,
        active_slots: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(output, (tuple, list)):
            if not isinstance(remapped_target, list) or not isinstance(valid_mask, list):
                raise RuntimeError("Deep supervision output requires list targets and masks.")

            total = output[0].sum() * 0.0
            n = min(len(output), len(remapped_target), len(valid_mask), len(self._ds_loss_weights))
            for i in range(n):
                weight = float(self._ds_loss_weights[i])
                if weight == 0.0:
                    continue
                total = total + weight * self.loss(output[i], remapped_target[i], valid_mask[i], active_slots)
            return total

        if isinstance(remapped_target, list) or isinstance(valid_mask, list):
            raise RuntimeError("Non-deep-supervision output received list target or mask.")
        return self.loss(output, remapped_target, valid_mask, active_slots)

    def _compute_structured_loss_main_only(
        self,
        output,
        remapped_target,
        valid_mask,
        active_slots: torch.Tensor,
    ) -> torch.Tensor:
        output_main = get_main_output(output)
        target_main = remapped_target[0] if isinstance(remapped_target, list) else remapped_target
        valid_main = valid_mask[0] if isinstance(valid_mask, list) else valid_mask
        return self.loss(output_main, target_main, valid_main, active_slots)

    def _unwrap_network(self):
        mod = self.network.module if self.is_ddp else self.network
        if hasattr(mod, "_orig_mod"):
            mod = mod._orig_mod
        return mod

    def _draw_val_group_ids(self, k: int) -> List[int]:
        k = int(max(1, min(k, self.num_dynamic_groups)))
        start = int(self._val_group_cursor)
        out = [int((start + j) % self.num_dynamic_groups) for j in range(k)]
        self._val_group_cursor = int((start + k) % self.num_dynamic_groups)
        return out

    def _get_val_group_ids(self) -> List[int]:
        if self.val_full_sweep_epochs > 0 and int(self.current_epoch) >= int(self.val_full_sweep_epochs):
            full_batch_limit = 0
        else:
            full_batch_limit = self.val_full_sweep_batches

        if full_batch_limit > 0 and self._val_step_counter < full_batch_limit:
            return list(range(self.num_dynamic_groups))
        if self.val_full_sweep_every <= 1:
            return list(range(self.num_dynamic_groups))
        if (int(self.current_epoch) % int(self.val_full_sweep_every)) == 0:
            return list(range(self.num_dynamic_groups))
        if self.val_groups_per_epoch >= self.num_dynamic_groups:
            return list(range(self.num_dynamic_groups))
        return self._draw_val_group_ids(self.val_groups_per_epoch)

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            # 2D fallback keeps default dataloader behavior; group IDs will be sampled in train_step.
            dl_tr = nnUNetDataLoader2D(
                dataset_tr,
                self.batch_size,
                initial_patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=tr_transforms,
            )
            dl_val = nnUNetDataLoader2D(
                dataset_val,
                self.batch_size,
                self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=val_transforms,
            )
        else:
            dl_tr = StructuredConditionalDataLoader3D(
                dataset_tr,
                self.batch_size,
                initial_patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=tr_transforms,
                p_present_group=self.p_present_group,
                seed=self.group_sampling_seed + int(self.local_rank),
            )
            dl_val = nnUNetDataLoader3D(
                dataset_val,
                self.batch_size,
                self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=val_transforms,
            )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )

        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def on_train_start(self):
        super().on_train_start()
        self._setup_wandb()
        self.print_to_log_file(
            "[StructuredConditional] "
            f"fixed_output_channels={self.fixed_num_output_channels}, "
            f"num_dynamic_groups={self.num_dynamic_groups}, "
            f"p_present_group={self.p_present_group:.3f}, "
            f"initial_lr={self.initial_lr}, "
            f"weight_decay={self.weight_decay}, "
            f"loss_cfg={self.loss_cfg}, "
            f"val_full_every={self.val_full_sweep_every}, "
            f"val_full_batches={self.val_full_sweep_batches}, "
            f"val_full_epochs={self.val_full_sweep_epochs}, "
            f"val_groups_per_epoch={self.val_groups_per_epoch}, "
            f"val_reuse_encoder={self.val_reuse_encoder}, "
            f"val_loss_mode={self.val_loss_mode}"
        )

    def on_validation_epoch_start(self):
        self._val_step_counter = 0
        super().on_validation_epoch_start()

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_target(batch["target"], self.device)
        target_high = target[0] if isinstance(target, list) else target

        group_ids = self._extract_group_ids_for_batch(batch, target_high)
        bincount = np.bincount(group_ids.detach().cpu().numpy(), minlength=self.num_dynamic_groups)
        self._group_sample_counter_epoch += bincount.astype(np.int64)

        remapped_target, valid_mask, active_slots = self._remap_target_for_group(target, group_ids)

        self.optimizer.zero_grad(set_to_none=True)
        amp_context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with amp_context:
            output = self.network(data, group_ids)
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

    def on_train_epoch_end(self, train_outputs: List[dict]):
        super().on_train_epoch_end(train_outputs)
        self.print_to_log_file(
            "[StructuredConditional] train group sample counts this epoch: "
            + json.dumps(self._group_sample_counter_epoch.tolist())
        )
        self._group_sample_counter_epoch[:] = 0

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = self._to_device_target(batch["target"], self.device)

        accum = empty_validation_accumulators()
        losses: List[float] = []

        self._val_step_counter += 1
        batch_size = int(data.shape[0])
        group_ids_list = self._get_val_group_ids()

        amp_context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        reuse_encoder = self.val_reuse_encoder
        skips = None
        if reuse_encoder:
            mod = self._unwrap_network()
            if hasattr(mod, "encode"):
                with amp_context:
                    skips = mod.encode(data)
            else:
                reuse_encoder = False

        for group_id in group_ids_list:
            group_ids = torch.full((batch_size,), int(group_id), dtype=torch.long, device=self.device)
            remapped_target, valid_mask, active_slots = self._remap_target_for_group(target, group_ids)

            with amp_context:
                if reuse_encoder and skips is not None:
                    output = self._unwrap_network().decode_from_skips(skips, group_ids)
                else:
                    output = self.network(data, group_ids)
                output = self._sanitize_output(output)
                if self.val_loss_mode == "full":
                    loss = self._compute_structured_loss(output, remapped_target, valid_mask, active_slots)
                elif self.val_loss_mode == "main_only":
                    loss = self._compute_structured_loss_main_only(output, remapped_target, valid_mask, active_slots)
                else:
                    loss = output[0].sum() * 0.0 if isinstance(output, list) else output.sum() * 0.0

            losses.append(float(loss.detach().cpu().item()))

            output_main = get_main_output(output)
            target_main = remapped_target[0] if isinstance(remapped_target, list) else remapped_target
            valid_main = valid_mask[0] if isinstance(valid_mask, list) else valid_mask

            (
                class_tp,
                class_fp,
                class_fn,
                cond_tp,
                cond_fp,
                cond_fn,
                merged_tp,
                merged_fp,
                merged_fn,
            ) = compute_group_confusion_from_logits(
                output_main,
                target_main,
                valid_main,
                group_id=group_id,
            )

            accum["class_tp"] += class_tp
            accum["class_fp"] += class_fp
            accum["class_fn"] += class_fn
            accum["cond_tp"][group_id] += cond_tp
            accum["cond_fp"][group_id] += cond_fp
            accum["cond_fn"][group_id] += cond_fn
            accum["merged_cond_tp"][group_id] += merged_tp[0]
            accum["merged_cond_fp"][group_id] += merged_fp[0]
            accum["merged_cond_fn"][group_id] += merged_fn[0]

        mean_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

        return {
            "loss": np.float32(mean_loss),
            "class_tp": accum["class_tp"],
            "class_fp": accum["class_fp"],
            "class_fn": accum["class_fn"],
            "cond_tp": accum["cond_tp"],
            "cond_fp": accum["cond_fp"],
            "cond_fn": accum["cond_fn"],
            "merged_cond_tp": accum["merged_cond_tp"],
            "merged_cond_fp": accum["merged_cond_fp"],
            "merged_cond_fn": accum["merged_cond_fn"],
        }

    @staticmethod
    def _ddp_sum_array(array_value: np.ndarray) -> np.ndarray:
        world_size = dist.get_world_size()
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, array_value)
        return np.stack(gathered, axis=0).sum(axis=0)

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs = collate_outputs(val_outputs)

        class_tp = np.sum(outputs["class_tp"], axis=0)
        class_fp = np.sum(outputs["class_fp"], axis=0)
        class_fn = np.sum(outputs["class_fn"], axis=0)
        cond_tp = np.sum(outputs["cond_tp"], axis=0)
        cond_fp = np.sum(outputs["cond_fp"], axis=0)
        cond_fn = np.sum(outputs["cond_fn"], axis=0)
        merged_cond_tp = np.sum(outputs["merged_cond_tp"], axis=0)
        merged_cond_fp = np.sum(outputs["merged_cond_fp"], axis=0)
        merged_cond_fn = np.sum(outputs["merged_cond_fn"], axis=0)

        if self.is_ddp:
            class_tp = self._ddp_sum_array(class_tp)
            class_fp = self._ddp_sum_array(class_fp)
            class_fn = self._ddp_sum_array(class_fn)
            cond_tp = self._ddp_sum_array(cond_tp)
            cond_fp = self._ddp_sum_array(cond_fp)
            cond_fn = self._ddp_sum_array(cond_fn)
            merged_cond_tp = self._ddp_sum_array(merged_cond_tp)
            merged_cond_fp = self._ddp_sum_array(merged_cond_fp)
            merged_cond_fn = self._ddp_sum_array(merged_cond_fn)

            losses_val = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_val, outputs["loss"])
            loss_here = float(np.vstack(losses_val).mean())
        else:
            loss_here = float(np.mean(outputs["loss"]))

        report = build_validation_report(
            class_tp=class_tp,
            class_fp=class_fp,
            class_fn=class_fn,
            cond_tp=cond_tp,
            cond_fp=cond_fp,
            cond_fn=cond_fn,
            merged_cond_tp=merged_cond_tp,
            merged_cond_fp=merged_cond_fp,
            merged_cond_fn=merged_cond_fn,
        )
        self._latest_structured_val_report = report

        mean_active_dice = float(report["summary"]["mean_original31_dice"])
        dice_foreground = report["original31_dice"]

        self.logger.log("mean_fg_dice", mean_active_dice, self.current_epoch)
        self.logger.log("dice_per_class_or_region", dice_foreground, self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)

        self.print_to_log_file("[StructuredConditional][val] " + json.dumps(report["summary"], sort_keys=True))

    def on_epoch_end(self):
        super().on_epoch_end()
        self._log_wandb_epoch()

    def on_train_end(self):
        try:
            super().on_train_end()
        finally:
            self._finish_wandb()

    @torch.no_grad()
    def infer_logits_for_group(self, image: torch.Tensor, group_id: int, use_amp: bool = True) -> torch.Tensor:
        self.network.eval()
        return predict_logits_for_group(self.network, image, group_id=group_id, use_amp=use_amp)

    @torch.no_grad()
    def infer_logits_all_groups(self, image: torch.Tensor, use_amp: bool = True):
        self.network.eval()
        return predict_logits_all_groups(self.network, image, use_amp=use_amp)

    @torch.no_grad()
    def infer_reconstruct_original_all_groups(self, image: torch.Tensor, use_amp: bool = True):
        self.network.eval()
        logits_by_group = predict_logits_all_groups(self.network, image, use_amp=use_amp)
        return reconstruct_original_labels_from_all_groups(logits_by_group)
