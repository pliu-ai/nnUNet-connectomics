from __future__ import annotations

import os
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from torch import autocast
from torch import distributed as dist
from torch._dynamo import OptimizedModule

from nnunetv2.configuration import get_allowed_n_proc_DA
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.conditional_film_unet import (
    ConditionalFiLMUNet,
)
from nnunetv2.training.nnUNetTrainer.variants.partial_label.condition_aware_dataloader_3d import (
    ConditionAwareDataLoader3D,
)
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.collate_outputs import collate_outputs


class nnUNetTrainerConditionalFiLM(nnUNetTrainer):
    """
    Conditional FiLM trainer:
    - training: binary prediction conditioned on sampled class id
    - inference/validation (condition=None): condition sweep to multiclass logits
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
        self.condition_positive_prob = 0.7
        self.condition_label_values: List[int] = self._extract_condition_labels(dataset_json)
        self.num_conditions = len(self.condition_label_values)
        self.label_to_condition_index = {int(v): i for i, v in enumerate(self.condition_label_values)}
        self._cond_label_tensor_cpu = torch.as_tensor(self.condition_label_values, dtype=torch.long)
        print(
            f"[ConditionalFiLM] num_conditions={self.num_conditions} "
            f"labels={self.condition_label_values[:6]}{'...' if self.num_conditions > 6 else ''}"
        )
        self._wandb_module = None
        self._wandb_run = None
        self._wandb_enabled = (
            str(os.environ.get("NNUNET_USE_WANDB", "0")).lower() in {"1", "true", "yes", "y"}
        )
        self._use_amp = str(os.environ.get("NNUNET_COND_USE_AMP", "0")).lower() in {"1", "true", "yes", "y"}
        env_batch_size = str(os.environ.get("NNUNET_COND_BATCH_SIZE", "")).strip()
        if env_batch_size != "":
            try:
                batch_size_override = int(env_batch_size)
            except ValueError as e:
                raise ValueError("NNUNET_COND_BATCH_SIZE must be a positive integer") from e
            if batch_size_override < 1:
                raise ValueError("NNUNET_COND_BATCH_SIZE must be >= 1")
            # Update planned global batch size and recompute per-rank batch size in DDP.
            self.configuration_manager.configuration["batch_size"] = batch_size_override
            self._set_batch_size_and_oversample()
            self.print_to_log_file(
                f"[ConditionalFiLM] batch size override enabled: global={batch_size_override}, local={self.batch_size}"
            )
        self.initial_lr = float(os.environ.get("NNUNET_COND_INITIAL_LR", "0.001"))
        self.weight_decay = float(os.environ.get("NNUNET_COND_WEIGHT_DECAY", str(self.weight_decay)))
        self.max_grad_norm = float(os.environ.get("NNUNET_COND_MAX_GRAD_NORM", "12.0"))
        self.num_epochs = int(os.environ.get("NNUNET_COND_NUM_EPOCHS", str(self.num_epochs)))
        if self.num_epochs < 1:
            raise ValueError("NNUNET_COND_NUM_EPOCHS must be >= 1")
        self.save_every = int(os.environ.get("NNUNET_COND_SAVE_EVERY", str(self.save_every)))
        if self.save_every < 1:
            raise ValueError("NNUNET_COND_SAVE_EVERY must be >= 1")
        self.num_iterations_per_epoch = int(
            os.environ.get("NNUNET_COND_NUM_ITERS_PER_EPOCH", str(self.num_iterations_per_epoch))
        )
        if self.num_iterations_per_epoch < 1:
            raise ValueError("NNUNET_COND_NUM_ITERS_PER_EPOCH must be >= 1")
        self.condition_sampling_strategy = str(
            os.environ.get("NNUNET_COND_SAMPLING_STRATEGY", "legacy")
        ).strip().lower()
        if self.condition_sampling_strategy not in {"legacy", "uniform_cycle"}:
            raise ValueError(
                f"Unknown NNUNET_COND_SAMPLING_STRATEGY={self.condition_sampling_strategy}. "
                f"Supported: legacy, uniform_cycle"
            )
        self._uniform_cond_cursor = int(getattr(self, "local_rank", 0)) % max(1, self.num_conditions)
        self._condition_sample_counter = np.zeros((self.num_conditions,), dtype=np.int64)
        self.multi_condition_enable = (
            str(os.environ.get("NNUNET_COND_MULTI_ENABLE", "0")).lower() in {"1", "true", "yes", "y"}
        )
        self.multi_condition_prob = float(os.environ.get("NNUNET_COND_MULTI_PROB", "0.5"))
        self.multi_condition_min_k = int(os.environ.get("NNUNET_COND_MULTI_MIN_K", "2"))
        self.multi_condition_max_k = int(os.environ.get("NNUNET_COND_MULTI_MAX_K", str(self.num_conditions)))
        self.multi_condition_prob = float(np.clip(self.multi_condition_prob, 0.0, 1.0))
        self.multi_condition_min_k = max(2, self.multi_condition_min_k)
        self.multi_condition_max_k = max(self.multi_condition_min_k, self.multi_condition_max_k)
        self.class_aware_patch = (
            str(os.environ.get("NNUNET_COND_CLASS_AWARE_PATCH", "0")).lower() in {"1", "true", "yes", "y"}
        )
        self.class_aware_resample_tries = int(os.environ.get("NNUNET_COND_CLASS_AWARE_RESAMPLE_TRIES", "8"))
        self.class_aware_resample_tries = max(0, self.class_aware_resample_tries)
        self.class_aware_fallback = str(
            os.environ.get("NNUNET_COND_CLASS_AWARE_FALLBACK", "legacy_fg")
        ).strip().lower()
        if self.class_aware_fallback not in {"legacy_fg", "random"}:
            raise ValueError(
                f"Unknown NNUNET_COND_CLASS_AWARE_FALLBACK={self.class_aware_fallback}. "
                f"Supported: legacy_fg, random"
            )

        self.pseudo_dice_eval_mode = str(
            os.environ.get("NNUNET_COND_PSEUDO_DICE_MODE", "condition_sampled")
        ).strip().lower()
        if self.pseudo_dice_eval_mode not in {"full_sweep", "condition_sampled"}:
            raise ValueError(
                f"Unknown NNUNET_COND_PSEUDO_DICE_MODE={self.pseudo_dice_eval_mode}. "
                f"Supported: full_sweep, condition_sampled"
            )
        self.val_repeat_per_condition = int(os.environ.get("NNUNET_COND_VAL_REPEAT_PER_CONDITION", "2"))
        self.val_repeat_per_condition = max(1, self.val_repeat_per_condition)

        self.val_condition_sampling_strategy = str(
            os.environ.get("NNUNET_COND_VAL_SAMPLING_STRATEGY", "uniform_cycle")
        ).strip().lower()
        if self.val_condition_sampling_strategy not in {"legacy", "uniform_cycle"}:
            raise ValueError(
                f"Unknown NNUNET_COND_VAL_SAMPLING_STRATEGY={self.val_condition_sampling_strategy}. "
                f"Supported: legacy, uniform_cycle"
            )
        self.val_condition_positive_prob = float(os.environ.get("NNUNET_COND_VAL_POSITIVE_PROB", "1.0"))
        self.val_condition_positive_prob = float(np.clip(self.val_condition_positive_prob, 0.0, 1.0))
        self.val_multi_condition_enable = (
            str(os.environ.get("NNUNET_COND_VAL_MULTI_ENABLE", "0")).lower() in {"1", "true", "yes", "y"}
        )
        self.val_multi_condition_prob = float(os.environ.get("NNUNET_COND_VAL_MULTI_PROB", "0.0"))
        self.val_multi_condition_prob = float(np.clip(self.val_multi_condition_prob, 0.0, 1.0))
        self.val_multi_condition_min_k = int(
            os.environ.get("NNUNET_COND_VAL_MULTI_MIN_K", str(self.multi_condition_min_k))
        )
        self.val_multi_condition_max_k = int(
            os.environ.get("NNUNET_COND_VAL_MULTI_MAX_K", str(self.multi_condition_max_k))
        )
        self.val_multi_condition_min_k = max(2, self.val_multi_condition_min_k)
        self.val_multi_condition_max_k = max(self.val_multi_condition_min_k, self.val_multi_condition_max_k)
        self.val_class_aware_patch = (
            str(os.environ.get("NNUNET_COND_VAL_CLASS_AWARE_PATCH", "1")).lower() in {"1", "true", "yes", "y"}
        )
        self.val_class_aware_resample_tries = int(
            os.environ.get(
                "NNUNET_COND_VAL_CLASS_AWARE_RESAMPLE_TRIES",
                str(self.class_aware_resample_tries),
            )
        )
        self.val_class_aware_resample_tries = max(0, self.val_class_aware_resample_tries)
        self.val_class_aware_fallback = str(
            os.environ.get("NNUNET_COND_VAL_CLASS_AWARE_FALLBACK", self.class_aware_fallback)
        ).strip().lower()
        if self.val_class_aware_fallback not in {"legacy_fg", "random"}:
            raise ValueError(
                f"Unknown NNUNET_COND_VAL_CLASS_AWARE_FALLBACK={self.val_class_aware_fallback}. "
                f"Supported: legacy_fg, random"
            )

        env_num_val_iters = os.environ.get("NNUNET_COND_NUM_VAL_ITERS_PER_EPOCH", None)
        if env_num_val_iters is not None and str(env_num_val_iters).strip() != "":
            self.num_val_iterations_per_epoch = int(env_num_val_iters)
        elif self.pseudo_dice_eval_mode == "condition_sampled":
            total_cond_samples = self.num_conditions * self.val_repeat_per_condition
            self.num_val_iterations_per_epoch = int(np.ceil(total_cond_samples / max(1, self.batch_size)))
        if self.num_val_iterations_per_epoch < 1:
            raise ValueError("NNUNET_COND_NUM_VAL_ITERS_PER_EPOCH must be >= 1")

        if not self._use_amp:
            self.grad_scaler = None
        self._nonfinite_skip_count_train = 0
        self._nonfinite_skip_count_val = 0

    def _do_i_compile(self):
        # Custom conditional forward currently favors stability over torch.compile speedups.
        return False

    def _setup_wandb(self):
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
            name = f"{self.__class__.__name__}_{self.plans_manager.dataset_name}_{self.configuration_name}_fold{self.fold}"
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
                "condition_positive_prob": self.condition_positive_prob,
                "num_conditions": self.num_conditions,
                "condition_sampling_strategy": self.condition_sampling_strategy,
                "multi_condition_enable": self.multi_condition_enable,
                "multi_condition_prob": self.multi_condition_prob,
                "multi_condition_min_k": self.multi_condition_min_k,
                "multi_condition_max_k": self.multi_condition_max_k,
                "class_aware_patch": self.class_aware_patch,
                "class_aware_resample_tries": self.class_aware_resample_tries,
                "class_aware_fallback": self.class_aware_fallback,
                "pseudo_dice_eval_mode": self.pseudo_dice_eval_mode,
                "val_repeat_per_condition": self.val_repeat_per_condition,
                "val_condition_sampling_strategy": self.val_condition_sampling_strategy,
                "val_condition_positive_prob": self.val_condition_positive_prob,
                "val_multi_condition_enable": self.val_multi_condition_enable,
                "val_multi_condition_prob": self.val_multi_condition_prob,
                "val_multi_condition_min_k": self.val_multi_condition_min_k,
                "val_multi_condition_max_k": self.val_multi_condition_max_k,
                "val_class_aware_patch": self.val_class_aware_patch,
                "val_class_aware_resample_tries": self.val_class_aware_resample_tries,
                "val_class_aware_fallback": self.val_class_aware_fallback,
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

    def _sanitize_logits(self, out):
        if isinstance(out, (tuple, list)):
            return [torch.nan_to_num(i, nan=0.0, posinf=1e4, neginf=-1e4) for i in out]
        return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

    def _train_autocast(self):
        return autocast(self.device.type, enabled=self._use_amp) if self.device.type == "cuda" else dummy_context()

    def _log_wandb_epoch(self):
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
            "val/mean_fg_dice": float(logs["mean_fg_dice"][-1]),
            "lr": float(logs["lrs"][-1]),
            "stability/nonfinite_skipped_train": int(self._nonfinite_skip_count_train),
            "stability/nonfinite_val_forced_zero": int(self._nonfinite_skip_count_val),
        }
        if (
            len(logs.get("epoch_start_timestamps", [])) > 0
            and len(logs.get("epoch_end_timestamps", [])) > 0
        ):
            payload["time/epoch_sec"] = float(
                logs["epoch_end_timestamps"][-1] - logs["epoch_start_timestamps"][-1]
            )
        self._wandb_run.log(payload, step=epoch_idx)

    def _finish_wandb(self):
        if self._wandb_run is not None and self.local_rank == 0:
            self._wandb_run.finish()
            self._wandb_run = None

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            f"[Stability] initial_lr={self.initial_lr}, weight_decay={self.weight_decay}, "
            f"use_amp={self._use_amp}, max_grad_norm={self.max_grad_norm}, "
            f"num_epochs={self.num_epochs}, save_every={self.save_every}"
        )
        self.print_to_log_file(
            f"[Conditional] sampling={self.condition_sampling_strategy}, "
            f"multi_enable={self.multi_condition_enable}, "
            f"multi_prob={self.multi_condition_prob}, "
            f"multi_k=[{self.multi_condition_min_k},{self.multi_condition_max_k}], "
            f"class_aware_patch={self.class_aware_patch}, "
            f"class_aware_resample_tries={self.class_aware_resample_tries}, "
            f"class_aware_fallback={self.class_aware_fallback}"
        )
        self.print_to_log_file(
            f"[ValPseudoDice] mode={self.pseudo_dice_eval_mode}, "
            f"repeat_per_condition={self.val_repeat_per_condition}, "
            f"num_val_iterations_per_epoch={self.num_val_iterations_per_epoch}, "
            f"val_sampling={self.val_condition_sampling_strategy}, "
            f"val_positive_prob={self.val_condition_positive_prob}, "
            f"val_multi_enable={self.val_multi_condition_enable}, "
            f"val_class_aware_patch={self.val_class_aware_patch}, "
            f"val_class_aware_resample_tries={self.val_class_aware_resample_tries}, "
            f"val_class_aware_fallback={self.val_class_aware_fallback}"
        )
        self._setup_wandb()

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.local_rank == 0 and self._condition_sample_counter.size > 0:
            cmin = int(self._condition_sample_counter.min())
            cmax = int(self._condition_sample_counter.max())
            cmean = float(self._condition_sample_counter.mean())
            self.print_to_log_file(
                f"[ConditionalSampling] strategy={self.condition_sampling_strategy}, "
                f"cumulative_counts(min/mean/max)=({cmin}/{cmean:.2f}/{cmax})"
            )
        self._log_wandb_epoch()

    def on_train_end(self):
        try:
            super().on_train_end()
        finally:
            self._finish_wandb()

    @staticmethod
    def _extract_condition_labels(dataset_json: dict) -> List[int]:
        labels = dataset_json.get("labels", {})
        vals: List[int] = []
        for name, value in labels.items():
            key = str(name).strip().lower()
            if key in {"background", "ignore"}:
                continue
            vals.append(int(value))
        vals = sorted(set(vals))
        if len(vals) == 0:
            raise RuntimeError("No foreground labels found in dataset_json; cannot build conditional trainer.")
        return vals

    def _unwrap_network(self):
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    def initialize(self):
        super().initialize()
        mod = self._unwrap_network()
        if isinstance(mod, ConditionalFiLMUNet):
            mod.set_condition_label_values(self.condition_label_values)
            if mod.num_conditions != self.num_conditions:
                raise RuntimeError(
                    f"Condition count mismatch between trainer ({self.num_conditions}) and network ({mod.num_conditions})"
                )

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> torch.nn.Module:
        # Binary conditional head (bg/fg) + FiLM modulation.
        binary_output_channels = 2
        num_conditions = max(1, int(num_output_channels) - 1)
        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            binary_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return ConditionalFiLMUNet(
            backbone=backbone,
            num_conditions=num_conditions,
            num_output_channels=int(num_output_channels),
            cond_dim=64,
        )

    def _build_loss(self):
        loss = DC_and_CE_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))], dtype=np.float32)
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _sample_condition_mask_legacy(self, target: torch.Tensor) -> torch.Tensor:
        """
        target: [B, 1, ...] integer label map at highest resolution.
        returns: [B, num_conditions] multi-hot condition mask.
        """
        b = int(target.shape[0])
        cond_mask = torch.zeros((b, self.num_conditions), dtype=torch.float32, device=target.device)
        ignore_label = self.label_manager.ignore_label
        all_indices = list(range(self.num_conditions))

        for i in range(b):
            t = target[i]
            if ignore_label is None:
                valid = torch.ones_like(t, dtype=torch.bool)
            else:
                valid = t != int(ignore_label)
            vals = torch.unique(t[valid]).tolist() if valid.any() else []
            present_idx = [
                self.label_to_condition_index[int(v)]
                for v in vals
                if int(v) in self.label_to_condition_index
            ]
            present_set = set(present_idx)
            negatives = [k for k in all_indices if k not in present_set]

            do_multi = (
                self.multi_condition_enable
                and self.num_conditions > 1
                and (torch.rand((), device=target.device).item() < self.multi_condition_prob)
            )

            if do_multi:
                do_positive = bool(torch.rand((), device=target.device).item() < self.condition_positive_prob)
                if do_positive and len(present_idx) > 0:
                    pool = present_idx
                else:
                    pool = negatives if len(negatives) > 0 else (present_idx if len(present_idx) > 0 else all_indices)

                max_k = min(self.multi_condition_max_k, len(pool), self.num_conditions)
                min_k = min(max(self.multi_condition_min_k, 2), max_k)
                if max_k <= 0:
                    choice = int(torch.randint(self.num_conditions, (1,), device=target.device).item())
                    cond_mask[i, choice] = 1.0
                    continue
                if min_k > max_k:
                    min_k = max_k
                if min_k == max_k:
                    k = int(min_k)
                else:
                    k = int(torch.randint(min_k, max_k + 1, (1,), device=target.device).item())
                pool_tensor = torch.as_tensor(pool, dtype=torch.long, device=target.device)
                sel = pool_tensor[torch.randperm(pool_tensor.numel(), device=target.device)[:k]]
                cond_mask[i, sel] = 1.0
                continue

            if len(present_idx) > 0:
                do_positive = bool(torch.rand((), device=target.device).item() < self.condition_positive_prob)
                if do_positive:
                    choice = present_idx[int(torch.randint(len(present_idx), (1,), device=target.device).item())]
                else:
                    pool = negatives if len(negatives) > 0 else present_idx
                    choice = pool[int(torch.randint(len(pool), (1,), device=target.device).item())]
            else:
                choice = int(torch.randint(self.num_conditions, (1,), device=target.device).item())
            cond_mask[i, int(choice)] = 1.0

        return cond_mask

    def _draw_uniform_cycle_indices(self, k: int) -> List[int]:
        if self.num_conditions <= 0:
            return []
        k = int(max(1, min(k, self.num_conditions)))
        start = int(self._uniform_cond_cursor)
        out = [int((start + j) % self.num_conditions) for j in range(k)]
        self._uniform_cond_cursor = int((start + k) % self.num_conditions)
        return out

    def _sample_condition_mask_uniform_cycle(self, target: torch.Tensor) -> torch.Tensor:
        b = int(target.shape[0])
        cond_mask = torch.zeros((b, self.num_conditions), dtype=torch.float32, device=target.device)
        for i in range(b):
            do_multi = (
                self.multi_condition_enable
                and self.num_conditions > 1
                and (torch.rand((), device=target.device).item() < self.multi_condition_prob)
            )
            if do_multi:
                max_k = min(self.multi_condition_max_k, self.num_conditions)
                min_k = min(max(self.multi_condition_min_k, 2), max_k)
                if min_k > max_k:
                    min_k = max_k
                if min_k == max_k:
                    k = int(min_k)
                else:
                    k = int(torch.randint(min_k, max_k + 1, (1,), device=target.device).item())
            else:
                k = 1
            sel = self._draw_uniform_cycle_indices(k)
            cond_mask[i, sel] = 1.0
        return cond_mask

    def _sample_condition_mask(self, target: torch.Tensor) -> torch.Tensor:
        if self.condition_sampling_strategy == "uniform_cycle":
            cond_mask = self._sample_condition_mask_uniform_cycle(target)
        else:
            cond_mask = self._sample_condition_mask_legacy(target)
        self._register_condition_counter(cond_mask)
        return cond_mask

    def _register_condition_counter(self, cond_mask: torch.Tensor) -> None:
        if cond_mask.numel() > 0:
            self._condition_sample_counter += cond_mask.sum(dim=0).detach().cpu().numpy().astype(np.int64)

    def _extract_condition_mask_from_batch(self, batch: dict, target_high: torch.Tensor) -> torch.Tensor:
        cond_mask_b = batch.get("condition_mask", None)
        if cond_mask_b is None:
            return self._sample_condition_mask(target_high)
        cond_mask = cond_mask_b.to(self.device, non_blocking=True).float()
        self._register_condition_counter(cond_mask)
        return cond_mask

    def _binaryize_target(self, target: torch.Tensor, cond_mask: torch.Tensor) -> torch.Tensor:
        """
        target: [B, 1, ...] integer class labels
        cond_mask: [B, num_conditions] multi-hot condition vectors
        """
        b = int(target.shape[0])
        labels = self._cond_label_tensor_cpu.to(target.device)
        binary = torch.zeros_like(target, dtype=torch.long)
        for i in range(b):
            idx = torch.nonzero(cond_mask[i] > 0, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            sel_labels = labels[idx]
            fg = torch.isin(target[i], sel_labels)
            binary[i] = fg.long()
        ignore_label = self.label_manager.ignore_label
        if ignore_label is not None:
            ignore = target == int(ignore_label)
            binary = torch.where(ignore, torch.full_like(binary, int(ignore_label)), binary)
        return binary

    @staticmethod
    def _to_device_target(target, device: torch.device):
        if isinstance(target, list):
            return [i.to(device, non_blocking=True) for i in target]
        return target.to(device, non_blocking=True)

    @staticmethod
    def _get_main_output(out):
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def _compute_classwise_stats_full_sweep(
        self,
        output_multi: torch.Tensor,
        target_for_metric: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_fg = int(output_multi.shape[1]) - 1
        tp_hard = np.zeros((num_fg,), dtype=np.float64)
        fp_hard = np.zeros((num_fg,), dtype=np.float64)
        fn_hard = np.zeros((num_fg,), dtype=np.float64)
        for cls in range(1, num_fg + 1):
            pred_cls = output_multi[:, cls:cls + 1] > 0
            gt_cls = target_for_metric == cls
            pred_cls = pred_cls & valid_mask
            gt_cls = gt_cls & valid_mask
            tp_hard[cls - 1] = (pred_cls & gt_cls).sum().item()
            fp_hard[cls - 1] = (pred_cls & (~gt_cls)).sum().item()
            fn_hard[cls - 1] = ((~pred_cls) & gt_cls).sum().item()
        return tp_hard, fp_hard, fn_hard

    def _compute_classwise_stats_condition_sampled(
        self,
        output_bin: torch.Tensor,
        target_for_metric: torch.Tensor,
        valid_mask: torch.Tensor,
        cond_mask: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_fg = self.num_conditions
        tp_hard = np.zeros((num_fg,), dtype=np.float64)
        fp_hard = np.zeros((num_fg,), dtype=np.float64)
        fn_hard = np.zeros((num_fg,), dtype=np.float64)

        pred_fg = output_bin[:, 1:2] > output_bin[:, 0:1]
        cond_labels = self._cond_label_tensor_cpu.to(target_for_metric.device)

        for b in range(int(target_for_metric.shape[0])):
            cond_idx = torch.nonzero(cond_mask[b] > 0, as_tuple=False).flatten()
            if cond_idx.numel() == 0:
                continue
            for idx in cond_idx.tolist():
                cls_label = int(cond_labels[int(idx)].item())
                pred_cls = pred_fg[b : b + 1] & valid_mask[b : b + 1]
                gt_cls = (target_for_metric[b : b + 1] == cls_label) & valid_mask[b : b + 1]
                tp_hard[int(idx)] += (pred_cls & gt_cls).sum().item()
                fp_hard[int(idx)] += (pred_cls & (~gt_cls)).sum().item()
                fn_hard[int(idx)] += ((~pred_cls) & gt_cls).sum().item()
        return tp_hard, fp_hard, fn_hard

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
            output_bin = self.network(data, cond_mask)
            output_bin = self._sanitize_logits(output_bin)
            l = self.loss(output_bin, target_bin)

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
            output_bin = self.network(data, cond_mask)
            output_bin = self._sanitize_logits(output_bin)
            l = self.loss(output_bin, target_bin)
            output_multi = None
            if self.pseudo_dice_eval_mode == "full_sweep":
                output_multi = self.network(data, None)
                output_multi = self._sanitize_logits(output_multi)

        if not torch.isfinite(l).item():
            self._nonfinite_skip_count_val += 1
            if self._nonfinite_skip_count_val <= 20 or self._nonfinite_skip_count_val % 50 == 0:
                self.print_to_log_file(
                    f"[NonFinite][val] forcing val loss to 0 at epoch={self.current_epoch}, "
                    f"count={self._nonfinite_skip_count_val}"
                )
            l = torch.zeros((), dtype=torch.float32, device=self.device)

        if self.enable_deep_supervision:
            output_bin_for_metric = self._get_main_output(output_bin)
            output_multi_for_metric = self._get_main_output(output_multi) if output_multi is not None else None
        else:
            output_bin_for_metric = output_bin
            output_multi_for_metric = output_multi
        target_for_metric = target[0] if isinstance(target, list) else target

        if self.label_manager.has_ignore_label:
            ignore_label = int(self.label_manager.ignore_label)
            valid_mask = target_for_metric != ignore_label
            target_for_metric = target_for_metric.clone()
            target_for_metric[~valid_mask] = 0
        else:
            valid_mask = torch.ones_like(target_for_metric, dtype=torch.bool)

        if self.pseudo_dice_eval_mode == "full_sweep":
            if output_multi_for_metric is None:
                raise RuntimeError("output_multi_for_metric is None under full_sweep mode")
            tp_hard, fp_hard, fn_hard = self._compute_classwise_stats_full_sweep(
                output_multi_for_metric,
                target_for_metric,
                valid_mask,
            )
        else:
            tp_hard, fp_hard, fn_hard = self._compute_classwise_stats_condition_sampled(
                output_bin_for_metric,
                target_for_metric,
                valid_mask,
                cond_mask,
            )

        return {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated["tp_hard"], 0)
        fp = np.sum(outputs_collated["fp_hard"], 0)
        fn = np.sum(outputs_collated["fn_hard"], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated["loss"])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated["loss"])

        denom = 2 * tp + fp + fn
        global_dc_per_class = np.divide(
            2 * tp,
            denom,
            out=np.zeros_like(tp, dtype=np.float64),
            where=denom > 0,
        ).tolist()
        mean_fg_dice = float(np.mean(global_dc_per_class)) if len(global_dc_per_class) > 0 else 0.0

        self.logger.log("mean_fg_dice", mean_fg_dice, self.current_epoch)
        self.logger.log("dice_per_class_or_region", global_dc_per_class, self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)

    def save_checkpoint(self, filename: str) -> None:
        # Periodic checkpointing from base trainer uses checkpoint_latest.pth and overwrites.
        # Keep epoch snapshots instead: checkpoint_ep50.pth, checkpoint_ep100.pth, ...
        if filename.endswith("checkpoint_latest.pth"):
            epoch_number = int(self.current_epoch) + 1
            snapshot = os.path.join(self.output_folder, f"checkpoint_ep{epoch_number}.pth")
            return super().save_checkpoint(snapshot)
        return super().save_checkpoint(filename)

    def get_dataloaders(self):
        use_conditional_train_loader = self.class_aware_patch
        use_conditional_val_loader = self.val_class_aware_patch or (self.pseudo_dice_eval_mode == "condition_sampled")

        if not use_conditional_train_loader and not use_conditional_val_loader:
            return super().get_dataloaders()

        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        if dim == 2:
            # Keep 2D behavior unchanged.
            return super().get_dataloaders()

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

        if use_conditional_train_loader:
            dl_tr = ConditionAwareDataLoader3D(
                dataset_tr,
                self.batch_size,
                initial_patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=tr_transforms,
                num_conditions=self.num_conditions,
                condition_label_values=self.condition_label_values,
                label_to_condition_index=self.label_to_condition_index,
                condition_sampling_strategy=self.condition_sampling_strategy,
                condition_positive_prob=self.condition_positive_prob,
                multi_condition_enable=self.multi_condition_enable,
                multi_condition_prob=self.multi_condition_prob,
                multi_condition_min_k=self.multi_condition_min_k,
                multi_condition_max_k=self.multi_condition_max_k,
                class_aware_resample_tries=self.class_aware_resample_tries,
                class_aware_fallback=self.class_aware_fallback,
            )
        else:
            dl_tr = nnUNetDataLoader3D(
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

        if use_conditional_val_loader:
            dl_val = ConditionAwareDataLoader3D(
                dataset_val,
                self.batch_size,
                self.configuration_manager.patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=val_transforms,
                num_conditions=self.num_conditions,
                condition_label_values=self.condition_label_values,
                label_to_condition_index=self.label_to_condition_index,
                condition_sampling_strategy=self.val_condition_sampling_strategy,
                condition_positive_prob=self.val_condition_positive_prob,
                multi_condition_enable=self.val_multi_condition_enable,
                multi_condition_prob=self.val_multi_condition_prob,
                multi_condition_min_k=self.val_multi_condition_min_k,
                multi_condition_max_k=self.val_multi_condition_max_k,
                class_aware_resample_tries=self.val_class_aware_resample_tries,
                class_aware_fallback=self.val_class_aware_fallback,
            )
        else:
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
