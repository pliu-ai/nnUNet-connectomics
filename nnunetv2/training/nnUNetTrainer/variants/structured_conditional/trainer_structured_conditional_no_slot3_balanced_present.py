from __future__ import annotations

import os

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

from .dataloader_structured_conditional_balanced_present_no_slot3 import (
    StructuredConditionalBalancedPresentDataLoader3D,
)
from .trainer_structured_conditional_no_slot3 import nnUNetTrainerStructuredConditionalNoSlot3


class nnUNetTrainerStructuredConditionalNoSlot3BalancedPresent(
    nnUNetTrainerStructuredConditionalNoSlot3
):
    """
    no_slot3 trainer variant with balanced-present sampling:
    - group IDs are scheduled uniformly per epoch over available groups
    - each sampled patch is targeted to contain the scheduled group
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device=None,
    ):
        if device is None:
            import torch

            device = torch.device("cuda")
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.balance_seed = int(
            os.environ.get("NNUNET_STRUCTCOND_BALANCE_SEED", str(self.group_sampling_seed))
        )
        self.balance_verify_present = (
            str(os.environ.get("NNUNET_STRUCTCOND_BALANCE_VERIFY_PRESENT", "1")).lower()
            in {"1", "true", "yes", "y"}
        )
        self.balance_force_fg = (
            str(os.environ.get("NNUNET_STRUCTCOND_BALANCE_FORCE_FG", "1")).lower()
            in {"1", "true", "yes", "y"}
        )
        self.balance_single_thread = (
            str(os.environ.get("NNUNET_STRUCTCOND_BALANCE_SINGLE_THREADED", "0")).lower()
            in {"1", "true", "yes", "y"}
        )

    def _maybe_set_epoch_on_train_loader(self) -> None:
        loader = self.dataloader_train
        if loader is None:
            return

        candidates = [loader]
        for attr in ("data_loader", "generator"):
            if hasattr(loader, attr):
                candidates.append(getattr(loader, attr))

        for candidate in candidates:
            if hasattr(candidate, "set_epoch"):
                try:
                    candidate.set_epoch(int(self.current_epoch))
                    return
                except Exception as e:
                    self.print_to_log_file(
                        f"[StructuredConditionalBalancedPresent] failed to set epoch on loader: {e}"
                    )
                    return

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._maybe_set_epoch_on_train_loader()

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            "[StructuredConditionalBalancedPresent] "
            f"balance_seed={self.balance_seed}, "
            f"verify_present={self.balance_verify_present}, "
            f"force_fg={self.balance_force_fg}, "
            f"single_threaded={self.balance_single_thread}"
        )

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
            dl_tr = StructuredConditionalBalancedPresentDataLoader3D(
                dataset_tr,
                self.batch_size,
                initial_patch_size,
                self.configuration_manager.patch_size,
                self.label_manager,
                oversample_foreground_percent=self.oversample_foreground_percent,
                sampling_probabilities=None,
                pad_sides=None,
                transforms=tr_transforms,
                num_iterations_per_epoch=self.num_iterations_per_epoch,
                balance_seed=self.balance_seed + int(self.local_rank),
                verify_present_after_transforms=self.balance_verify_present,
                force_fg_for_group=self.balance_force_fg,
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

        allowed_num_processes = 0 if self.balance_single_thread else get_allowed_n_proc_DA()
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
