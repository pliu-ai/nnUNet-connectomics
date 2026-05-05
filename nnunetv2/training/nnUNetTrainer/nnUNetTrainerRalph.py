import json
import os
from pathlib import Path
from typing import Any, List, Mapping, Tuple, Union

import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform

from nnunetv2.training.loss.compound_losses import (
    DC_and_BCE_loss,
    DC_and_CE_loss,
    DC_and_topk_loss,
)
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# Per-class loss weights are clamped to this range to prevent pathological gradients.
_LOSS_WEIGHT_BOUNDS: tuple[float, float] = (0.1, 10.0)

# Augmentation presets recognised by this trainer.
_KNOWN_AUG_PRESETS = {"heavy_elastic", "heavy_intensity", "heavy_rotation"}


class nnUNetTrainerRalph(nnUNetTrainer):
    """
    Ralph-aware trainer.

    Reads an external JSON config pointed to by env var ``RALPH_CONFIG_JSON`` and
    applies supported knobs:

    ==================  ========================================================
    Key                 Effect
    ==================  ========================================================
    initial_lr          Overrides ``self.initial_lr`` (also supports ``learning_rate``)
    num_epochs          Overrides ``self.num_epochs``
    lr_schedule         ``"poly"`` (default) or ``"cosine"``
    weight_ce           CE term weight in DC+CE loss (default 1.0)
    weight_dice         Dice term weight in DC+CE loss (default 1.0)
    loss_type           ``DC_and_CE`` (default) or ``DC_and_TopK_CE``
    oversample_foreground_percent
                        Overrides foreground patch oversampling ratio.
    batch_dice          Overrides plan's batch_dice bool.
    loss_weights        ``{class_name: float}`` – per-class CE weight tensor.
                        Values are clamped to [0.1, 10.0].
    augmentations       List of preset strings.  Supported values:
                        ``heavy_elastic``  – enable elastic deformation
                                            (parent default: disabled)
                        ``heavy_rotation`` – increase rotation probability 0.2→0.5
                        ``heavy_intensity``– increase Gaussian noise/blur probs
    augmentation_settings
                        Fine-grained augmentation fields from action_space:
                        ``enable_elastic``, ``rotation_range``, ``scale_range``,
                        ``gamma_range``, ``mirror_axes``
    ==================  ========================================================
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
        self._ralph_config: dict[str, Any] = self._load_ralph_config_from_env()
        self._ralph_lr_schedule: str = "poly"
        self._ralph_loss_weights: dict[str, float] = {}
        self._ralph_weight_ce: float = 1.0
        self._ralph_weight_dice: float = 1.0
        self._ralph_loss_type: str = "DC_and_CE"
        self._ralph_batch_dice: bool | None = None
        self._ralph_aug_presets: set[str] = set()
        self._ralph_rotation_range_deg: float | None = None
        self._ralph_scale_range: tuple[float, float] | None = None
        self._ralph_gamma_range: tuple[float, float] | None = None
        self._ralph_mirror_axes: tuple[int, ...] | None = None
        self._apply_ralph_overrides()

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_ralph_config_from_env() -> dict[str, Any]:
        cfg_path = os.environ.get("RALPH_CONFIG_JSON", "").strip()
        if not cfg_path:
            return {}
        path = Path(cfg_path)
        if not path.exists():
            print(
                f"[nnUNetTrainerRalph] WARNING: RALPH_CONFIG_JSON path does not exist: {path}. "
                f"All LLM decisions for this run will be IGNORED."
            )
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(
                f"[nnUNetTrainerRalph] WARNING: failed to parse ralph config at {path}: {exc}. "
                f"All LLM decisions for this run will be IGNORED."
            )
            return {}
        if not isinstance(payload, dict):
            print(
                "[nnUNetTrainerRalph] WARNING: ralph config is not a JSON object. "
                "All LLM decisions for this run will be IGNORED."
            )
            return {}
        return payload

    def _apply_ralph_overrides(self) -> None:
        cfg = self._ralph_config
        if not cfg:
            self.print_to_log_file(
                "nnUNetTrainerRalph: no external Ralph config found, using nnUNet defaults"
            )
            return

        lr = cfg.get("initial_lr", cfg.get("learning_rate", None))
        if isinstance(lr, (int, float)) and float(lr) > 0:
            self.initial_lr = float(lr)

        num_epochs = cfg.get("num_epochs", None)
        if isinstance(num_epochs, int) and num_epochs > 0:
            self.num_epochs = int(num_epochs)

        lr_schedule = str(cfg.get("lr_schedule", "poly")).strip().lower()
        if lr_schedule in {"poly", "cosine"}:
            self._ralph_lr_schedule = lr_schedule

        weight_ce = cfg.get("weight_ce", None)
        if isinstance(weight_ce, (int, float)) and float(weight_ce) > 0:
            self._ralph_weight_ce = float(weight_ce)

        weight_dice = cfg.get("weight_dice", None)
        if isinstance(weight_dice, (int, float)) and float(weight_dice) > 0:
            self._ralph_weight_dice = float(weight_dice)

        loss_type = cfg.get("loss_type", None)
        if isinstance(loss_type, str) and loss_type.strip():
            self._ralph_loss_type = loss_type.strip()

        batch_dice = cfg.get("batch_dice", None)
        if isinstance(batch_dice, bool):
            self._ralph_batch_dice = batch_dice

        oversample_fg = cfg.get("oversample_foreground_percent", None)
        if isinstance(oversample_fg, (int, float)) and 0.0 < float(oversample_fg) <= 1.0:
            self.oversample_foreground_percent = float(oversample_fg)

        loss_weights = cfg.get("loss_weights", {})
        if isinstance(loss_weights, Mapping):
            lo, hi = _LOSS_WEIGHT_BOUNDS
            parsed: dict[str, float] = {}
            for k, v in loss_weights.items():
                if not (isinstance(k, str) and isinstance(v, (int, float))):
                    continue
                w = float(v)
                if not (lo <= w <= hi):
                    print(
                        f"[nnUNetTrainerRalph] WARNING: loss_weight for '{k}' = {w:.4f} "
                        f"is outside [{lo}, {hi}]; clamping."
                    )
                    w = max(lo, min(hi, w))
                parsed[k.strip().lower()] = w
            self._ralph_loss_weights = parsed

        augmentations = cfg.get("augmentations", [])
        if isinstance(augmentations, list):
            presets = {str(a).strip().lower() for a in augmentations if isinstance(a, str)}
            unknown = presets - _KNOWN_AUG_PRESETS
            if unknown:
                print(
                    f"[nnUNetTrainerRalph] WARNING: unknown augmentation presets ignored: {unknown}. "
                    f"Known presets: {_KNOWN_AUG_PRESETS}"
                )
            self._ralph_aug_presets = presets & _KNOWN_AUG_PRESETS

        aug_settings = cfg.get("augmentation_settings", {})
        if isinstance(aug_settings, Mapping):
            if "enable_elastic" in aug_settings:
                if bool(aug_settings.get("enable_elastic", False)):
                    self._ralph_aug_presets.add("heavy_elastic")
                else:
                    self._ralph_aug_presets.discard("heavy_elastic")

            rotation_range = aug_settings.get("rotation_range", None)
            if isinstance(rotation_range, (int, float)) and float(rotation_range) > 0:
                self._ralph_rotation_range_deg = float(rotation_range)
                if self._ralph_rotation_range_deg > 30:
                    self._ralph_aug_presets.add("heavy_rotation")
                else:
                    self._ralph_aug_presets.discard("heavy_rotation")

            scale_range = aug_settings.get("scale_range", None)
            if (
                isinstance(scale_range, (list, tuple))
                and len(scale_range) == 2
                and all(isinstance(x, (int, float)) for x in scale_range)
            ):
                lo = float(scale_range[0])
                hi = float(scale_range[1])
                if lo > 0 and hi > 0 and lo <= hi:
                    self._ralph_scale_range = (lo, hi)

            gamma_range = aug_settings.get("gamma_range", None)
            if (
                isinstance(gamma_range, (list, tuple))
                and len(gamma_range) == 2
                and all(isinstance(x, (int, float)) for x in gamma_range)
            ):
                lo = float(gamma_range[0])
                hi = float(gamma_range[1])
                if lo > 0 and hi > 0 and lo <= hi:
                    self._ralph_gamma_range = (lo, hi)
                    self._ralph_aug_presets.add("heavy_intensity")

            mirror_axes = aug_settings.get("mirror_axes", None)
            if (
                isinstance(mirror_axes, (list, tuple))
                and all(isinstance(x, int) for x in mirror_axes)
            ):
                # empty tuple is a valid override: disable mirroring.
                self._ralph_mirror_axes = tuple(int(x) for x in mirror_axes)

        self.print_to_log_file(
            "nnUNetTrainerRalph overrides applied:",
            {
                "initial_lr": self.initial_lr,
                "num_epochs": self.num_epochs,
                "lr_schedule": self._ralph_lr_schedule,
                "loss_type": self._ralph_loss_type,
                "batch_dice": self._ralph_batch_dice,
                "oversample_foreground_percent": self.oversample_foreground_percent,
                "weight_ce": self._ralph_weight_ce,
                "weight_dice": self._ralph_weight_dice,
                "num_loss_weight_classes": len(self._ralph_loss_weights),
                "aug_presets": sorted(self._ralph_aug_presets),
                "rotation_range_deg": self._ralph_rotation_range_deg,
                "scale_range": self._ralph_scale_range,
                "gamma_range": self._ralph_gamma_range,
                "mirror_axes": self._ralph_mirror_axes,
            },
        )

    # ------------------------------------------------------------------
    # Loss building
    # ------------------------------------------------------------------

    def _build_name_to_label_id(self) -> dict[str, int]:
        labels = self.dataset_json.get("labels", {})
        name_to_id: dict[str, int] = {}
        if not isinstance(labels, Mapping):
            return name_to_id
        for name, raw in labels.items():
            if not isinstance(name, str):
                continue
            label_id: int | None = None
            if isinstance(raw, int):
                label_id = raw
            elif isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], int):
                label_id = raw[0]
            if label_id is not None:
                name_to_id[name.strip().lower()] = int(label_id)
        return name_to_id

    def _build_ce_weight_tensor(self) -> torch.Tensor | None:
        if not self._ralph_loss_weights:
            return None
        if self.label_manager.has_regions:
            self.print_to_log_file(
                "nnUNetTrainerRalph: region-based labels detected; per-class CE weights skipped"
            )
            return None

        num_heads = int(self.label_manager.num_segmentation_heads)
        if num_heads <= 0:
            return None

        name_to_id = self._build_name_to_label_id()
        weights = torch.ones(num_heads, dtype=torch.float32)
        applied = 0
        missing: list[str] = []

        for cls_name, cls_weight in self._ralph_loss_weights.items():
            cls_id = name_to_id.get(cls_name)
            if cls_id is None or cls_id < 0 or cls_id >= num_heads:
                missing.append(cls_name)
                continue
            weights[cls_id] = float(cls_weight)
            applied += 1

        self.print_to_log_file(
            "nnUNetTrainerRalph class weights:",
            {"applied": applied, "missing": missing[:20], "num_heads": num_heads},
        )
        return weights if applied > 0 else None

    def _build_loss(self):
        # NOTE: We re-implement rather than call super() because we need to inject
        # ce_weights into the DC_and_CE_loss constructor; setting them post-hoc via
        # buffer assignment is fragile across nnUNet versions.
        ce_weights = self._build_ce_weight_tensor()
        batch_dice = (
            bool(self.configuration_manager.batch_dice)
            if self._ralph_batch_dice is None
            else bool(self._ralph_batch_dice)
        )
        loss_type = str(self._ralph_loss_type).strip()

        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    "batch_dice": batch_dice,
                    "do_bg": True,
                    "smooth": 1e-5,
                    "ddp": self.is_ddp,
                },
                weight_ce=self._ralph_weight_ce,
                weight_dice=self._ralph_weight_dice,
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        else:
            ce_kwargs: dict[str, Any] = {}
            if ce_weights is not None:
                ce_kwargs["weight"] = ce_weights.to(self.device)
            if loss_type == "DC_and_TopK_CE":
                ce_kwargs = dict(ce_kwargs)
                ce_kwargs["k"] = 10
                loss = DC_and_topk_loss(
                    {
                        "batch_dice": batch_dice,
                        "smooth": 1e-5,
                        "do_bg": False,
                        "ddp": self.is_ddp,
                    },
                    ce_kwargs,
                    weight_ce=self._ralph_weight_ce,
                    weight_dice=self._ralph_weight_dice,
                    ignore_label=self.label_manager.ignore_label,
                )
            else:
                if loss_type in {"DC_and_Focal", "Tversky"}:
                    self.print_to_log_file(
                        f"nnUNetTrainerRalph: loss_type '{loss_type}' not implemented, fallback to DC_and_CE"
                    )
                elif loss_type not in {"DC_and_CE", ""}:
                    self.print_to_log_file(
                        f"nnUNetTrainerRalph: unknown loss_type '{loss_type}', fallback to DC_and_CE"
                    )
                loss = DC_and_CE_loss(
                    {
                        "batch_dice": batch_dice,
                        "smooth": 1e-5,
                        "do_bg": False,
                        "ddp": self.is_ddp,
                    },
                    ce_kwargs,
                    weight_ce=self._ralph_weight_ce,
                    weight_dice=self._ralph_weight_dice,
                    ignore_label=self.label_manager.ignore_label,
                    dice_class=MemoryEfficientSoftDiceLoss,
                )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        if self._ralph_lr_schedule == "cosine":
            # eta_min=1e-6 prevents lr from reaching 0, which is important when
            # continuing from a checkpoint (avoids a jarring lr jump on resume).
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, int(self.num_epochs)), eta_min=1e-6
            )
        else:
            lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int, ...]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions=None,
        ignore_label: int = None,
    ):
        """
        Delegates to the parent static implementation, then modifies the composed
        transform pipeline in-place according to ``self._ralph_aug_presets``.

        Supported presets
        -----------------
        heavy_elastic
            Enables elastic deformation in the ``SpatialTransform`` (parent default:
            ``p_elastic_deform=0``).  Sets ``p_elastic_deform=0.4`` and
            ``elastic_deform_magnitude=(0, 30)`` pixels.
        heavy_rotation
            Increases rotation probability in ``SpatialTransform`` from 0.2 to 0.5.
        heavy_intensity
            Multiplies ``apply_probability`` of GaussianNoise (×2.5, cap 0.4) and
            GaussianBlur (×2.0, cap 0.4) ``RandomTransform`` wrappers.
        """
        composed = nnUNetTrainer.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
        )

        if not self._ralph_aug_presets:
            return composed

        # --- spatial preset modifications (SpatialTransform is always first) ---
        for t in composed.transforms:
            if isinstance(t, SpatialTransform):
                if "heavy_elastic" in self._ralph_aug_presets:
                    t.p_elastic_deform = 0.4
                    # Default magnitude (0, 0.2) pixels is essentially zero; 30 px
                    # gives meaningful deformation on typical 3D EM patches.
                    t.elastic_deform_magnitude = (0, 30)
                if "heavy_rotation" in self._ralph_aug_presets:
                    t.p_rotation = 0.5
                if self._ralph_rotation_range_deg is not None:
                    rad = self._ralph_rotation_range_deg / 360.0 * 2.0 * np.pi
                    t.rotation = (-rad, rad)
                    # Keep rotation active when user explicitly set the range.
                    t.p_rotation = max(float(getattr(t, "p_rotation", 0.0)), 0.2)
                if self._ralph_scale_range is not None:
                    t.scaling = self._ralph_scale_range
                break

        # --- intensity preset modifications ---
        if "heavy_intensity" in self._ralph_aug_presets:
            for t in composed.transforms:
                if not isinstance(t, RandomTransform):
                    continue
                if isinstance(t.transform, GaussianNoiseTransform):
                    t.apply_probability = min(t.apply_probability * 2.5, 0.4)
                elif isinstance(t.transform, GaussianBlurTransform):
                    t.apply_probability = min(t.apply_probability * 2.0, 0.4)
                elif (
                    self._ralph_gamma_range is not None
                    and isinstance(t.transform, GammaTransform)
                ):
                    t.transform.gamma = BGContrast(self._ralph_gamma_range)

        # Explicit gamma override should still apply even without heavy_intensity preset.
        if self._ralph_gamma_range is not None:
            for t in composed.transforms:
                if isinstance(t, RandomTransform) and isinstance(t.transform, GammaTransform):
                    t.transform.gamma = BGContrast(self._ralph_gamma_range)

        # Mirror axes override: () disables mirroring.
        if self._ralph_mirror_axes is not None:
            mirror_idx = None
            for i, t in enumerate(composed.transforms):
                if isinstance(t, MirrorTransform):
                    mirror_idx = i
                    break

            if len(self._ralph_mirror_axes) == 0:
                if mirror_idx is not None:
                    composed.transforms.pop(mirror_idx)
            else:
                if mirror_idx is not None:
                    composed.transforms[mirror_idx].allowed_axes = self._ralph_mirror_axes
                else:
                    composed.transforms.append(MirrorTransform(allowed_axes=self._ralph_mirror_axes))

        self.print_to_log_file(
            "nnUNetTrainerRalph: augmentation presets applied:", sorted(self._ralph_aug_presets)
        )
        return composed
    
    def perform_actual_validation(self, *args, **kwargs):
            pass