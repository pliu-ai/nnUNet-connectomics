import os
import pickle
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from skimage.exposure import equalize_adapthist

from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.trainer_structured_conditional_no_slot3 import (
    nnUNetTrainerStructuredConditionalNoSlot3,
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "1" if default else "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _env_labels(name: str, default: Sequence[int]) -> List[int]:
    raw = os.environ.get(name, "")
    if raw is None or str(raw).strip() == "":
        return list(default)
    out: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if token == "":
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out if len(out) > 0 else list(default)


def _resolve_data_key(data_dict: Dict[str, Any]) -> Optional[str]:
    if "data" in data_dict:
        return "data"
    if "image" in data_dict:
        return "image"
    return None


def _resolve_target_key(data_dict: Dict[str, Any]) -> Optional[str]:
    if "target" in data_dict:
        return "target"
    if "segmentation" in data_dict:
        return "segmentation"
    return None


def _to_numpy(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    if torch.is_tensor(arr):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _from_numpy(arr: np.ndarray, ref: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if torch.is_tensor(ref):
        out = torch.from_numpy(arr)
        return out.to(dtype=ref.dtype)
    return arr


def _ensure_5d(arr: np.ndarray) -> Tuple[np.ndarray, bool]:
    if arr.ndim == 5:
        return arr, False
    if arr.ndim == 4:
        return arr[None], True
    raise ValueError(f"Expected 4D or 5D tensor, got shape {arr.shape}.")


def _undo_5d(arr: np.ndarray, was_4d: bool) -> np.ndarray:
    if was_4d:
        return arr[0]
    return arr


def _extract_primary_target(
    target_obj: Any,
) -> Tuple[Optional[Union[np.ndarray, torch.Tensor]], bool]:
    if target_obj is None:
        return None, False
    if isinstance(target_obj, list):
        if len(target_obj) == 0:
            return None, True
        return target_obj[0], True
    return target_obj, False


def _parse_organ_from_key(key: str) -> str:
    stem = os.path.basename(str(key))
    stem = stem.split(".")[0]
    parts = stem.split("_")
    if len(parts) == 0:
        return stem
    if parts[0] == "jrc" and len(parts) > 1:
        return parts[1]
    return parts[0]


class CLAHETransform3D(AbstractTransform):
    """Apply slice-wise CLAHE on 3D EM volumes."""

    def __call__(self, **data_dict: Any) -> Dict[str, Any]:
        data_key = _resolve_data_key(data_dict)
        if data_key is None:
            return data_dict

        data_ref = data_dict[data_key]
        data_np = _to_numpy(data_ref).astype(np.float32, copy=False)
        data_batched, was_4d = _ensure_5d(data_np)

        out = data_batched.copy()
        bsz, ch, depth = out.shape[0], out.shape[1], out.shape[2]
        for b in range(bsz):
            for c in range(ch):
                for z in range(depth):
                    sl = np.clip(out[b, c, z], 0.0, 1.0)
                    out[b, c, z] = equalize_adapthist(sl, clip_limit=0.03).astype(np.float32, copy=False)

        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
        out = _undo_5d(out, was_4d)
        data_dict[data_key] = _from_numpy(out, data_ref)
        return data_dict


class FDATransform3D(AbstractTransform):
    """Apply 3D Fourier Domain Adaptation between samples."""

    def __init__(self, beta: float = 0.1, p_per_sample: float = 0.3) -> None:
        self.beta = float(beta)
        self.p_per_sample = float(p_per_sample)

    def __call__(self, **data_dict: Any) -> Dict[str, Any]:
        data_key = _resolve_data_key(data_dict)
        if data_key is None:
            return data_dict

        data_ref = data_dict[data_key]
        data_np = _to_numpy(data_ref).astype(np.float32, copy=False)
        data_batched, was_4d = _ensure_5d(data_np)

        bsz, ch, zdim, ydim, xdim = data_batched.shape
        if bsz < 2:
            data_dict[data_key] = data_ref
            return data_dict

        out = data_batched.copy()
        block = int(self.beta * min(zdim, ydim, xdim))
        if block < 1:
            data_dict[data_key] = data_ref
            return data_dict

        for b in range(bsz):
            if np.random.rand() >= self.p_per_sample:
                continue
            candidates = [i for i in range(bsz) if i != b]
            if len(candidates) == 0:
                continue
            partner = int(np.random.choice(candidates))

            for c in range(ch):
                src = out[b, c]
                tgt = out[partner, c]

                src_fft = np.fft.fftn(src)
                tgt_fft = np.fft.fftn(tgt)

                src_amp = np.abs(src_fft)
                src_phase = np.angle(src_fft)
                tgt_amp = np.abs(tgt_fft)

                src_amp[:block, :block, :block] = tgt_amp[:block, :block, :block]
                mixed = np.fft.ifftn(src_amp * np.exp(1j * src_phase)).real
                out[b, c] = mixed.astype(np.float32, copy=False)

        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
        out = _undo_5d(out, was_4d)
        data_dict[data_key] = _from_numpy(out, data_ref)
        return data_dict


class EMArtifactTransform3D(AbstractTransform):
    """Simulate streak and missing-slice EM artifacts."""

    def __init__(self, p_streak: float = 0.2, p_missing_slice: float = 0.15) -> None:
        self.p_streak = float(p_streak)
        self.p_missing_slice = float(p_missing_slice)

    def __call__(self, **data_dict: Any) -> Dict[str, Any]:
        data_key = _resolve_data_key(data_dict)
        if data_key is None:
            return data_dict

        data_ref = data_dict[data_key]
        data_np = _to_numpy(data_ref).astype(np.float32, copy=False)
        data_batched, was_4d = _ensure_5d(data_np)
        out = data_batched.copy()

        bsz, ch, zdim, ydim, xdim = out.shape
        for b in range(bsz):
            if np.random.rand() < self.p_streak and zdim > 0:
                z_idx = int(np.random.randint(0, zdim))
                if np.random.rand() < 0.5:
                    stripe = (np.random.randn(ydim) * 0.05).astype(np.float32)
                    out[b, :, z_idx] += stripe[None, :, None]
                else:
                    stripe = (np.random.randn(xdim) * 0.05).astype(np.float32)
                    out[b, :, z_idx] += stripe[None, None, :]
                out = np.clip(out, 0.0, 1.0)

            if np.random.rand() < self.p_missing_slice and zdim > 2:
                z_idx = int(np.random.randint(1, zdim - 1))
                out[b, :, z_idx] = out[b, :, z_idx - 1]
                out = np.clip(out, 0.0, 1.0)

        out = np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
        out = _undo_5d(out, was_4d)
        data_dict[data_key] = _from_numpy(out, data_ref)
        return data_dict


class CrossOrganCutMixTransform3D(AbstractTransform):
    """Paste cross-organ organelle patches from a prebuilt patch bank."""

    def __init__(
        self,
        patch_bank_path: str,
        target_labels: Optional[Sequence[int]] = None,
        p_per_sample: float = 0.3,
    ) -> None:
        self.patch_bank_path = patch_bank_path
        self.p_per_sample = float(p_per_sample)
        self.target_labels = (
            list(target_labels)
            if target_labels is not None
            else [6, 8, 10, 12, 14, 18, 29]
        )
        self.patch_bank: Dict[int, List[Dict[str, Any]]] = self._load_patch_bank(self.patch_bank_path)

    @staticmethod
    def _load_patch_bank(path: str) -> Dict[int, List[Dict[str, Any]]]:
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("patch_bank.pkl must contain a dict: {label_id: [patch dicts]}")
        out: Dict[int, List[Dict[str, Any]]] = {}
        for k, v in loaded.items():
            try:
                label_id = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, list):
                out[label_id] = v
        return out

    @staticmethod
    def _compute_start(center: int, patch_dim: int, vol_dim: int) -> int:
        if vol_dim <= patch_dim:
            return 0
        return int(np.clip(center - patch_dim // 2, 0, vol_dim - patch_dim))

    def __call__(self, **data_dict: Any) -> Dict[str, Any]:
        data_key = _resolve_data_key(data_dict)
        target_key = _resolve_target_key(data_dict)
        if data_key is None or target_key is None:
            return data_dict

        data_ref = data_dict[data_key]
        target_ref = data_dict[target_key]
        primary_target_ref, target_is_list = _extract_primary_target(target_ref)
        if primary_target_ref is None:
            return data_dict

        data_np = _to_numpy(data_ref).astype(np.float32, copy=False)
        seg_np = _to_numpy(primary_target_ref).astype(np.int32, copy=False)
        data_batched, data_was_4d = _ensure_5d(data_np)
        seg_batched, seg_was_4d = _ensure_5d(seg_np)

        if data_batched.shape[0] != seg_batched.shape[0]:
            return data_dict
        if data_batched.shape[2:] != seg_batched.shape[2:]:
            return data_dict

        out_data = data_batched.copy()
        out_seg = seg_batched.copy()

        batch_size = out_data.shape[0]
        keys = data_dict.get("keys", [f"sample_{i}" for i in range(batch_size)])
        if not isinstance(keys, (list, tuple)):
            keys = [str(keys) for _ in range(batch_size)]

        for b in range(batch_size):
            if np.random.rand() >= self.p_per_sample:
                continue

            seg_vol = out_seg[b, 0]
            labels_present = [lbl for lbl in self.target_labels if np.any(seg_vol == lbl)]
            if len(labels_present) == 0:
                continue
            label_id = int(np.random.choice(labels_present))

            current_key = str(keys[b]) if b < len(keys) else f"sample_{b}"
            current_organ = _parse_organ_from_key(current_key)

            candidates = self.patch_bank.get(label_id, [])
            cross_organ = [p for p in candidates if str(p.get("organ", "")) != current_organ]
            pool = cross_organ if len(cross_organ) > 0 else candidates
            if len(pool) == 0:
                continue
            patch = pool[int(np.random.randint(0, len(pool)))]

            patch_img = np.asarray(patch["img"], dtype=np.float32)
            patch_seg = np.asarray(patch["seg"], dtype=np.int32)
            if "mask" not in patch:
                continue
            patch_mask = np.asarray(patch["mask"], dtype=bool)
            if patch_img.ndim != 3 or patch_seg.ndim != 3 or patch_mask.ndim != 3:
                continue
            if not (patch_img.shape == patch_seg.shape == patch_mask.shape):
                continue

            # Clip organelle bbox to fit inside the destination volume.
            zdim, ydim, xdim = seg_vol.shape
            pz, py, px = patch_img.shape
            pz_eff, py_eff, px_eff = min(pz, zdim), min(py, ydim), min(px, xdim)
            if pz_eff <= 0 or py_eff <= 0 or px_eff <= 0:
                continue

            # Place organelle centered on the centroid of the target label in the destination.
            label_coords = np.argwhere(seg_vol == label_id)
            if label_coords.shape[0] == 0:
                continue
            centroid = label_coords.mean(axis=0).astype(int)
            z0 = self._compute_start(int(centroid[0]), pz_eff, zdim)
            y0 = self._compute_start(int(centroid[1]), py_eff, ydim)
            x0 = self._compute_start(int(centroid[2]), px_eff, xdim)
            z1, y1, x1 = z0 + pz_eff, y0 + py_eff, x0 + px_eff

            # Mask-based paste: only overwrite voxels belonging to the organelle instance.
            effective_mask = patch_mask[:pz_eff, :py_eff, :px_eff]
            out_data[b, :, z0:z1, y0:y1, x0:x1] = np.where(
                effective_mask[np.newaxis],
                patch_img[np.newaxis, :pz_eff, :py_eff, :px_eff],
                out_data[b, :, z0:z1, y0:y1, x0:x1],
            )
            out_seg[b, 0, z0:z1, y0:y1, x0:x1] = np.where(
                effective_mask,
                patch_seg[:pz_eff, :py_eff, :px_eff],
                out_seg[b, 0, z0:z1, y0:y1, x0:x1],
            )

        out_data = np.clip(out_data, 0.0, 1.0).astype(np.float32, copy=False)
        out_data = _undo_5d(out_data, data_was_4d)
        out_seg = _undo_5d(out_seg.astype(seg_np.dtype, copy=False), seg_was_4d)

        data_dict[data_key] = _from_numpy(out_data, data_ref)
        converted_target = _from_numpy(out_seg, primary_target_ref)
        if target_is_list and isinstance(target_ref, list) and len(target_ref) > 0:
            target_ref[0] = converted_target
            data_dict[target_key] = target_ref
        else:
            data_dict[target_key] = converted_target
        return data_dict


class nnUNetTrainerCustomAug(nnUNetTrainerStructuredConditionalNoSlot3):
    """
    Structured-conditional trainer + custom augmentations.

    This keeps the same conditional network/training pipeline as
    nnUNetTrainerStructuredConditionalNoSlot3 and appends CLAHE/FDA/EM/CutMix.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.patch_bank_path = os.path.join(self.preprocessed_dataset_folder, "patch_bank.pkl")

        # optimizer/schedule overrides
        self.initial_lr = _env_float("NNUNET_CUSTOMAUG_INITIAL_LR", float(self.initial_lr))
        self.weight_decay = _env_float("NNUNET_CUSTOMAUG_WEIGHT_DECAY", float(self.weight_decay))
        self.num_epochs = _env_int("NNUNET_CUSTOMAUG_NUM_EPOCHS", int(self.num_epochs))

        self.use_clahe = _env_flag("NNUNET_CUSTOMAUG_USE_CLAHE", True)
        self.use_fda = _env_flag("NNUNET_CUSTOMAUG_USE_FDA", True)
        self.use_em_artifact = _env_flag("NNUNET_CUSTOMAUG_USE_EM_ARTIFACT", True)
        self.use_cutmix = _env_flag("NNUNET_CUSTOMAUG_USE_CUTMIX", True)

        self.fda_beta = _env_float("NNUNET_CUSTOMAUG_FDA_BETA", 0.1)
        self.fda_p = _env_float("NNUNET_CUSTOMAUG_FDA_P", 0.3)
        self.em_streak_p = _env_float("NNUNET_CUSTOMAUG_EM_STREAK_P", 0.2)
        self.em_missing_p = _env_float("NNUNET_CUSTOMAUG_EM_MISSING_P", 0.15)
        self.cutmix_p = _env_float("NNUNET_CUSTOMAUG_CUTMIX_P", 0.3)
        self.cutmix_target_labels = _env_labels(
            "NNUNET_CUSTOMAUG_CUTMIX_TARGET_LABELS",
            [6, 8, 10, 12, 14, 18, 29],
        )

        self.print_to_log_file(f"[CustomAug] patch bank path: {self.patch_bank_path}")
        self.print_to_log_file(
            f"[CustomAug] optimizer config: initial_lr={self.initial_lr}, "
            f"weight_decay={self.weight_decay}, num_epochs={self.num_epochs}"
        )
        self.print_to_log_file(
            "[CustomAug] switches: "
            f"CLAHE={self.use_clahe}, FDA={self.use_fda}, EM_ART={self.use_em_artifact}, CUTMIX={self.use_cutmix}"
        )

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int, ...]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: Optional[List[bool]] = None,
        is_cascaded: bool = False,
        foreground_labels: Optional[Union[Tuple[int, ...], List[int]]] = None,
        regions: Optional[List[Union[List[int], Tuple[int, ...], int]]] = None,
        ignore_label: Optional[int] = None,
    ) -> BasicTransform:
        transforms = super().get_training_transforms(
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

        if self.use_clahe:
            transforms.transforms.append(CLAHETransform3D())
        if self.use_fda:
            transforms.transforms.append(FDATransform3D(beta=self.fda_beta, p_per_sample=self.fda_p))
        if self.use_em_artifact:
            transforms.transforms.append(
                EMArtifactTransform3D(
                    p_streak=self.em_streak_p,
                    p_missing_slice=self.em_missing_p,
                )
            )

        if self.use_cutmix and os.path.isfile(self.patch_bank_path):
            try:
                transforms.transforms.append(
                    CrossOrganCutMixTransform3D(
                        patch_bank_path=self.patch_bank_path,
                        target_labels=self.cutmix_target_labels,
                        p_per_sample=self.cutmix_p,
                    )
                )
                self.print_to_log_file("[CustomAug] CrossOrganCutMixTransform3D enabled.")
            except Exception as e:
                self.print_to_log_file(
                    f"[CustomAug][WARNING] Failed to load patch bank ({self.patch_bank_path}): {e}. "
                    f"Skipping CrossOrganCutMixTransform3D."
                )
        elif self.use_cutmix:
            self.print_to_log_file(
                f"[CustomAug][WARNING] patch_bank.pkl not found at {self.patch_bank_path}. "
                f"Skipping CrossOrganCutMixTransform3D."
            )
        else:
            self.print_to_log_file("[CustomAug] CrossOrganCutMixTransform3D disabled by env.")
        return transforms
