#!/usr/bin/env python3
import argparse
import glob
import os
import pickle
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from scipy.ndimage import find_objects as scipy_find_objects
from scipy.ndimage import label as scipy_label


LABEL_IDS: Dict[str, int] = {
    "ecs": 0,
    "pm": 1,
    "cyto": 2,
    "mito_mem": 3,
    "mito_lum": 4,
    "mito_ribo": 5,
    "golgi_mem": 6,
    "golgi_lum": 7,
    "ves_mem": 8,
    "ves_lum": 9,
    "endo_mem": 10,
    "endo_lum": 11,
    "lyso_mem": 12,
    "lyso_lum": 13,
    "ld_mem": 14,
    "ld_lum": 15,
    "er_mem": 16,
    "er_lum": 17,
    "eres_mem": 18,
    "eres_lum": 19,
    "ne_mem": 20,
    "ne_lum": 21,
    "np_out": 22,
    "np_in": 23,
    "hchrom": 24,
    "echrom": 25,
    "nucpl": 26,
    "mt_out": 27,
    "mt_in": 28,
    "perox_mem": 29,
    "perox_lum": 30,
}

# Default labels used by CrossOrganCutMixTransform3D. Only these are processed.
DEFAULT_TARGET_LABEL_IDS: Set[int] = {6, 8, 10, 12, 14, 18, 29}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build patch bank for cross-organ CutMix.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--max_instances",
        type=int,
        default=5,
        help="Max organelle instances to store per label per volume. "
             "Largest instances are preferred.",
    )
    parser.add_argument(
        "--min_voxels",
        type=int,
        default=100,
        help="Minimum voxel count to consider a connected component as a valid instance.",
    )
    parser.add_argument(
        "--target_labels",
        type=int,
        nargs="+",
        default=None,
        help="Label IDs to process. Defaults to CutMix target labels: "
             f"{sorted(DEFAULT_TARGET_LABEL_IDS)}. Pass --target_labels 6 8 10 to override.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _to_zyx(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[-1] == 1:
            return arr[..., 0]
        return arr[0]
    raise ValueError(f"Unsupported array shape {arr.shape}. Expected 3D or 4D.")


def _extract_case_id(seg_path: str) -> str:
    base = os.path.basename(seg_path)
    if not base.endswith("_seg.npy"):
        raise ValueError(f"Unexpected segmentation filename: {seg_path}")
    return base[: -len("_seg.npy")]


def _extract_organ(case_id: str) -> str:
    parts = case_id.split("_")
    if len(parts) == 0:
        return case_id
    if parts[0] == "jrc" and len(parts) > 1:
        return parts[1]
    return parts[0]


def _discover_cases(data_root: str) -> List[Tuple[str, str, str]]:
    seg_paths = sorted(glob.glob(os.path.join(data_root, "*_seg.npy")))
    cases: List[Tuple[str, str, str]] = []
    for seg_path in seg_paths:
        case_id = _extract_case_id(seg_path)
        img_path = os.path.join(data_root, f"{case_id}.npy")
        if not os.path.isfile(img_path):
            continue
        cases.append((case_id, img_path, seg_path))
    return cases


def _extract_organelle_instances(
    img: np.ndarray,
    seg: np.ndarray,
    label_id: int,
    organ: str,
    label_name: str,
    max_instances: int,
    min_voxels: int,
) -> List[Dict[str, Any]]:
    """Extract connected-component instances of label_id as whole-organelle patches."""
    binary = seg == label_id
    if not binary.any():
        return []

    labeled, n_components = scipy_label(binary)
    if n_components == 0:
        return []

    # Single bincount scan replaces n_components full-array scans.
    sizes = np.bincount(labeled.ravel())[1:]  # index 0 is background

    valid_mask = sizes >= min_voxels
    if not valid_mask.any():
        return []

    valid_cids = np.where(valid_mask)[0] + 1  # component IDs are 1-based
    valid_sizes = sizes[valid_cids - 1]
    top_indices = np.argsort(valid_sizes)[::-1][:max_instances]
    selected_cids = valid_cids[top_indices]

    # find_objects returns bounding-box slices for each component directly.
    bbox_slices = scipy_find_objects(labeled)

    patches: List[Dict[str, Any]] = []
    for cid in selected_cids:
        sl = bbox_slices[cid - 1]
        if sl is None:
            continue
        img_crop = img[sl].astype(np.float16)   # float16 halves storage vs float32
        seg_crop = seg[sl].astype(np.int16)
        mask_crop = labeled[sl] == cid           # bool, 1 byte/voxel

        patches.append(
            {
                "img": img_crop.copy(),
                "seg": seg_crop.copy(),
                "mask": mask_crop.copy(),
                "organ": organ,
                "label_id": int(label_id),
                "label_name": label_name,
                "voxel_count": int(sizes[cid - 1]),
            }
        )

    return patches


def build_patch_bank(
    data_root: str,
    max_instances: int,
    min_voxels: int,
    target_label_ids: Set[int],
) -> Dict[int, List[Dict[str, Any]]]:
    cases = _discover_cases(data_root)
    if len(cases) == 0:
        raise RuntimeError(f"No valid (image, seg) pairs found in {data_root}")

    id_to_name = {v: k for k, v in LABEL_IDS.items()}
    target_labels = [(label_id, id_to_name[label_id]) for label_id in sorted(target_label_ids)
                     if label_id in id_to_name]

    patch_bank: Dict[int, List[Dict[str, Any]]] = {label_id: [] for label_id, _ in target_labels}

    print(f"Found {len(cases)} cases under: {data_root}")
    print(f"Config: max_instances={max_instances}, min_voxels={min_voxels}")
    print(f"Processing {len(target_labels)} labels: "
          f"{[f'{lid}({lname})' for lid, lname in target_labels]}")

    for idx, (case_id, img_path, seg_path) in enumerate(cases, start=1):
        organ = _extract_organ(case_id)
        img = _to_zyx(np.load(img_path, mmap_mode="r")).astype(np.float32, copy=False)
        seg = _to_zyx(np.load(seg_path, mmap_mode="r")).astype(np.int32, copy=False)

        if img.shape != seg.shape:
            print(f"[WARN] shape mismatch for {case_id}: img={img.shape}, seg={seg.shape}; skipping.")
            continue

        for label_id, label_name in target_labels:
            patches = _extract_organelle_instances(
                img=img,
                seg=seg,
                label_id=label_id,
                organ=organ,
                label_name=label_name,
                max_instances=max_instances,
                min_voxels=min_voxels,
            )
            patch_bank[label_id].extend(patches)

        if idx % 10 == 0 or idx == len(cases):
            print(f"Processed {idx}/{len(cases)} cases")

    return patch_bank


def print_summary(patch_bank: Dict[int, List[Dict[str, Any]]]) -> None:
    print("\nPatch bank summary:")
    id_to_name = {v: k for k, v in LABEL_IDS.items()}
    total_mb = 0.0
    for label_id in sorted(patch_bank.keys()):
        patches = patch_bank[label_id]
        organ_count = len({str(p.get("organ", "")) for p in patches})
        voxels = [p.get("voxel_count", 0) for p in patches]
        label_mb = sum(
            p["img"].nbytes + p["seg"].nbytes + p["mask"].nbytes
            for p in patches
        ) / 1024 ** 2
        total_mb += label_mb
        size_info = f", voxels: min={min(voxels)} max={max(voxels)}" if voxels else ""
        print(
            f"  label_id={label_id:>2d} ({id_to_name.get(label_id, '?'):<10}) "
            f"instances={len(patches):>5d}, organs={organ_count:>3d}"
            f"{size_info}, ~{label_mb:.1f} MB"
        )
    print(f"\nTotal in-memory size: ~{total_mb:.1f} MB")


def main() -> None:
    args = parse_args()
    target_label_ids = set(args.target_labels) if args.target_labels else DEFAULT_TARGET_LABEL_IDS

    patch_bank = build_patch_bank(
        data_root=args.data_root,
        max_instances=int(args.max_instances),
        min_voxels=int(args.min_voxels),
        target_label_ids=target_label_ids,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(patch_bank, f, protocol=pickle.HIGHEST_PROTOCOL)

    print_summary(patch_bank)
    print(f"\nSaved patch bank to: {args.output}")


if __name__ == "__main__":
    main()
