#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare cerebellum mito training data: normalize TIFF volumes, "
            "convert instance masks to binary masks, and generate 3-class contour labels."
        )
    )
    parser.add_argument("--p0-dir", type=Path, required=True)
    parser.add_argument("--p7-proofreaded-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--contour-script", type=Path, required=True)
    parser.add_argument("--contour-width", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def robust_read_tiff(path: Path) -> np.ndarray:
    """Read a 3D TIFF robustly by stacking pages to avoid malformed-series issues."""
    with tifffile.TiffFile(str(path)) as tf:
        if len(tf.pages) > 1:
            pages = [page.asarray() for page in tf.pages]
            volume = np.stack(pages, axis=0)
        else:
            volume = tf.asarray()
    if volume.ndim == 2:
        volume = volume[None]
    if volume.ndim != 3:
        raise RuntimeError(f"Expected 3D volume, got shape={volume.shape} for {path}")
    return volume


def write_tiff(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr, compression="zlib")


def resolve_image_for_mask(mask_path: Path) -> Path | None:
    stem = mask_path.stem
    if not stem.endswith("_mask"):
        return None
    img_stem = stem[:-5]
    candidates = [
        mask_path.with_name(f"{img_stem}{mask_path.suffix}"),
        mask_path.with_name(f"{img_stem}.tif"),
        mask_path.with_name(f"{img_stem}.tiff"),
    ]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def discover_pairs(folder: Path, prefix: str) -> list[dict]:
    pairs = []
    for mask_path in sorted(folder.glob("*_mask.tif*")):
        img_path = resolve_image_for_mask(mask_path)
        if img_path is None:
            print(f"[WARN] cannot find matching image for mask: {mask_path}")
            continue
        case_stem = img_path.stem
        case_id = f"{prefix}_{case_stem}"
        pairs.append(
            {
                "case_id": case_id,
                "image": str(img_path),
                "instance_mask": str(mask_path),
                "source_group": prefix,
            }
        )
    return pairs


def main() -> None:
    args = parse_args()

    if not args.p0_dir.is_dir():
        raise FileNotFoundError(f"p0 dir not found: {args.p0_dir}")
    if not args.p7_proofreaded_dir.is_dir():
        raise FileNotFoundError(f"p7 proofreaded dir not found: {args.p7_proofreaded_dir}")
    if not args.contour_script.is_file():
        raise FileNotFoundError(f"contour script not found: {args.contour_script}")

    work_dir = args.work_dir
    normalized_images_dir = work_dir / "normalized" / "images"
    normalized_instance_dir = work_dir / "normalized" / "instance_masks"
    binary_masks_dir = work_dir / "binary_masks"
    contour_labels_dir = work_dir / "contour_labels"
    manifests_dir = work_dir / "manifests"

    if args.overwrite and work_dir.exists():
        shutil.rmtree(work_dir)

    for d in [normalized_images_dir, normalized_instance_dir, binary_masks_dir, contour_labels_dir, manifests_dir]:
        d.mkdir(parents=True, exist_ok=True)

    pairs = []
    pairs.extend(discover_pairs(args.p0_dir, "p0"))
    pairs.extend(discover_pairs(args.p7_proofreaded_dir, "p7"))
    if not pairs:
        raise RuntimeError("No valid image/mask pairs discovered.")

    prepared_cases: list[dict] = []
    for pair in tqdm(pairs, desc="Normalizing image/mask"):
        case_id = pair["case_id"]
        image_src = Path(pair["image"])
        mask_src = Path(pair["instance_mask"])

        image_vol = robust_read_tiff(image_src)
        inst_mask_vol = robust_read_tiff(mask_src)

        if image_vol.shape != inst_mask_vol.shape:
            raise RuntimeError(
                f"Shape mismatch for {case_id}: image {image_vol.shape} vs mask {inst_mask_vol.shape}"
            )

        binary_vol = (inst_mask_vol > 0).astype(np.uint8)

        image_out = normalized_images_dir / f"{case_id}.tiff"
        inst_out = normalized_instance_dir / f"{case_id}.tiff"
        binary_out = binary_masks_dir / f"{case_id}.tiff"

        write_tiff(image_out, image_vol)
        write_tiff(inst_out, inst_mask_vol.astype(np.uint16, copy=False))
        write_tiff(binary_out, binary_vol)

        prepared_cases.append(
            {
                **pair,
                "image_normalized": str(image_out),
                "instance_mask_normalized": str(inst_out),
                "binary_mask": str(binary_out),
                "shape_zyx": list(image_vol.shape),
                "image_dtype": str(image_vol.dtype),
                "instance_dtype": str(inst_mask_vol.dtype),
                "instance_max": int(inst_mask_vol.max()),
                "binary_positive_voxels": int(binary_vol.sum()),
            }
        )

    contour_cmd = [
        sys.executable,
        str(args.contour_script),
        "-i",
        str(normalized_instance_dir),
        "-o",
        str(contour_labels_dir),
        "-w",
        str(args.contour_width),
    ]
    print("[INFO] Running contour generation:", " ".join(contour_cmd))
    subprocess.run(contour_cmd, check=True)

    for case in prepared_cases:
        case_id = case["case_id"]
        contour_file = contour_labels_dir / f"{case_id}.tiff"
        if not contour_file.exists():
            raise RuntimeError(f"Missing contour label for case {case_id}: {contour_file}")
        contour_vol = robust_read_tiff(contour_file)
        uniq = np.unique(contour_vol)
        if not set(uniq.tolist()).issubset({0, 1, 2}):
            raise RuntimeError(f"Unexpected labels in {contour_file}: {uniq}")
        case["contour_label"] = str(contour_file)
        case["contour_unique_labels"] = [int(v) for v in uniq]

    manifest = {
        "num_cases": len(prepared_cases),
        "p0_dir": str(args.p0_dir),
        "p7_proofreaded_dir": str(args.p7_proofreaded_dir),
        "contour_width": args.contour_width,
        "cases": prepared_cases,
    }
    manifest_path = manifests_dir / "cases.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[DONE] Prepared {len(prepared_cases)} cases")
    print(f"[DONE] Manifest: {manifest_path}")
    print(f"[DONE] Normalized images: {normalized_images_dir}")
    print(f"[DONE] Binary masks: {binary_masks_dir}")
    print(f"[DONE] Contour labels: {contour_labels_dir}")


if __name__ == "__main__":
    main()
