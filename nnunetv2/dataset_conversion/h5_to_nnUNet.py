#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert paired <id>_im.h5 / <id>_mito.h5 files to nnUNet format (NIfTI),
and generate instance contour map for mito labels.
"""

import argparse
from pathlib import Path
import json
import re
import h5py
import numpy as np
import SimpleITK as sitk
from connectomics.data.utils.data_segmentation import seg_to_instance_bd


def parse_args():
    parser = argparse.ArgumentParser(description="H5 → nnUNet converter with contour (SimpleITK)")
    parser.add_argument("--in_dir", required=True, type=Path,
                        help="Folder containing paired H5 files")
    parser.add_argument("--out_dir", required=True, type=Path,
                        help="Output task folder (e.g. Dataset501_Mito)")
    parser.add_argument("--resolution", nargs=3, required=True, type=float,
                        metavar=("Xnm", "Ynm", "Znm"),
                        help="Voxel size in nanometers, e.g. 10 10 30")
    parser.add_argument("--dataset_key", default="main",
                        help="HDF5 dataset key holding the array")
    parser.add_argument("--as_test", action="store_true",
                        help="If set, write to imagesTs (no labels)")
    return parser.parse_args()


def nm_to_mm(res_nm):
    """Convert spacing from nanometers to millimeters."""
    return [x * 1e-6 for x in res_nm]


def save_nifti_sitk(array, spacing_mm, filepath, dtype=np.float32):
    """
    Save a numpy array as NIfTI via SimpleITK.
    - array: Z,Y,X or Z,Y,X channels
    - spacing_mm: [sx, sy, sz]
    """
    img = sitk.GetImageFromArray(array.astype(dtype))
    img.SetSpacing(tuple(spacing_mm))
    sitk.WriteImage(img, str(filepath), useCompression=True)


def main():
    args = parse_args()
    img_dir = args.out_dir / ("imagesTs" if args.as_test else "imagesTr")
    lbl_dir = args.out_dir / "labelsTr"

    img_dir.mkdir(parents=True, exist_ok=True)
    if not args.as_test:
        lbl_dir.mkdir(parents=True, exist_ok=True)

    spacing_mm = nm_to_mm(args.resolution)
    pattern = re.compile(r"(.+)_im\.h5$")
    pair_count = 0
    print(f"[INFO] Converting H5 files in {args.in_dir} to nnUNet format...")
    for im_path in sorted(args.in_dir.glob("*_im.h5")):
        match = pattern.match(im_path.name)
        print(f"[INFO] Processing {im_path.name}..., match: {match is not None}")
        if not match:
            continue
        case_id = match.group(1)
        mito_path = im_path.with_name(f"{case_id}_mito.h5")
        if not mito_path.exists():
            print(f"[WARNING] Missing label file for {case_id}; skipping.")
            continue

        # --- read image & label ---
        with h5py.File(im_path, "r") as f_im, h5py.File(mito_path, "r") as f_lb:
            img = f_im[args.dataset_key][()]
            lbl = f_lb[args.dataset_key][()]

        # if image has channel axis, squeeze
        if img.ndim == 4:
            img = img[0]

        # --- save image NIfTI ---
        img_out = img_dir / f"{case_id}_0000.nii.gz"
        save_nifti_sitk(img, spacing_mm, img_out, dtype=np.float32)

        # --- generate contour for mito and save label NIfTI ---
        if not args.as_test:
            # binary mask
            binary = (lbl > 0).astype(np.uint8)

            # ==================== MODIFICATION START ====================
            # Generate boundary from all three dimensions by transposing the array
            
            # 1. Boundary from Z-axis slices (original orientation Z,Y,X)
            contour_z = seg_to_instance_bd(binary, tsz_h=3)
            
            # 2. Boundary from Y-axis slices (transpose to Y,Z,X)
            binary_y_sliced = binary.transpose(1, 0, 2)
            contour_y_transposed = seg_to_instance_bd(binary_y_sliced, tsz_h=3)
            contour_y = contour_y_transposed.transpose(1, 0, 2)
            
            # 3. Boundary from X-axis slices (transpose to X,Y,Z)
            binary_x_sliced = binary.transpose(2, 1, 0)
            contour_x_transposed = seg_to_instance_bd(binary_x_sliced, tsz_h=3)
            contour_x = contour_x_transposed.transpose(2, 1, 0)
            
            # 4. Combine boundaries from all three axes
            # Add them up and convert to a single binary mask for the final contour
            contour = (contour_z + contour_y + contour_x).astype(bool).astype(np.uint8)
            
            # Set boundary value to 2
            contour[contour > 0] = 2
            
            # Combine: interior=1, contour=2.
            # Where they overlap (value=3), prioritize interior by setting it back to 1.
            combined = binary + contour
            combined[combined > 2] = 1
            # ===================== MODIFICATION END =====================

            lbl_out = lbl_dir / f"{case_id}.nii.gz"
            save_nifti_sitk(combined, spacing_mm, lbl_out, dtype=np.uint8)

        pair_count += 1
        print(f"[OK] Converted {case_id}")

    if pair_count == 0:
        raise RuntimeError("No valid H5 pairs found.")

    # --- write dataset.json ---
    # (The rest of the function remains unchanged)
    imgs = sorted(img_dir.glob("*_0000.nii.gz"))
    dataset_json = {
        "name": args.out_dir.name,
        "description": "Converted from H5 with 3D contour (SimpleITK)",
        "tensorImageSize": "3D",
        "reference": "",
        "licence": "",
        "release": "0.0",
        "modality": {"0": "SEM"},
        "labels": {"0": "background", "1": "mitochondria", "2": "boundary"},
        "numTraining": 0 if args.as_test else pair_count,
        "numTest": pair_count if args.as_test else 0,
        "training": [] if args.as_test else [
            {"image": f"./imagesTr/{p.stem}_0000.nii.gz",
             "label": f"./labelsTr/{p.stem}.nii.gz"}
            for p in imgs
        ],
        "test": [] if not args.as_test else [
            f"./imagesTs/{p.stem}_0000.nii.gz" for p in imgs
        ]
    }
    with open(args.out_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"[DONE] {pair_count} pairs converted. dataset.json saved.")


if __name__ == "__main__":
    main()
