#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build nnUNet raw dataset from prepared cerebellum work directory."
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--nnunet-data-root", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, default=302)
    parser.add_argument("--dataset-name", type=str, default="CerebellumMitoContour")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifests_path = args.work_dir / "manifests" / "cases.json"
    if not manifests_path.is_file():
        raise FileNotFoundError(f"Missing manifest from contour job: {manifests_path}")

    with manifests_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    cases = sorted(manifest.get("cases", []), key=lambda x: x["case_id"])
    if not cases:
        raise RuntimeError("No cases available in manifest.")

    dataset_folder_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    raw_root = args.nnunet_data_root / "nnUNet_raw"
    dataset_dir = raw_root / dataset_folder_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"

    if dataset_dir.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Dataset folder exists: {dataset_dir}. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(dataset_dir)

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    copied_cases = []
    for case in cases:
        case_id = case["case_id"]
        src_img = Path(case["image_normalized"])
        src_lbl = Path(case["contour_label"])

        if not src_img.is_file():
            raise FileNotFoundError(f"Missing normalized image for {case_id}: {src_img}")
        if not src_lbl.is_file():
            raise FileNotFoundError(f"Missing contour label for {case_id}: {src_lbl}")

        dst_img = images_tr / f"{case_id}_0000.tiff"
        dst_lbl = labels_tr / f"{case_id}.tiff"

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)
        copied_cases.append(case_id)

    generate_dataset_json(
        output_folder=str(dataset_dir),
        channel_names={0: "em"},
        labels={"background": 0, "mitochondria": 1, "contour": 2},
        num_training_cases=len(copied_cases),
        file_ending=".tiff",
        dataset_name=dataset_folder_name,
        description="Cerebellum mito 3-class segmentation (background/mitochondria/contour)",
        overwrite_image_reader_writer="Tiff3DIO",
    )

    summary = {
        "dataset_id": args.dataset_id,
        "dataset_folder": str(dataset_dir),
        "num_cases": len(copied_cases),
        "cases": copied_cases,
        "source_manifest": str(manifests_path),
    }

    summary_path = dataset_dir / "dataset_build_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Built nnUNet raw dataset: {dataset_dir}")
    print(f"[DONE] num_cases={len(copied_cases)}")
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
