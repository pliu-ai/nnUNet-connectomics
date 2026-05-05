#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import tifffile
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict cerebellum p7 *_im.h5 with trained nnUNet Dataset302 model."
    )
    parser.add_argument("--input-h5-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", type=str, default="302")
    parser.add_argument("--config", type=str, default="3d_fullres")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainer")
    parser.add_argument("--plans", type=str, default="nnUNetPlans")
    parser.add_argument("--folds", nargs="+", default=["all"])
    parser.add_argument("--checkpoint", type=str, default="checkpoint_final.pth")
    parser.add_argument("--dataset-key", type=str, default="main")
    parser.add_argument("--nnunet-predict-bin", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--npp", type=int, default=1)
    parser.add_argument("--nps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_h5_volume(h5_path: Path, dataset_key: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"Dataset key '{dataset_key}' not found in {h5_path}. Available: {list(f.keys())}")
        vol = f[dataset_key][()]
    if vol.ndim == 4:
        if vol.shape[0] == 1:
            vol = vol[0]
        else:
            raise RuntimeError(f"Unsupported 4D shape for {h5_path}: {vol.shape}")
    if vol.ndim != 3:
        raise RuntimeError(f"Expected 3D volume for {h5_path}, got shape={vol.shape}")
    return vol


def main() -> None:
    args = parse_args()

    input_h5_dir = args.input_h5_dir
    output_dir = args.output_dir

    if not input_h5_dir.is_dir():
        raise FileNotFoundError(f"Input H5 folder not found: {input_h5_dir}")
    if not args.nnunet_predict_bin.is_file():
        raise FileNotFoundError(f"nnUNet predict binary not found: {args.nnunet_predict_bin}")

    cases = sorted(input_h5_dir.glob("*_im.h5"))
    if not cases:
        raise RuntimeError(f"No *_im.h5 files found in {input_h5_dir}")

    input_tiff_dir = output_dir / "input_tiff"
    pred_tiff_dir = output_dir / "pred_tiff"
    pred_h5_dir = output_dir / "pred_h5"

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    input_tiff_dir.mkdir(parents=True, exist_ok=True)
    pred_tiff_dir.mkdir(parents=True, exist_ok=True)
    pred_h5_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "input_h5_dir": str(input_h5_dir),
        "output_dir": str(output_dir),
        "dataset_id": args.dataset_id,
        "config": args.config,
        "trainer": args.trainer,
        "plans": args.plans,
        "folds": args.folds,
        "checkpoint": args.checkpoint,
        "dataset_key": args.dataset_key,
        "save_probabilities": bool(args.save_probabilities),
        "num_cases": len(cases),
        "cases": [],
    }

    print(f"[INFO] Converting {len(cases)} H5 files to nnUNet TIFF input...")
    for h5_path in tqdm(cases, desc="H5 -> TIFF"):
        case_id = h5_path.name[:-6]  # strip '_im.h5'
        vol = load_h5_volume(h5_path, args.dataset_key)
        tiff_path = input_tiff_dir / f"{case_id}_0000.tiff"
        tifffile.imwrite(str(tiff_path), vol.astype(np.uint8, copy=False), compression="zlib")
        manifest["cases"].append(
            {
                "case_id": case_id,
                "input_h5": str(h5_path),
                "input_tiff": str(tiff_path),
                "shape_zyx": list(vol.shape),
                "dtype": str(vol.dtype),
            }
        )

    predict_cmd = [
        str(args.nnunet_predict_bin),
        "-i",
        str(input_tiff_dir),
        "-o",
        str(pred_tiff_dir),
        "-d",
        args.dataset_id,
        "-c",
        args.config,
        "-tr",
        args.trainer,
        "-p",
        args.plans,
        "-f",
        *args.folds,
        "-chk",
        args.checkpoint,
        "-npp",
        str(args.npp),
        "-nps",
        str(args.nps),
        "-device",
        args.device,
        "--disable_progress_bar",
    ]
    if args.save_probabilities:
        predict_cmd.append("--save_probabilities")
    print("[INFO] Running prediction:")
    print(" ".join(predict_cmd))
    subprocess.run(predict_cmd, check=True)

    print("[INFO] Converting predicted TIFF -> H5...")
    for item in tqdm(manifest["cases"], desc="TIFF -> H5"):
        case_id = item["case_id"]
        pred_tiff = pred_tiff_dir / f"{case_id}.tiff"
        if not pred_tiff.exists():
            # fallback in case output ending is .tif
            alt = pred_tiff_dir / f"{case_id}.tif"
            if alt.exists():
                pred_tiff = alt
            else:
                raise FileNotFoundError(f"Prediction not found for {case_id} in {pred_tiff_dir}")

        pred_vol = tifffile.imread(str(pred_tiff)).astype(np.uint8, copy=False)
        pred_h5 = pred_h5_dir / f"{case_id}_pred.h5"
        with h5py.File(pred_h5, "w") as f:
            f.create_dataset("main", data=pred_vol, compression="gzip")

        item["pred_tiff"] = str(pred_tiff)
        item["pred_h5"] = str(pred_h5)
        item["pred_unique_labels"] = [int(v) for v in np.unique(pred_vol)]
        if args.save_probabilities:
            prob_npz = pred_tiff_dir / f"{case_id}.npz"
            prob_pkl = pred_tiff_dir / f"{case_id}.pkl"
            if prob_npz.exists():
                item["probability_npz"] = str(prob_npz)
            if prob_pkl.exists():
                item["probability_pkl"] = str(prob_pkl)

    summary_path = output_dir / "prediction_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[DONE] Predicted {len(cases)} cases")
    print(f"[DONE] Outputs: {output_dir}")
    print(f"[DONE] Summary: {summary_path}")


if __name__ == "__main__":
    main()
