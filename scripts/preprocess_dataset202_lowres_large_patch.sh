#!/usr/bin/env bash
set -euo pipefail

NNUNET_REPO="/projects/weilab/liupeng/projects/frameworks/nnUNet"
PYTHON_BIN="/projects/weilab/liupeng/conda/envs/ssl_seg/bin/python"
NNUNET_DATA_ROOT="/projects/weilab/liupeng/data/nnUNet"
DATASET_ID="202"
TARGET_CONFIG="3d_lowres_large_patch"

export PYTHONPATH="${NNUNET_REPO}:${PYTHONPATH:-}"
export nnUNet_raw="${NNUNET_DATA_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_DATA_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_DATA_ROOT}/nnUNet_results"

DATASET_DIR="$(find "${nnUNet_raw}" -maxdepth 1 -type d -name "Dataset${DATASET_ID}_*" | head -n 1)"
if [[ -z "${DATASET_DIR}" ]]; then
  echo "Dataset ${DATASET_ID} not found under ${nnUNet_raw}" >&2
  exit 1
fi
DATASET_NAME="$(basename "${DATASET_DIR}")"
PLANS_FILE="${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans.json"

echo "[1/3] Planning dataset ${DATASET_ID} (${DATASET_NAME})"
"${PYTHON_BIN}" -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints \
  -d "${DATASET_ID}" \
  --no_pp \
  -npfp 8

echo "[2/3] Injecting ${TARGET_CONFIG} (spacing=[4,4,4], patch=[160,160,160]) into ${PLANS_FILE}"
"${PYTHON_BIN}" - "${PLANS_FILE}" <<'PY'
import json
import sys
from copy import deepcopy
from pathlib import Path

plans_file = Path(sys.argv[1])
if not plans_file.exists():
    raise FileNotFoundError(f"Plans file not found: {plans_file}")

plans = json.loads(plans_file.read_text())
configs = plans.get("configurations", {})
if "3d_lowres" not in configs:
    raise RuntimeError("3d_lowres config not found in plans; cannot derive 3d_lowres_large_patch")

cfg = deepcopy(configs["3d_lowres"])
cfg["data_identifier"] = "nnUNetPlans_3d_lowres_large_patch"
cfg["spacing"] = [4.0, 4.0, 4.0]
cfg["patch_size"] = [160, 160, 160]

orig_spacing = plans.get("original_median_spacing_after_transp")
orig_shape = plans.get("original_median_shape_after_transp")
if isinstance(orig_spacing, list) and isinstance(orig_shape, list) and len(orig_spacing) == 3 and len(orig_shape) == 3:
    cfg["median_image_size_in_voxels"] = [int(round(orig_shape[i] * orig_spacing[i] / cfg["spacing"][i])) for i in range(3)]

configs["3d_lowres_large_patch"] = cfg
plans["configurations"] = configs
plans_file.write_text(json.dumps(plans, indent=2) + "\n")
print(f"Updated {plans_file}")
print("spacing:", cfg["spacing"]) 
print("patch_size:", cfg["patch_size"]) 
print("median_image_size_in_voxels:", cfg.get("median_image_size_in_voxels"))
PY

echo "[3/3] Preprocessing ${TARGET_CONFIG} for dataset ${DATASET_ID}"
"/projects/weilab/liupeng/conda/envs/ssl_seg/bin/nnUNetv2_preprocess" \
  -d "${DATASET_ID}" \
  -plans_name "nnUNetPlans" \
  -c "${TARGET_CONFIG}" \
  -np 8 \
  --verbose

echo "Done: dataset ${DATASET_ID}, config ${TARGET_CONFIG}"
