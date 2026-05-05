# Cerebellum Mito 3-Class Training Workflow (nnUNet)

## 1. Goal
Train an nnUNet 3-class segmentation model with:
- `0`: background
- `1`: mitochondria
- `2`: contour

Input datasets:
- `p0`: `/projects/weilab/liupeng/dataset/mito/cerebellum/p0`
- `p7/proofreaded`: `/projects/weilab/liupeng/dataset/mito/cerebellum/p7/proofreaded`

nnUNet data root:
- `/projects/weilab/liupeng/projects/frameworks/nnUNet/DATASET`

## 2. Pipeline Overview
1. Read paired image + instance mask volumes.
2. Convert instance masks to binary masks.
3. Run `generate_contour.py` to produce 3-class labels (background/interior/contour).
4. Build nnUNet raw dataset `Dataset302_CerebellumMitoContour`.
5. Run `plan_and_preprocess` and then `train`.

## 3. Implemented Scripts
### Data preparation + contour generation
- `/projects/weilab/liupeng/projects/frameworks/nnUNet/scripts/cerebellum_nnunet302/prepare_cerebellum_binary_and_contour.py`

### Build nnUNet raw dataset
- `/projects/weilab/liupeng/projects/frameworks/nnUNet/scripts/cerebellum_nnunet302/build_nnunet_dataset302.py`

### Slurm scripts
- contour job: `/projects/weilab/liupeng/projects/frameworks/nnUNet/slurm/cerebellum302_contour.slurm`
- nnUNet job: `/projects/weilab/liupeng/projects/frameworks/nnUNet/slurm/cerebellum302_nnunet_train.slurm`
- chained submit script: `/projects/weilab/liupeng/projects/frameworks/nnUNet/submit_cerebellum302_pipeline.sh`

## 4. Environment Requirements
- Python: `/projects/weilab/liupeng/conda/envs/ssl_seg/bin/python`
- Required imports:
  - `tifffile`
  - `nnunetv2`
  - `connectomics.data.utils.data_segmentation` (provided by `pytorch_connectomics`)

The Slurm scripts already set:
- `PYTHONPATH=/projects/weilab/liupeng/projects/frameworks/nnUNet:/projects/weilab/liupeng/projects/frameworks/pytorch_connectomics:$PYTHONPATH`

## 5. Training Steps
### 5.1 One-command submission (recommended)
```bash
cd /projects/weilab/liupeng/projects/frameworks/nnUNet
./submit_cerebellum302_pipeline.sh
```

This submits two jobs:
1. contour generation job
2. training job (automatically starts after contour job succeeds)

### 5.2 Manual two-step submission (optional)
```bash
sbatch /projects/weilab/liupeng/projects/frameworks/nnUNet/slurm/cerebellum302_contour.slurm
sbatch --dependency=afterok:<contour_job_id> /projects/weilab/liupeng/projects/frameworks/nnUNet/slurm/cerebellum302_nnunet_train.slurm
```

## 6. Output Directories
### Intermediate work directory
- `/projects/weilab/liupeng/projects/frameworks/nnUNet/DATASET/cerebellum302_work/normalized/images`
- `/projects/weilab/liupeng/projects/frameworks/nnUNet/DATASET/cerebellum302_work/normalized/instance_masks`
- `/projects/weilab/liupeng/projects/frameworks/nnUNet/DATASET/cerebellum302_work/binary_masks`
- `/projects/weilab/liupeng/projects/frameworks/nnUNet/DATASET/cerebellum302_work/contour_labels`
- `/projects/weilab/liupeng/projects/frameworks/nnUNet/DATASET/cerebellum302_work/manifests/cases.json`

### nnUNet raw dataset
- `/projects/weilab/liupeng/projects/frameworks/nnUNet/DATASET/nnUNet_raw/Dataset302_CerebellumMitoContour`
  - `imagesTr/*_0000.tiff`
  - `labelsTr/*.tiff`
  - `dataset.json`

### Training outputs
- preprocessed: `/projects/weilab/liupeng/projects/frameworks/nnUNet/DATASET/nnUNet_preprocessed/Dataset302_CerebellumMitoContour`
- trained model: `/projects/weilab/liupeng/projects/frameworks/nnUNet/DATASET/nnUNet_trained_models/Dataset302_CerebellumMitoContour`

## 7. Monitoring and Logs
### Queue status
```bash
squeue -u liupen
```

### Contour job logs
```bash
tail -f /projects/weilab/liupeng/projects/frameworks/nnUNet/logs/slurm/cereb302_contour-<jobid>.out
tail -f /projects/weilab/liupeng/projects/frameworks/nnUNet/logs/slurm/cereb302_contour-<jobid>.err
```

### Training job logs
```bash
tail -f /projects/weilab/liupeng/projects/frameworks/nnUNet/logs/slurm/cereb302_train-<jobid>.out
tail -f /projects/weilab/liupeng/projects/frameworks/nnUNet/logs/slurm/cereb302_train-<jobid>.err
```

## 8. Common Issues
### Issue 1: `ModuleNotFoundError: No module named 'connectomics'`
Cause: `generate_contour.py` depends on `pytorch_connectomics`, but it is missing from `PYTHONPATH`.  
Fix:
```bash
export PYTHONPATH="/projects/weilab/liupeng/projects/frameworks/nnUNet:/projects/weilab/liupeng/projects/frameworks/pytorch_connectomics:${PYTHONPATH:-}"
```

### Issue 2: TIFF read errors
The preparation script reads TIFFs page-by-page and rewrites normalized TIFF volumes, which avoids failures caused by non-standard TIFF series metadata in the original files.

## 9. Reuse for New Datasets
Update these variables:
- `slurm/cerebellum302_contour.slurm`
  - `P0_DIR`
  - `P7_PROOFREADED_DIR`
  - `WORK_DIR`
- `slurm/cerebellum302_nnunet_train.slurm`
  - `DATASET_ID`
  - `DATASET_NAME`
  - `CONFIG`
  - `TRAINER`

Use a new `DATASET_ID` each time to avoid overwriting previous runs.
