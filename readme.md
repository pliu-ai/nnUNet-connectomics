# Modified nnU-Net v2 for Connectomics

## Installation

```bash
git clone https://github.com/luckieucas/nnUNet-connectomics.git
cd nnUNet-connectomics
conda create -n nnunet python=3.12
conda activate nnunet
pip install -e .
pip install -r requirements.txt
```

## Model Download

1. Download the [pre-trained model](https://drive.google.com/file/d/1aIDpVFUI8BspNewvPbGastzQybqtWHv-/view?usp=drive_link)
2. Unzip and place it under `./DATASET/nnUNet_trained_models/`

## Usage

### Prediction

```bash
python nnunetv2/inference/predict_mito_2d.py -i /path/to/img.tiff -o /path/to/img_prediction.tiff
```

### Semantic to Instance Segmentation

#### 1. Watershed

```bash
python nnunetv2/postprocessing/watershed.py -i /path/to/img_prediction.tiff -o /path/to/img_prediction_waterz.tiff
```

#### 2. IoU Tracking

```bash
python nnunetv2/postprocessing/iou_tracking.py -i /path/to/img_prediction.tiff -o /path/to/img_prediction_iou_tracking.tiff
```

### Evaluate Results

```bash
python nnunetv2/evaluation/evaluate_mito_pred.py --gt_file /path/to/gt.tiff --pred_file /path/to/img_prediction_iou_tracking.tiff
```