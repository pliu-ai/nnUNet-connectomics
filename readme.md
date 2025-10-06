# modified nnU-Net v2 for connectomics
## Install
```bash
git clone https://github.com/luckieucas/nnUNet-connectomics.git
cd nnUNet-connectomics
conda create -n nnunet python=3.12
pip install -e .
pip install -r requirements.txt
```

## Model download
### download [model][https://drive.google.com/file/d/1aIDpVFUI8BspNewvPbGastzQybqtWHv-/view?usp=drive_link]

### unzip and put under the ./DATASET/nnUNet_trained_models/

## Usage


### predict
```bash
python nnunetv2/inference/predict_mito_2d.py -i /path/to/img.tiff -o /path/to/img_prediction.tiff
```
### semantic to instance
1. watersetd
```bash
python nnunetv2/postprocessing/watershed.py -i /path/to/img_prediction.tiff -o /path/to/img_prediction_waterz.tiff
```

2. Iou tracking
```bash
python nnunetv2/postprocessing/iou_tracking.py -i /path/to/img_prediction.tiff -o /path/to/img_prediction_iou_tracking.tiff
```
### evaluate results
```bash
python 
```